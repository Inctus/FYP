import torch
import torch.nn as nn
from mechanism import BaseHyperparameters, BaseMechanism, TrainingResults
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from datasets import BaseDataset
from util.privacy import PrivacyBudget


class SGDMechanism(BaseMechanism):
    """
    A mechanism that trains a model using SGD, designed for fairness research.
    
    This implementation includes proper early stopping, comprehensive logging,
    and handles the dual-output nature of our MLP model.
    """
    
    def __init__(self, model_constructor, dataset: BaseDataset, privacy_budget: PrivacyBudget):
        super().__init__(model_constructor, dataset, privacy_budget)
        print("Initialized SGDMechanism")

    def _setup_training(self, hyperparameters: BaseHyperparameters):
        """Setup model, device, data loaders, and hyperparameters."""
        # Model and device setup
        model = self.model_constructor()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Training on device: {device}")
        
        # Dataset setup
        train_dataset, val_dataset, test_dataset = self.dataset.to_torch(include_protected=False)
        
        # Hyperparameters with defaults
        config = {
            'num_epochs': kwargs.get('num_epochs', 100),
            'learning_rate': kwargs.get('learning_rate', 0.01),
            'batch_size': kwargs.get('batch_size', 32),
            'patience': kwargs.get('patience', 100)
        }
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        self._log_setup(config, train_dataset, val_dataset, test_dataset)
        
        return model, device, train_loader, val_loader, config
    
    def _log_setup(self, config, train_dataset, val_dataset, test_dataset):
        """Log training configuration and dataset information."""
        print(f"Training configuration:")
        for key, value in config.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print(f"Dataset splits:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
    
    def _setup_optimizer_and_criterion(self, model, learning_rate):
        """Setup optimizer and loss function."""
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.BCELoss()
        return optimizer, criterion
    
    def _process_batch(self, batch_X, batch_y, model, device):
        """Process a single batch and return formatted inputs."""
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_y = batch_y.view(-1, 1)  # Ensure correct shape for BCELoss
        return batch_X, batch_y
    
    def _train_epoch(self, model, train_loader, optimizer, criterion, device):
        """Execute one training epoch and return metrics."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = self._process_batch(batch_X, batch_y, model, device)
            
            optimizer.zero_grad()
            logits, outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            # Accumulate statistics
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _validate_epoch(self, model, val_loader, criterion, device):
        """Execute one validation epoch and return metrics."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = self._process_batch(batch_X, batch_y, model, device)
                
                hidden_repr, outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Accumulate statistics
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _should_stop_early(self, val_loss, best_val_loss, patience_counter, patience):
        """Check early stopping conditions and update tracking variables."""
        if val_loss < best_val_loss:
            return val_loss, 0, True  # new_best_loss, new_patience_counter, should_save
        else:
            return best_val_loss, patience_counter + 1, False
    
    def _log_epoch_progress(self, epoch, num_epochs, train_loss, train_acc, val_loss, val_acc):
        """Log training progress for specific epochs."""
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    def _evaluate_final_model(self, model, val_dataset, device, batch_size):
        """Evaluate the final model and return AUROC and accuracy."""
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        correct = 0
        total = 0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = self._process_batch(batch_X, batch_y, model, device)
                
                _, outputs = model(batch_X)
                predicted = (outputs > 0.5).float()
                
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                
                all_outputs.extend(outputs.squeeze().cpu().numpy().tolist())
                all_labels.extend(batch_y.squeeze().cpu().numpy().tolist())
        
        accuracy = correct / total
        auroc = roc_auc_score(all_labels, all_outputs)
        return auroc, accuracy
    
    def train(self, **kwargs) -> TrainingResults:
        """
        Trains the model using Stochastic Gradient Descent (SGD).
        Args:
            **kwargs: Additional hyperparameters for training.
                - num_epochs (int): Number of training epochs.
                - learning_rate (float): Learning rate for the optimizer.
                - batch_size (int): Size of each training batch.
                - patience (int): Early stopping patience.
        Returns:
            TrainingResults: Contains AUROC score, accuracy, mechanism name, and hyperparameters.
        """
        # Setup phase
        model, device, train_loader, val_loader, config = self._setup_training(**kwargs)
        optimizer, criterion = self._setup_optimizer_and_criterion(model, config['learning_rate'])
        
        # Training tracking variables
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("\nStarting training...")
        
        # Training loop
        for epoch in range(config['num_epochs']):
            # Train and validate
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion, device)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Log progress
            self._log_epoch_progress(epoch, config['num_epochs'], train_loss, train_acc, val_loss, val_acc)
            
            # Early stopping logic
            best_val_loss, patience_counter, should_save = self._should_stop_early(
                val_loss, best_val_loss, patience_counter, config['patience']
            )
            
            if should_save:
                best_model_state = model.state_dict().copy()
                print(f'New best validation loss: {best_val_loss:.4f}')
            
            if patience_counter >= config['patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f}')
                break
        
        # Restore best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Restored best model weights from validation phase")
        
        # Final evaluation
        _, val_dataset, _ = self.dataset.to_torch(include_protected=False)
        val_auroc, val_accuracy = self._evaluate_final_model(model, val_dataset, device, config['batch_size'])
        
        # Save trained model
        self.model = model
        
        return TrainingResults(
            auroc_score=val_auroc,
            accuracy=val_accuracy,
            mechanism_name="SGD",
            hyperparameters=config
        )
    
    def predict(self):
        """
        Generate predictions on the test set using the trained model.
        Returns:
            List of prediction probabilities.
        """
        device = next(self.model.parameters()).device
        _, _, test_dataset = self.dataset.to_torch(include_protected=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(device)
                _, outputs = self.model(batch_X)
                predictions.extend(outputs.squeeze().cpu().numpy().tolist())
        
        return predictions