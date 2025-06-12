from dataclasses import asdict

import optuna
import torch
import torch.nn as nn
from mechanisms.mechanism import BaseHyperparameters, BaseMechanism, TrainingResults
from torch.utils.data import DataLoader

from datasets.dataset import BaseDataset


class SGDMechanism(BaseMechanism):
    """
    A mechanism that trains a model using SGD, designed for fairness research.
    
    This implementation includes proper early stopping, comprehensive logging,
    and handles the dual-output nature of our MLP model.
    """
    
    def __init__(self, model_constructor, dataset: BaseDataset):
        super().__init__(model_constructor, dataset)
        print("Initialized SGDMechanism")

    def _setup_training(self, hyperparameters: BaseHyperparameters, device: str):
        """Setup model, device, data loaders, and hyperparameters."""
        # Model and device setup
        model = self.model_constructor()
        model.to(device)
        print(f"Training on device: {device}")
        
        # Dataset setup
        train_dataset, val_dataset, test_dataset = self.dataset.to_torch()
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=hyperparameters.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=hyperparameters.batch_size, shuffle=False)
        
        self._log_setup(hyperparameters, train_dataset, val_dataset, test_dataset)
        
        return model, device, train_loader, val_loader

    def _log_setup(self, hyperparameters, train_dataset, val_dataset, test_dataset):
        """Log training configuration and dataset information."""
        print(f"Training configuration:")
        for key, value in asdict(hyperparameters).items():
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
    
    def _train_epoch(self, model, train_loader, optimizer, criterion, device):
        """Execute one training epoch and return metrics."""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            batch_y = batch_y.view(-1, 1)  # Ensure correct shape for BCELoss
            
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
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_y = batch_y.view(-1, 1)  # Ensure correct shape for BCELoss
                
                logits, outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Accumulate statistics
                total_loss += loss.item()
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def _should_stop_early(self, val_loss, best_val_loss, patience_counter):
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
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_y = batch_y.view(-1, 1)
                
                _, outputs = model(batch_X)
                predicted = (outputs > 0.5).float()
                
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)
                
                all_outputs.extend(outputs.squeeze().cpu().numpy().tolist())
                all_labels.extend(batch_y.squeeze().cpu().numpy().tolist())
        
        accuracy = correct / total

        return accuracy
    
    def train(self, hyperparameters: BaseHyperparameters, device: str) -> TrainingResults:
        """
        Trains the model using Stochastic Gradient Descent (SGD).
        Args:
            hyperparameters (BaseHyperparameters): Hyperparameters for training.
                - num_epochs (int): Number of training epochs.
                - learning_rate (float): Learning rate for the optimizer.
                - batch_size (int): Size of each training batch.
                - patience (int): Early stopping patience.
            device (str): Device to run the training on (e.g., 'cpu' or 'cuda').
        Returns:
            TrainingResults: Contains AUROC score, accuracy, mechanism name, and hyperparameters.
        """
        # Setup phase
        model, device, train_loader, val_loader = self._setup_training(hyperparameters, device)
        optimizer, criterion = self._setup_optimizer_and_criterion(model, hyperparameters.learning_rate)

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
        for epoch in range(hyperparameters.n_epochs):
            # Train and validate
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = self._validate_epoch(model, val_loader, criterion, device)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            
            # Log progress
            self._log_epoch_progress(epoch, hyperparameters.n_epochs, train_loss, train_acc, val_loss, val_acc)
            
            # Early stopping logic
            best_val_loss, patience_counter, should_save = self._should_stop_early(
                val_loss, best_val_loss, patience_counter,
            )
            
            if should_save:
                best_model_state = model.state_dict().copy()
                print(f'New best validation loss: {best_val_loss:.4f}')

            if patience_counter >= hyperparameters.patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f}')
                break
        
        # Restore best model and evaluate
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Restored best model weights from validation phase")
        
        # Final evaluation
        _, val_dataset, _ = self.dataset.to_torch()
        val_accuracy = self._evaluate_final_model(model, val_dataset, device, hyperparameters.batch_size)
        
        # Save trained model
        self.model = model
        
        return TrainingResults(
            accuracy=val_accuracy,
            mechanism_name="SGD",
            hyperparameters=hyperparameters
        )
    
    def predict(self, device: str):
        """
        Generate predictions on the test set using the trained model.
        Returns:
            List of prediction probabilities.
        """
        _, _, test_dataset = self.dataset.to_torch()
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
        
        self.model.eval()
        predictions = []
        true_labels = []
        protected_attrs = []
        
        with torch.no_grad():
            for batch_X, t, p in test_loader:
                batch_X = batch_X.to(device)
                _, outputs = self.model(batch_X)
                outputs = (outputs > 0.5).float()
                predictions.extend(outputs.squeeze().cpu().numpy().tolist())
                true_labels.extend(t.squeeze().cpu().numpy().tolist())
                protected_attrs.extend(p.squeeze().cpu().numpy().tolist())

        return predictions, true_labels, protected_attrs

    def save(self, path: str):
        """
        Save the trained model to a file.
        
        Args:
            path (str): The file path where the model will be saved.
        """
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """
        Load a trained model from a file.
        
        Args:
            path (str): The file path from which the model will be loaded.
        """
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print(f"Model loaded from {path}")
    
    def suggest_hyperparameters(self, trial: optuna.Trial) -> BaseHyperparameters:
        """
        Suggest hyperparameters for the mechanism based on the given trial.
        
        Args:
            trial (optuna.Trial): The Optuna trial object for hyperparameter optimization.
        
        Returns:
            BaseHyperparameters: Suggested hyperparameters for training.
        """
        n_epochs = trial.suggest_int("n_epochs", 10, 100)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        patience = trial.suggest_int("patience", 5, 20)
        
        return BaseHyperparameters(
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience
        )
