from mechanism import BaseMechanism, TrainingResults
from datasets import BaseDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Opacus imports for differential privacy
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator

from util.privacy import PrivacyBudget
from sklearn.metrics import roc_auc_score


class DPSGDMechanism(BaseMechanism):
    """
    A mechanism that trains a model using DP-SGD via the Opacus library.
    
    The privacy budget (ε, δ) is managed automatically by Opacus's privacy engine,
    which tracks privacy expenditure across all training steps.
    """
    
    def __init__(self, model_constructor, dataset: BaseDataset, privacy_budget: PrivacyBudget):
        super().__init__(model_constructor, dataset, privacy_budget)
        print(f"DPSGD Mechanism initialized with privacy budget: ε={privacy_budget.epsilon}, δ={privacy_budget.delta}")
    
    def _setup_model_and_device(self):
        """Initialize model and set device."""
        model = self.model_constructor()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Training on device: {device}")
        return model, device
    
    def _extract_hyperparameters(self, **kwargs):
        """Extract and validate hyperparameters with DP-specific defaults."""
        config = {
            'num_epochs': kwargs.get('num_epochs', 100),
            'learning_rate': kwargs.get('learning_rate', 0.01),
            'batch_size': kwargs.get('batch_size', 32),
            'max_grad_norm': kwargs.get('max_grad_norm', 1.0),  # DP-specific
            'patience': kwargs.get('patience', 20)  # Reduced for DP training
        }
        
        epsilon, delta = self.privacy_budget.epsilon, self.privacy_budget.delta
        print(f"Target privacy budget: ε={epsilon}, δ={delta}")

        self._log_training_config(config)
        
        return config, epsilon, delta
    
    def _log_training_config(self, config):
        """Log training configuration."""
        print("Training configuration:")
        print(f"  Epochs: {config['num_epochs']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Max gradient norm (clipping): {config['max_grad_norm']}")
        print(f"  Early stopping patience: {config['patience']}")
    
    def _setup_data_loaders(self, config):
        """Create data loaders with DP-specific requirements."""
        train_dataset, val_dataset, test_dataset = self.dataset.to_torch(include_protected=False)
        
        # DP requires drop_last=True for consistent batch sizes
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['batch_size'], 
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        self._log_dataset_info(train_dataset, val_dataset, test_dataset, train_loader)
        
        return train_loader, val_loader
    
    def _log_dataset_info(self, train_dataset, val_dataset, test_dataset, train_loader):
        """Log dataset split information."""
        print("Dataset splits:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Training batches: {len(train_loader)}")
    
    def _validate_model_for_opacus(self, model):
        """Validate and fix model for Opacus compatibility."""
        try:
            model = ModuleValidator.fix(model)
            print("Model validated and fixed for Opacus compatibility")
        except Exception as e:
            print(f"Warning: Model validation failed: {e}")
            print("Proceeding anyway, but may encounter issues...")
        return model
    
    def _setup_optimizer_and_criterion(self, model, learning_rate):
        """Setup optimizer and loss function for DP training."""
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.BCELoss(reduction='none')  # No reduction for per-sample losses
        return optimizer, criterion
    
    def _setup_privacy_engine(self, model, optimizer, train_loader, config, epsilon, delta):
        """Initialize and configure the privacy engine."""
        privacy_engine = PrivacyEngine()
        
        # Transform model, optimizer, and data loader for DP
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=config['num_epochs'],
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=config['max_grad_norm'],
        )
        
        self._log_privacy_setup(privacy_engine, optimizer)
        
        return privacy_engine, model, optimizer, train_loader
    
    def _log_privacy_setup(self, privacy_engine, optimizer):
        """Log privacy engine configuration."""
        print(f"Opacus computed noise multiplier: {optimizer.noise_multiplier:.4f}")
        print(f"Privacy accountant: {privacy_engine.accountant.__class__.__name__}")
    
    def _process_batch_dp(self, batch_X, batch_y, device):
        """Process batch for DP training with proper tensor formatting."""
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        batch_y = batch_y.view(-1, 1).float()
        return batch_X, batch_y
    
    def _train_epoch_dp(self, model, train_loader, optimizer, criterion, device, batch_size):
        """Execute one DP training epoch."""
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # Use BatchMemoryManager for memory efficiency
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=batch_size,
            optimizer=optimizer
        ) as memory_safe_data_loader:
            
            for batch_X, batch_y in memory_safe_data_loader:
                batch_X, batch_y = self._process_batch_dp(batch_X, batch_y, device)
                
                optimizer.zero_grad()
                
                # Forward pass
                logits, outputs = model(batch_X)
                
                # Compute per-sample losses (no reduction)
                per_sample_losses = criterion(outputs, batch_y)
                loss = per_sample_losses.mean()  # Average for backprop
                
                # Backward pass with DP
                loss.backward()
                optimizer.step()
                
                # Accumulate statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
        
        avg_loss = train_loss / len(train_loader)
        accuracy = train_correct / train_total
        return avg_loss, accuracy
    
    def _validate_epoch_dp(self, model, val_loader, device):
        """Execute one validation epoch (no DP needed)."""
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_criterion = nn.BCELoss()  # Standard loss for validation
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = self._process_batch_dp(batch_X, batch_y, device)
                
                logits, outputs = model(batch_X)
                loss = val_criterion(outputs, batch_y)
                
                # Accumulate statistics
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        avg_loss = val_loss / len(val_loader)
        accuracy = val_correct / val_total
        return avg_loss, accuracy
    
    def _check_privacy_budget_exhausted(self, current_epsilon, target_epsilon):
        """Check if privacy budget has been exhausted."""
        if current_epsilon > target_epsilon:
            print(f"\nPrivacy budget exhausted! Current ε={current_epsilon:.4f} > target ε={target_epsilon}")
            return True
        return False
    
    def _update_early_stopping(self, val_loss, best_val_loss, patience_counter, model):
        """Update early stopping variables and return new state."""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            return best_val_loss, patience_counter, best_model_state
        else:
            return best_val_loss, patience_counter + 1, None
    
    def _log_epoch_progress_dp(self, epoch, num_epochs, train_loss, train_acc, val_loss, val_acc, current_epsilon):
        """Log training progress with privacy information."""
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | '
                  f'Privacy: ε={current_epsilon:.4f}')
    
    def _log_final_privacy_stats(self, privacy_engine, delta, epsilon):
        """Log final privacy expenditure statistics."""
        final_epsilon = privacy_engine.get_epsilon(delta)
        print(f"\nTraining completed!")
        print(f"Final privacy expenditure: ε={final_epsilon:.4f}, δ={delta}")
        print(f"Privacy budget utilization: {(final_epsilon/epsilon)*100:.1f}%")
        return final_epsilon
    
    def _evaluate_final_model_dp(self, model, val_dataset, device, batch_size):
        """Evaluate final model and return metrics."""
        model.eval()
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        val_total, val_correct = 0, 0
        all_outputs, all_labels = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = self._process_batch_dp(batch_X, batch_y, device)
                
                _, outputs = model(batch_X)
                predicted = (outputs > 0.5).float()
                
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
                
                all_outputs.extend(outputs.squeeze().cpu().numpy().tolist())
                all_labels.extend(batch_y.squeeze().cpu().numpy().tolist())
        
        val_accuracy = val_correct / val_total
        val_auroc = roc_auc_score(all_labels, all_outputs)
        return val_auroc, val_accuracy
    
    def train(self, **kwargs) -> TrainingResults:
        """
        Trains the model using DP-SGD.

        Args:
            **kwargs: Additional hyperparameters for training, such as:
                - num_epochs (int): Number of training epochs.
                - learning_rate (float): Learning rate for the optimizer.
                - batch_size (int): Size of each training batch.
                - max_grad_norm (float): Maximum gradient norm for clipping.
                - patience (int): Early stopping patience.
        Returns:
            TrainingResults: A structured object containing training metrics and hyperparameters.
        """
        # Setup phase
        model, device = self._setup_model_and_device()
        config, epsilon, delta = self._extract_hyperparameters(**kwargs)
        train_loader, val_loader = self._setup_data_loaders(config)
        
        # Model validation and DP setup
        model = self._validate_model_for_opacus(model)
        optimizer, criterion = self._setup_optimizer_and_criterion(model, config['learning_rate'])
        privacy_engine, model, optimizer, train_loader = self._setup_privacy_engine(
            model, optimizer, train_loader, config, epsilon, delta
        )
        
        # Training tracking variables
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        privacy_spent = []
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("\nStarting DP-SGD training...")
        print("Note: DP training is typically slower due to per-sample gradient computation")
        
        # Training loop
        for epoch in range(config['num_epochs']):
            # Train and validate
            train_loss, train_acc = self._train_epoch_dp(
                model, train_loader, optimizer, criterion, device, config['batch_size']
            )
            val_loss, val_acc = self._validate_epoch_dp(model, val_loader, device)
            
            # Get current privacy expenditure
            current_epsilon = privacy_engine.get_epsilon(delta)
            
            # Store metrics
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            privacy_spent.append(current_epsilon)
            
            # Log progress
            self._log_epoch_progress_dp(epoch, config['num_epochs'], train_loss, train_acc, 
                                       val_loss, val_acc, current_epsilon)
            
            # Check privacy budget
            if self._check_privacy_budget_exhausted(current_epsilon, epsilon):
                break
            
            # Early stopping logic
            best_val_loss, patience_counter, new_best_state = self._update_early_stopping(
                val_loss, best_val_loss, patience_counter, model
            )
            
            if new_best_state is not None:
                best_model_state = new_best_state
            
            if patience_counter >= config['patience']:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f}')
                break
        
        # Restore best model and finalize
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Restored best model weights")
        
        self._log_final_privacy_stats(privacy_engine, delta, epsilon)
        
        # Final evaluation
        _, val_dataset, _ = self.dataset.to_torch(include_protected=False)
        val_auroc, val_accuracy = self._evaluate_final_model_dp(model, val_dataset, device, config['batch_size'])
        
        # Save trained model
        self.model = model
        
        return TrainingResults(
            auroc_score=val_auroc,
            accuracy=val_accuracy,
            mechanism_name="DPSGD",
            hyperparameters={
                **config,
                'noise_multiplier': optimizer.noise_multiplier
            }
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