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
        print("Initialized DPSGDMechanism using Opacus for privacy-preserving training")
    
    def train(self, **kwargs) -> TrainingResults:
        # Instantiate model and set device
        model = self.model_constructor()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Training on device: {device}")
        
        # Extract target privacy budget from stored object
        epsilon, delta = self.privacy_budget.epsilon, self.privacy_budget.delta
        print(f"Target privacy budget: ε={epsilon}, δ={delta}")
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = self.dataset.to_torch(include_protected=False)
        
        # Validate privacy parameters
        if epsilon <= 0 or delta <= 0 or delta >= 1:
            raise ValueError("Privacy budget must satisfy ε > 0 and 0 < δ < 1")
        
        # Extract hyperparameters with DP-specific considerations
        num_epochs = kwargs.get('num_epochs', 100)
        learning_rate = kwargs.get('learning_rate', 0.01)
        batch_size = kwargs.get('batch_size', 32)
        
        # DP-specific hyperparameters
        max_grad_norm = kwargs.get('max_grad_norm', 1.0)  # Gradient clipping threshold
        noise_multiplier = kwargs.get('noise_multiplier', None)  # Will be auto-computed if None
        patience = kwargs.get('patience', 20)  # Reduced for DP training
        
        print(f"Training configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Max gradient norm (clipping): {max_grad_norm}")
        print(f"  Early stopping patience: {patience}")
        
        # Create data loaders - note that Opacus requires specific batch handling
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            drop_last=True  # Important for DP: ensures consistent batch sizes
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Dataset splits:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        print(f"  Training batches: {len(train_loader)}")
        
        # Validate model compatibility with Opacus
        # Opacus requires models to follow certain patterns for per-sample gradients
        try:
            model = ModuleValidator.fix(model)
            print("Model validated and fixed for Opacus compatibility")
        except Exception as e:
            print(f"Warning: Model validation failed: {e}")
            print("Proceeding anyway, but may encounter issues...")
        
        # Set up optimizer - must be done before PrivacyEngine
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.BCELoss(reduction='none')  # Important: no reduction for per-sample losses
        
        # Initialize the Privacy Engine - this is the core of Opacus
        privacy_engine = PrivacyEngine()
        
        # Attach the privacy engine to model, optimizer, and data loader
        # This transforms them to support differential privacy
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            epochs=num_epochs,
            target_epsilon=epsilon,
            target_delta=delta,
            max_grad_norm=max_grad_norm,
        )
        
        # The privacy engine automatically computed the noise multiplier
        print(f"Opacus computed noise multiplier: {optimizer.noise_multiplier:.4f}")
        print(f"Privacy accountant: {privacy_engine.accountant.__class__.__name__}")
        
        # Training history tracking
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        privacy_spent = []  # Track privacy expenditure over time
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("\nStarting DP-SGD training...")
        print("Note: DP training is typically slower due to per-sample gradient computation")
        
        for epoch in range(num_epochs):
            # ===== TRAINING PHASE =====
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # Use BatchMemoryManager for memory efficiency with large models
            # This processes batches in smaller chunks if needed
            with BatchMemoryManager(
                data_loader=train_loader,
                max_physical_batch_size=batch_size,
                optimizer=optimizer
            ) as memory_safe_data_loader:
                
                for batch_X, batch_y in memory_safe_data_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    batch_y = batch_y.view(-1, 1).float()
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass - handle dual output from MLP
                    hidden_repr, outputs = model(batch_X)
                    
                    # Compute per-sample losses (no reduction)
                    per_sample_losses = criterion(outputs, batch_y)
                    loss = per_sample_losses.mean()  # Average for backprop
                    
                    # Backward pass - Opacus handles the DP magic here
                    loss.backward()
                    
                    # Optimizer step - includes gradient clipping and noise addition
                    optimizer.step()
                    
                    # Accumulate training statistics
                    train_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()
            
            # ===== VALIDATION PHASE =====
            # Validation doesn't need DP, so we use the original model
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    batch_y = batch_y.view(-1, 1).float()
                    
                    # Forward pass
                    hidden_repr, outputs = model(batch_X)
                    
                    # For validation, we can use standard loss reduction
                    val_criterion = nn.BCELoss()
                    loss = val_criterion(outputs, batch_y)
                    
                    # Accumulate validation statistics
                    val_loss += loss.item()
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate epoch metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            # Get current privacy expenditure from Opacus
            current_epsilon = privacy_engine.get_epsilon(delta)
            
            # Store history
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            privacy_spent.append(current_epsilon)
            
            # Print progress with privacy information
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f} | '
                      f'Privacy: ε={current_epsilon:.4f}')
            
            # Check if we've exceeded our privacy budget
            if current_epsilon > epsilon:
                print(f"\nPrivacy budget exhausted! Current ε={current_epsilon:.4f} > target ε={epsilon}")
                break
            
            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model state
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            # Check if we should stop early
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f}')
                break
        
        # Restore the best model if we found one
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Restored best model weights")

        final_epsilon = privacy_engine.get_epsilon(delta)
        print(f"\nTraining completed!")
        print(f"Final privacy expenditure: ε={final_epsilon:.4f}, δ={delta}")
        print(f"Privacy budget utilization: {(final_epsilon/epsilon)*100:.1f}%")
        
        # Evaluate on test set
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_total, test_correct = 0, 0
        all_outputs, all_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                batch_y = batch_y.view(-1, 1).float()
                _, outputs = model(batch_X)
                preds = (outputs > 0.5).float()
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)
                all_outputs.extend(outputs.squeeze().cpu().numpy().tolist())
                all_labels.extend(batch_y.squeeze().cpu().numpy().tolist())
        test_accuracy = test_correct / test_total
        test_auroc = roc_auc_score(all_labels, all_outputs)

        # Save trained model
        self.model = model
        # Return structured training results
        return TrainingResults(
            auroc_score=test_auroc,
            accuracy=test_accuracy,
            mechanism_name=self.__class__.__name__,
            hyperparameters={
                'num_epochs': num_epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_grad_norm': max_grad_norm,
                'noise_multiplier': optimizer.noise_multiplier,
                'patience': patience
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
