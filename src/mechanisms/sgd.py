from mechanism import BaseMechanism, TrainingResults
from datasets import BaseDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from util.privacy import PrivacyBudget
from sklearn.metrics import roc_auc_score


class SGDMechanism(BaseMechanism):
    """
    A mechanism that trains a model using SGD, designed for fairness research.
    
    This implementation includes proper early stopping, comprehensive logging,
    and handles the dual-output nature of our MLP model.
    """
    
    def __init__(self, model_constructor, dataset: BaseDataset, privacy_budget: PrivacyBudget):
        super().__init__(model_constructor, dataset, privacy_budget)
        print("Initialized SGDMechanism for training models with fairness considerations")
    
    def train(self, **kwargs) -> TrainingResults:
        # Instantiate model and get dataset
        model = self.model_constructor()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print(f"Training on device: {device}")
        
        # Access dataset and ignore stored privacy budget
        train_dataset, val_dataset, test_dataset = self.dataset.to_torch(include_protected=False)
        
        # Extract hyperparameters with sensible defaults
        num_epochs = kwargs.get('num_epochs', 100)
        learning_rate = kwargs.get('learning_rate', 0.01)
        batch_size = kwargs.get('batch_size', 32)
        patience = kwargs.get('patience', 100)  # For early stopping
        
        print(f"Training configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Batch size: {batch_size}")
        print(f"  Early stopping patience: {patience}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Dataset splits:")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Test samples: {len(test_dataset)}")
        
        # Set up optimizer and loss function
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        criterion = nn.BCELoss()  # Binary Cross-Entropy for binary classification
        
        # Training history tracking
        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("\nStarting training...")
        
        for epoch in range(num_epochs):
            # ===== TRAINING PHASE =====
            model.train()  # Enable dropout and batch norm training mode
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                # Move data to device
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Ensure labels have correct shape for BCELoss
                batch_y = batch_y.view(-1, 1)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass - IMPORTANT: handle dual output from MLP
                logits, outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                # Accumulate training statistics
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # ===== VALIDATION PHASE =====
            model.eval()  # Disable dropout and set batch norm to eval mode
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():  # Disable gradient computation for efficiency
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    batch_y = batch_y.view(-1, 1)
                    
                    # Forward pass
                    hidden_repr, outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
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
            
            # Store history
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_acc)
            val_losses.append(avg_val_loss)
            val_accuracies.append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1:3d}/{num_epochs}] | '
                      f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | '
                      f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # CRITICAL FIX: Implement early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save the best model state
                best_model_state = model.state_dict().copy()
                print(f'New best validation loss: {best_val_loss:.4f}')
            else:
                patience_counter += 1
            
            # Check if we should stop early
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                print(f'Best validation loss: {best_val_loss:.4f}')
                break
        
        # Restore the best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print("Restored best model weights from validation phase")
        
            # Evaluate on test set
            model.eval()
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_total, test_correct = 0, 0
            all_outputs, all_labels = [], []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    batch_y = batch_y.view(-1, 1)
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