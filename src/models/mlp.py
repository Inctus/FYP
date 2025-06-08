
import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna
from dataclasses import dataclass

@dataclass
class MLPHyperparameters:
    """
    Represents the hyperparameters for the MLP model.
    
    Attributes:
        mlp_layers (list): List of integers representing the number of neurons in each hidden layer.
        p_dropout (float): Dropout probability for regularization.
    """
    mlp_layers: list
    p_dropout: float


class BinaryClassificationMLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification tasks.
    
    This implementation is based on the Fair-Fairness Benchmark repository
    and includes some modifications for better integration with our training framework.
    """
    
    def __init__(self, n_features, hyperparameters: MLPHyperparameters):
        super(BinaryClassificationMLP, self).__init__()
        self.num_classes = 1
        self.mlp_layers = [n_features] + hyperparameters.mlp_layers
        self.p_dropout = hyperparameters.p_dropout

        # Create the hidden layers
        self.network = nn.ModuleList([
            nn.Linear(i, o) for i, o in zip(self.mlp_layers[:-1], self.mlp_layers[1:])
        ])
        
        # Final classification head
        self.head = nn.Linear(self.mlp_layers[-1], 1)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Returns both hidden representation and final prediction to allow for
        analysis of intermediate representations (useful for fairness research).
        """
        # Pass through hidden layers with ReLU activation and dropout
        for layer in self.network:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.p_dropout, training=self.training)
        
        # Store hidden representation (useful for fairness analysis)
        h = x
        
        # Final classification layer with sigmoid activation
        x = self.head(x)
        logits = torch.sigmoid(x)
        
        return h, logits

    @staticmethod
    def suggest_hyperparameters(trial: optuna.Trial) -> MLPHyperparameters:
        """
        Suggest hyperparameters for the MLP model based on the given Optuna trial.
        
        Args:
            trial (optuna.Trial): The trial object from Optuna.
        
        Returns:
            MLPHyperparameters: Suggested hyperparameters for the MLP model.
        """
        mlp_layers = [
            trial.suggest_categorical(f"mlp_hidden_dim_l{i}", [32, 64, 128, 256])
            for i in range(2)
        ]
        p_dropout = trial.suggest_float("mlp_dropout_p", 0.0, 0.5, step=0.1)
        
        return MLPHyperparameters(mlp_layers=mlp_layers, p_dropout=p_dropout)