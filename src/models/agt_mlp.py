
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


class AGTBCMLP(nn.Sequential):
    """
    A Multi-Layer Perceptron (MLP) for binary classification tasks.
    
    This class extends `nn.Sequential` to create a flexible MLP architecture based on the provided hyperparameters.
    """
    def __init__(self, n_features: int, hyperparameters: MLPHyperparameters):
        layers = []
        input_dim = n_features
        
        # Create hidden layers
        for layer_size in hyperparameters.mlp_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(hyperparameters.p_dropout))
            input_dim = layer_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))  # Single output for binary classification
        
        super().__init__(*layers)

    @staticmethod
    def suggest_hyperparameters(trial: optuna.Trial) -> MLPHyperparameters:
        """
        Suggest hyperparameters for the MLP model using Optuna.
        
        Args:
            trial (optuna.Trial): The current trial object.
        
        Returns:
            MLPHyperparameters: Suggested hyperparameters for the MLP model.
        """
        mlp_layers = [
            trial.suggest_categorical(f"mlp_hidden_dim_l{i}", [32, 64, 128, 256])
            for i in range(2)
        ]
        p_dropout = trial.suggest_float("p_dropout", 0.0, 0.2)
        
        return MLPHyperparameters(mlp_layers=mlp_layers, p_dropout=p_dropout)
