import torch.nn as nn

import optuna

from dataclasses import dataclass

@dataclass
class AGTBCMLPHyperparameters:
    """
    Hyperparameters for the AGTBCMLP model.
    
    Inherits from MLPHyperparameters and can be extended with additional parameters specific to AGTBCMLP.
    """
    mlp_layers: list


class AGTBCMLP(nn.Sequential):
    """
    A Multi-Layer Perceptron (MLP) for binary classification tasks.
    
    This class extends `nn.Sequential` to create a flexible MLP architecture based on the provided hyperparameters.
    """
    def __init__(self, n_features: int, hyperparameters: AGTBCMLPHyperparameters):
        layers = []
        input_dim = n_features
        
        # Create hidden layers
        for layer_size in hyperparameters.mlp_layers:
            layers.append(nn.Linear(input_dim, layer_size))
            layers.append(nn.ReLU())
            input_dim = layer_size
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))  # Single output for binary classification
        
        super().__init__(*layers)

    @staticmethod
    def suggest_hyperparameters(trial: optuna.Trial) -> AGTBCMLPHyperparameters:
        """
        Suggest hyperparameters for the MLP model using Optuna.
        
        Args:
            trial (optuna.Trial): The current trial object.
        
        Returns:
            MLPHyperparameters: Suggested hyperparameters for the MLP model.
        """
        mlp_layer_0 = trial.suggest_categorical("mlp_layer_0", [32, 64, 128])
        mlp_layer_1 = trial.suggest_categorical("mlp_layer_1", [32, 64, 128, 256])
        
        return AGTBCMLPHyperparameters(mlp_layers=[mlp_layer_0, mlp_layer_1])
