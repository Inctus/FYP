
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Multi-Layer Perceptron for classification tasks.
    
    This implementation is based on the Fair-Fairness Benchmark repository
    and includes some modifications for better integration with our training framework.
    """
    
    def __init__(self, n_features, mlp_layers=[128, 32], p_dropout=0.2, num_classes=1):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.mlp_layers = [n_features] + mlp_layers
        self.p_dropout = p_dropout
        
        # Create the hidden layers
        self.network = nn.ModuleList([
            nn.Linear(i, o) for i, o in zip(self.mlp_layers[:-1], self.mlp_layers[1:])
        ])
        
        # Final classification head
        self.head = nn.Linear(self.mlp_layers[-1], num_classes)
    
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