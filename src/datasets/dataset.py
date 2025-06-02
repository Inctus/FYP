from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset as TorchDataset

from util.constants import RANDOM_SEED, TEST_SPLIT, TRAIN_SPLIT, VAL_SPLIT


class AIF360TorchDataset(TorchDataset):
    """
    Wraps an AIF360 BinaryLabelDataset as a PyTorch Dataset.
    
    This wrapper converts AIF360's numpy-based datasets into PyTorch-compatible
    format while preserving all the fairness-related metadata like protected attributes.
    """
    
    def __init__(self, aif360_dataset, include_protected=True):
        """
        Args:
            aif360_dataset: an AIF360 dataset object (e.g. AdultDataset(), COMPASDataset(), etc.)
            include_protected: if True, will also expose protected_attributes
        """
        # Extract numpy arrays from AIF360
        X = aif360_dataset.features
        y = aif360_dataset.labels.ravel()
        
        # Handle protected attributes if requested
        if include_protected:
            prot = aif360_dataset.protected_attributes
            self.protected_attrs = torch.tensor(prot, dtype=torch.float32)
        else:
            self.protected_attrs = None
        
        # Convert to PyTorch tensors with appropriate data types
        self.features = torch.tensor(X, dtype=torch.float32)
        # CRITICAL FIX: Use float32 for labels since we're using BCELoss
        self.labels = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        """
        Return features and labels in the format expected by DataLoader.
        For simplicity in training loop, we return a tuple (features, label)
        rather than a dictionary.
        """
        return self.features[idx], self.labels[idx]


class BaseDataset(ABC):
    """
    Abstract base class for dataset wrappers that provides a universal interface
    for loading, processing and accessing datasets in any format.
    """
    
    def __init__(self):
        """Initialize the dataset by loading the underlying AIF360 data."""
        self._aif360_dataset = self.load_data()
    
    @abstractmethod
    def load_data(self):
        """
        Loads the dataset from the AIF360 dataset object.
        Must be implemented by subclasses.
        
        Returns:
            AIF360 dataset object
        """
        raise NotImplementedError("load_data() must be implemented in subclasses")
    
    def to_torch(self, include_protected=True):
        """
        Converts the AIF360 dataset to PyTorch Datasets with train/val/test splits.
        
        Args:
            include_protected: if True, will also expose protected_attributes
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        converted_dataset = AIF360TorchDataset(self._aif360_dataset, include_protected)
        
        # Calculate split sizes
        total_size = len(converted_dataset)
        train_size = int(TRAIN_SPLIT * total_size)
        val_size = int(VAL_SPLIT * total_size)
        test_size = total_size - train_size - val_size
        # Ensure the splits sum to total size
        assert train_size + val_size + test_size == total_size, \
            "Train, validation, and test splits must sum to the total dataset size"
        # Log the sizes for debugging
        print(f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        # Perform the split with fixed random seed for reproducibility
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            converted_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(RANDOM_SEED)
        )
        
        return train_dataset, val_dataset, test_dataset