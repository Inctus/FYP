from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch.utils.data import Dataset as TorchDataset
import numpy as np
from sklearn.preprocessing import StandardScaler

from util.constants import RANDOM_SEED, TRAIN_SPLIT, VAL_SPLIT

class FairnessDataset(TorchDataset):
    """
    Wraps pre-processed features, labels, and optional protected attributes as a PyTorch Dataset.
    Data is expected to be in NumPy array format and will be converted to PyTorch tensors.
    """
    
    def __init__(self, features_np: np.ndarray, labels_np: np.ndarray, protected_attrs_np: np.ndarray | None = None):
        """
        Args:
            features_np: NumPy array of features.
            labels_np: NumPy array of labels.
            protected_attrs_np: Optional NumPy array of protected attributes.
        """
        self.features = torch.tensor(features_np, dtype=torch.float32)
        # Ensure labels are float32 for losses like BCELoss
        self.labels = torch.tensor(labels_np, dtype=torch.float32) 
        
        if protected_attrs_np is not None:
            self.protected_attrs = torch.tensor(protected_attrs_np, dtype=torch.float32)
        else:
            self.protected_attrs = None
    
    def __len__(self):
        return self.features.shape[0]
    
    def __getitem__(self, idx):
        """
        Return features, labels, and optionally protected attributes.
        """
        item_features = self.features[idx]
        item_labels = self.labels[idx]
        
        if self.protected_attrs is not None:
            return item_features, item_labels, self.protected_attrs[idx]
        else:
            return item_features, item_labels


class BaseDataset(ABC):
    """
    Abstract base class for dataset wrappers that provides a universal interface
    for loading, processing and accessing datasets in any format.
    """
    
    def __init__(self):
        """Initialize the dataset by loading the underlying AIF360 data."""
        self._aif360_dataset_original = self.load_data()
    
    @abstractmethod
    def load_data(self):
        """
        Loads the dataset from the AIF360 dataset object.
        Must be implemented by subclasses.
        
        Returns:
            AIF360 dataset object
        """
        raise NotImplementedError("load_data() must be implemented in subclasses")
    
    def to_torch(self, include_protected=True) -> Tuple[FairnessDataset, FairnessDataset, FairnessDataset]:
        """
        Processes the AIF360 dataset:
        1. Extracts features, labels, and (optionally) protected attributes.
        2. Splits data into train, validation, and test sets.
        3. Normalizes non-binary features (scaler fitted on training data only).
        4. Wraps the processed data into FairnessDataset instances.

        Args:
            include_protected: If True, protected attributes will be extracted and included.
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset) as FairnessDataset instances.
        """
        aif360_data = self._aif360_dataset_original

        # 1. Extract and validate data
        X_all_np = aif360_data.features.astype(np.float32)
        # This should work because AIF360 datasets are already
        # One-Hot encoded for categorical features. i.e. no strings here! Let's verify.

        y_all_np = aif360_data.labels.ravel().astype(np.float32)
        
        if X_all_np.shape[0] == 0:
            raise ValueError("Dataset contains no samples")
        if X_all_np.shape[1] == 0:
            raise ValueError("Dataset contains no features")
        
        prot_all_np = None
        if include_protected:
            if (hasattr(aif360_data, 'protected_attributes') and 
                aif360_data.protected_attributes is not None and
                aif360_data.protected_attributes.shape[1] > 0):
                prot_all_np = aif360_data.protected_attributes.astype(np.float32)
            else:
                print("Warning: include_protected is True, but protected_attributes are not available or empty.")

        # 2. Create deterministic split indices - SIMPLIFIED APPROACH
        total_size = X_all_np.shape[0]
        train_size = int(TRAIN_SPLIT * total_size)
        val_size = int(VAL_SPLIT * total_size)
        test_size = total_size - train_size - val_size
        
        if train_size == 0:
            raise ValueError("Training set would be empty with current split ratios")
        
        # Generate shuffled indices deterministically
        rng = np.random.RandomState(RANDOM_SEED)
        shuffled_indices = rng.permutation(total_size)
        
        train_indices = shuffled_indices[:train_size]
        val_indices = shuffled_indices[train_size:train_size + val_size]
        test_indices = shuffled_indices[train_size + val_size:]

        # 3. Split data
        X_train_np = X_all_np[train_indices]
        X_val_np = X_all_np[val_indices] if val_size > 0 else np.array([]).reshape(0, X_all_np.shape[1])
        X_test_np = X_all_np[test_indices] if test_size > 0 else np.array([]).reshape(0, X_all_np.shape[1])
        
        y_train_np = y_all_np[train_indices]
        y_val_np = y_all_np[val_indices] if val_size > 0 else np.array([])
        y_test_np = y_all_np[test_indices] if test_size > 0 else np.array([])
        
        prot_train_np = prot_val_np = prot_test_np = None
        if prot_all_np is not None:
            prot_train_np = prot_all_np[train_indices]
            prot_val_np = prot_all_np[val_indices] if val_size > 0 else np.array([]).reshape(0, prot_all_np.shape[1])
            prot_test_np = prot_all_np[test_indices] if test_size > 0 else np.array([]).reshape(0, prot_all_np.shape[1])

        # 4. Scale features (sklearn handles binary features gracefully)
        scaler = StandardScaler()
        X_train_np = scaler.fit_transform(X_train_np)
        if val_size > 0:
            X_val_np = scaler.transform(X_val_np)
        if test_size > 0:
            X_test_np = scaler.transform(X_test_np)

        # 5. Create FairnessDataset instances
        train_dataset = FairnessDataset(X_train_np, y_train_np, prot_train_np if include_protected else None)
        val_dataset = FairnessDataset(X_val_np, y_val_np, prot_val_np if include_protected else None)
        test_dataset = FairnessDataset(X_test_np, y_test_np, prot_test_np if include_protected else None)
        
        print(f"Dataset sizes - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset