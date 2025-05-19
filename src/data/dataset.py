import torch
from torch.utils.data import Dataset as TorchDataset

from abc import ABC, abstractmethod

class AIF360TorchDataset(TorchDataset):
    """
    Wraps an AIF360 BinaryLabelDataset (or similar) as a PyTorch Dataset.
    Exposes:
      - features:          torch.float32 tensor of shape (n_samples, n_features)
      - labels:            torch.long tensor of shape (n_samples,)
      - protected_attrs:   torch.float32 tensor of shape (n_samples, n_protected_attrs)
    """

    def __init__(self, aif360_dataset, include_protected=True, transform=None):
        """
        Args:
            aif360_dataset:      an AIF360 dataset object (e.g. AdultDataset(), COMPASDataset(), etc.)
            include_protected:   if True, will also expose protected_attributes
            transform:           optional callable to apply to each feature vector
        """
        # raw numpy arrays from AIF360
        X = aif360_dataset.features
        y = aif360_dataset.labels.ravel()
        
        # protected_attributes is shape (n_samples, n_protected_attrs)
        if include_protected:
            prot = aif360_dataset.protected_attributes
            self.protected_attrs = torch.tensor(prot, dtype=torch.float32)
        else:
            self.protected_attrs = None
        
        # convert to torch tensors
        self.features = torch.tensor(X, dtype=torch.float32)
        self.labels   = torch.tensor(y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        x = self.features[idx]
        if self.transform:
            x = self.transform(x)
        sample = {
            'features': x,
            'label':     self.labels[idx]
        }
        if self.protected_attrs is not None:
            sample['protected_attributes'] = self.protected_attrs[idx]
        return sample


class BaseDataset(ABC):
    """
    A wrapper for the AIF360 dataset that provides a universal interface for
    loading, processing and accessing datasets in any format
    """

    def __init__(self):
        """
        Args:
            aif360_dataset:      an AIF360 dataset object (e.g. AdultDataset(), COMPASDataset(), etc.)
        """
        self._aif360_dataset = self.load_data()
    
    @abstractmethod
    def load_data(self):
        """
        Loads the dataset from the AIF360 dataset object
        Returns:
            AIF360 dataset object
        """
        raise NotImplementedError("load_data() must be implemented in subclasses")

    def to_torch(self, include_protected=True, transform=None):
        """
        Converts the AIF360 dataset to a PyTorch Dataset
        Args:
            include_protected:   if True, will also expose protected_attributes
            transform:           optional callable to apply to each feature vector
        Returns:
            AIF360TorchDataset:  a PyTorch Dataset object
        """
        return AIF360TorchDataset(self._aif360_dataset, include_protected, transform)
