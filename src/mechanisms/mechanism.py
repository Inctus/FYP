"""
A mechanism should take an untrained model, a dataset, privacy budget and (a set of hyperparameters) and return a trained model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from datasets import BaseDataset
from util.reproducibility import make_reproducible
from util.privacy import PrivacyBudget

@dataclass
class TrainingResults:
    """
    Represents the results of a training process.

    Attributes:
        auroc_score (float): The Area Under the Receiver Operating Characteristic Curve (AUROC) score of the model.
        accuracy (float): The accuracy of the model on the test set.

        mechanism_name (str): The name of the mechanism used for training.
        hyperparameters (dict): The hyperparameters used during training.
    """
    auroc_score: float
    accuracy: float
    
    mechanism_name: str
    hyperparameters: dict


class BaseMechanism(ABC):
    """
    A base class for mechanisms that train models with differential privacy.

    TODO: Refactor this to make the init functions 
    """
    def __init__(self, model_constructor, dataset: BaseDataset, privacy_budget: PrivacyBudget):
        """
        A basic initialisation that makes sure mechanisms are reproducible.
        """
        make_reproducible() # Ensures reproducibility across runs
        
        self.model_constructor = model_constructor
        self.dataset = dataset
        self.privacy_budget = privacy_budget

    @abstractmethod
    def train(self, **kwargs) -> TrainingResults:
        """
        Train the mechanism on the dataset with the given privacy budget.
        This modifies the internal state of the class to include the trained model.
        The trained model is not returned directly, but we expose functionality using the "predict" method.

        Args:
            **kwargs: Hyperparameters for the training process.

        Returns:
            The results of the training process.
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Make predictions using the trained model.

        Returns:
            The predictions made by the trained model.
        """
        pass
