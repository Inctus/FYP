"""
A mechanism should take an untrained model, a dataset, privacy budget and (a set of hyperparameters) and return a trained model.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from datasets import BaseDataset
from util.privacy import PrivacyBudget
from util.reproducibility import make_reproducible


@dataclass
class BaseHyperparameters:
    """
    Represents the hyperparameters used for training a mechanism.

    Attributes:
        learning_rate (float): The learning rate for the optimizer.
        n_epochs (int): The number of epochs to train the mechanism.
        batch_size (int): The size of the batches used during training.
        patience (int): The number of epochs with no improvement after which training will be stopped.
    """
    learning_rate: float
    n_epochs: int
    batch_size: int
    patience: int


@dataclass
class TrainingResults:
    """
    Represents the results of a training process.

    Attributes:
        auroc_score (float): The Area Under the Receiver Operating Characteristic Curve (AUROC) score of the model.
        accuracy (float): The accuracy of the model on the test set.

        mechanism_name (str): The name of the mechanism used for training.
        hyperparameters (BaseHyperparameters): The hyperparameters used during training.
    """
    auroc_score: float
    accuracy: float
    
    mechanism_name: str
    hyperparameters: BaseHyperparameters


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
    def train(self, hyperparameters: BaseHyperparameters) -> TrainingResults:
        """
        Train the mechanism on the dataset with the given privacy budget.
        This modifies the internal state of the class to include the trained model.
        The trained model is not returned directly, but we expose functionality using the "predict" method.

        Args:
            hyperparameters (BaseHyperparameters): Hyperparameters for the training process.

        Returns:
            The results of the training process.
        """
        pass

    @abstractmethod
    def predict(self, n_queries: int):
        """
        Make predictions using the trained model.

        Args:
            n_queries (int): The number of queries to make with the trained model.
                This is the number of predictions to return from the test set.

        Returns:
            The predictions made by the trained model.
        """
        pass
