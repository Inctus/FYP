"""
A mechanism should take an untrained model, a dataset, privacy budget (if applicable) 
and (a set of hyperparameters) and return a trained model or allow predictions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import optuna

from datasets.dataset import BaseDataset
from util.privacy import PrivacyBudget
from util.reproducibility import make_reproducible


@dataclass
class BaseHyperparameters:
    """
    Represents the base hyperparameters used for training a mechanism.

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
        accuracy (float): The accuracy of the model on a relevant dataset (e.g., test or validation).
        mechanism_name (str): The name of the mechanism used for training.
        hyperparameters (BaseHyperparameters): The hyperparameters used during training.
    """
    accuracy: float
    mechanism_name: str
    hyperparameters: BaseHyperparameters


class BaseMechanism(ABC):
    """
    A base class for mechanisms that train models.
    This class defines the API for non-private training and prediction over the entire test set.
    The mechanism itself does not store the trained model state directly after training;
    `save` and `load` methods are used to manage the persistence and restoration of this state.
    """
    def __init__(self, model_constructor, dataset: BaseDataset):
        """
        A basic initialisation that makes sure mechanisms are reproducible.
        """
        make_reproducible() # Ensures reproducibility across runs
        
        self.model_constructor = model_constructor
        self.dataset = dataset

    @abstractmethod
    def train(self, hyperparameters: BaseHyperparameters, device: str) -> TrainingResults:
        """
        Train the mechanism using the provided hyperparameters on the specified device.
        This method is responsible for the training process but does not necessarily
        store the final trained state within the instance. The state should be
        savable via the `save` method.

        Args:
            hyperparameters (BaseHyperparameters): Hyperparameters for the training process.
            device (str): The device to use for training (e.g., 'cpu' or 'cuda').

        Returns:
            TrainingResults: The results of the training process.
        """
        pass

    @abstractmethod
    def predict(self, device: str) -> Any:
        """
        Make predictions using the state established by `load()` (or a preceding `train()`
        call if the mechanism's implementation supports it).
        Predictions are made over the entire test set.

        Args:
            device (str): The device to use for inference (e.g., 'cpu' or 'cuda').

        Returns:
            Any: The predictions (e.g., raw scores or class labels).
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """
        Save the trainable state of the mechanism (e.g., model parameters, ensemble components)
        to a file. This state should be sufficient to later `load` and `predict`.

        Args:
            path (str): The path where the state should be saved.
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """
        Load the trainable state of the mechanism from a file.
        This prepares the mechanism for making predictions via the `predict` method.

        Args:
            path (str): The path from which the state should be loaded.
        """
        pass

    @abstractmethod
    def suggest_hyperparameters(self, trial: optuna.Trial) -> BaseHyperparameters:
        """
        Suggest hyperparameters for the mechanism based on the given Optuna trial.

        Args:
            trial (optuna.Trial): The trial object from Optuna.

        Returns:
            BaseHyperparameters: Suggested hyperparameters for the mechanism.
        """
        pass


class DPLearningMechanism(BaseMechanism):
    """
    A base class for mechanisms that train models with differential privacy during the learning process.
    The training API includes a privacy budget.
    Prediction is non-private over the entire test set, using the state established by `load()`.
    """
    def __init__(self, model_constructor, dataset: BaseDataset):
        """
        Initializes the DPLearningMechanism with a model constructor and dataset.
        Ensures reproducibility across runs.
        """
        super().__init__(model_constructor, dataset)

    @abstractmethod
    def train(self, hyperparameters: BaseHyperparameters, privacy_budget: PrivacyBudget, device: str) -> TrainingResults:
        """
        Train the mechanism with differential privacy, using the provided hyperparameters,
        privacy budget, on the specified device.
        The resulting state should be savable via the `save` method.

        Note: This method overrides BaseMechanism.train with an additional privacy_budget parameter.

        Args:
            hyperparameters (BaseHyperparameters): Hyperparameters for the training process.
            privacy_budget (PrivacyBudget): The privacy budget for DP training.
            device (str): The device to use for training (e.g., 'cpu' or 'cuda').

        Returns:
            TrainingResults: The results of the training process.
        """
        pass


class DPPredictionMechanism(BaseMechanism):
    """
    A base class for mechanisms that apply differential privacy during the prediction phase.
    Training is standard (non-private). The trained state is managed via `save` and `load`.
    The prediction API includes a number of queries and a privacy budget for those queries.
    """
    def __init__(self, model_constructor, dataset: BaseDataset):
        """
        Initializes the DPPredictionMechanism with a model constructor and dataset.
        Ensures reproducibility across runs.
        """
        super().__init__(model_constructor, dataset)

    @abstractmethod
    def predict(self, n_queries: int, privacy_budget: PrivacyBudget, device: str) -> Any:
        """
        Make a specified number of predictions using the state established by `load()`,
        applying differential privacy to the prediction process.
        The privacy budget is spent over these queries.

        Note: This method overrides BaseMechanism.predict with additional parameters
              n_queries and privacy_budget.

        Args:
            n_queries (int): The number of queries (predictions) to make.
            privacy_budget (PrivacyBudget): The privacy budget for these queries.
            device (str): The device to use for inference (e.g., 'cpu' or 'cuda').

        Returns:
            Any: The private predictions made by the model.
        """
        pass