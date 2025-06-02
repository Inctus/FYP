import abstract_gradient_training as agt
from mechanism import BaseMechanism, BaseHyperparameters

from datasets import BaseDataset
from util.privacy import PrivacyBudget
from util.reproducibility import make_reproducible

from dataclasses import dataclass


@dataclass
class AGTHyperparameters(BaseHyperparameters):
    """
    Hyperparameters for the AGT Mechanism.
    Inherited Attributes:
        - learning_rate: float
        - n_epochs: int
        - batch_size: int
        - patience: int
    Additional Attributes:
        - clip_gamma: float
    """
    clip_gamma: float


class AGTMechanism(BaseMechanism):
    """
    AGT (Abstract Gradient Training) Mechanism for Differentially Private Prediction.
    This mechanism is designed as a prediction-privacy mechanism.
    """
    def __init__(self, model_constructor, dataset: BaseDataset, privacy_budget: PrivacyBudget):
        super().__init__(model_constructor, dataset, privacy_budget)
        self.k_values = [0, 1, 10, 20, 50, 100] # TODO: Check how to derive these values from the dataset

        print(f"AGT Mechanism initialized with k_values={self.k_values} and privacy budget: {self.privacy_budget}")

    def train(self, hyperparameters: AGTHyperparameters):
        model = self.model_constructor()

        config = agt.AGTConfig(
            learning_rate=hyperparameters.learning_rate,
            n_epochs=hyperparameters.n_epochs
            loss="cross_entropy",
            log_level="WARNING",
            device="cuda:0",
            clip_gamma=hyperparameters.clip_gamma,
        )
        bounded_model_dict = {}  # we'll store our results for each value of 'k' as a dictionary from 'k' to the bounded model

        for k_private in self.k_values:
            config.k_private=k_private
            make_reproducible()
            
            bounded_model = agt.bounded_models.IntervalBoundedModel(model)
            bounded_model = agt.privacy_certified_training(bounded_model, config, dataloader_train)
            bounded_model_dict[k_private] = bounded_model
            
            # as a metric, compute the number of predictions in the test set certified at this value of k_private
            certified_preds = agt.test_metrics.certified_predictions(bounded_model, x_test)
            print(f"Certified Predictions at k={k_private}: {certified_preds:.2f}")

    def predict(self):
        raise NotImplementedError("AGT prediction not yet implemented.")
