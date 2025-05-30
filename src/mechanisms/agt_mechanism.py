from datasets import BaseDataset
from mechanism import BaseMechanism
from util.privacy import PrivacyBudget


class AGTMechanism(BaseMechanism):
    """
    AGT (Abstract Gradient Training) Mechanism for Differentially Private Prediction.
    This mechanism is designed as a prediction-privacy mechanism.
    """
    def __init__(self, model_constructor, dataset: BaseDataset, privacy_budget: PrivacyBudget, k: int):
        super().__init__(model_constructor, dataset, privacy_budget)
        self.k = k

        print(f"AGT Mechanism initialized with k={self.k} and privacy budget: {self.privacy_budget}")

    def train(self, **kwargs):
        raise NotImplementedError("AGT training not yet implemented.")

    def predict(self):
        raise NotImplementedError("AGT prediction not yet implemented.")
