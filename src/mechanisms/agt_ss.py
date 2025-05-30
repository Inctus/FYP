from datasets import BaseDataset
from mechanism import BaseMechanism
from util.privacy import PrivacyBudget


class AGTSSMechanism(BaseMechanism):
    """
    AGT-SS (Abstract Gradient Training with Smooth Sensitivity) Mechanism for Differentially Private Prediction.
    This mechanism is designed as a prediction-privacy mechanism.
    """
    def __init__(self, model_constructor, dataset: BaseDataset, privacy_budget: PrivacyBudget):
        super().__init__(model_constructor, dataset, privacy_budget)

    def train(self, **kwargs):
        raise NotImplementedError("AGT-SS training not yet implemented.")

    def predict(self):
        raise NotImplementedError("AGT-SS prediction not yet implemented.")
