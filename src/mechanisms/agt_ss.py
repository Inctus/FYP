from dataclasses import dataclass

import abstract_gradient_training as agt
import optuna
from mechanism import BaseHyperparameters, BaseMechanism, TrainingResults
from torch.utils.data import DataLoader

from datasets import BaseDataset
from util.privacy import PrivacyBudget
from util.reproducibility import make_reproducible


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
        self.bounded_model_dict = {}  # Dictionary to store bounded models for each k value

        print(f"AGT Mechanism initialized with k_values={self.k_values} and privacy budget: {self.privacy_budget}")
    
    def train(self, hyperparameters: AGTHyperparameters, device: str):
        model = self.model_constructor()

        config = agt.AGTConfig(
            learning_rate=hyperparameters.learning_rate,
            n_epochs=hyperparameters.n_epochs,
            loss="cross_entropy",
            log_level="WARNING",
            device=device,
            clip_gamma=hyperparameters.clip_gamma,
        )

        # Prepare the dataset
        train_dataset, val_dataset, test_dataset = self.dataset.to_torch()
        dataloader_train = DataLoader(train_dataset, batch_size=hyperparameters.batch_size, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=hyperparameters.batch_size, shuffle=False)

        for k_private in self.k_values:
            config.k_private=k_private
            make_reproducible()
            
            bounded_model = agt.bounded_models.IntervalBoundedModel(model)
            bounded_model = agt.privacy_certified_training(bounded_model, config, dataloader_train, dataloader_val)
            self.bounded_model_dict[k_private] = bounded_model
            
            # as a metric, compute the number of predictions in the test set certified at this value of k_private
            certified_preds = agt.test_metrics.certified_predictions(bounded_model, test_dataset) # TODO: Amend this if it works haha it is meant
            print(f"Certified Predictions at k={k_private}: {certified_preds:.2f}")

        noise_free_val_acc_l = agt.test_metrics.test_accuracy(self.bounded_model_dict[0], val_dataset)[0]

        return TrainingResults(
            accuracy= noise_free_val_acc_l,
            mechanism_name="AGT SS Mechanism",
            hyperparameters=hyperparameters,
        )

    def predict(self, device: str):
        raise NotImplementedError("AGT prediction not yet implemented.")

    def save(self, path: str):
        """
        Save the AGT mechanism to a file.
        This method should serialize the mechanism's state, including the model and hyperparameters.
        """
        for k_private, bounded_model in self.bounded_model_dict.items():
            bounded_model.save_params(f"{path}/agt_model_k_{k_private}")
    
    def load(self, path: str):
        """
        Load the AGT mechanism from a file.
        This method should deserialize the mechanism's state, including the model and hyperparameters.
        """
        model = self.model_constructor()

        for k_private in self.k_values:
            bounded_model = agt.bounded_models.IntervalBoundedModel(model)
            bounded_model.load_params(f"{path}/agt_model_k_{k_private}")
            self.bounded_model_dict[k_private] = bounded_model

    def suggest_hyperparameters(self, trial: optuna.Trial):
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        n_epochs = trial.suggest_int("n_epochs", 1, 100)
        batch_size = trial.suggest_int("batch_size", 1, 64)
        patience = trial.suggest_int("patience", 1, 10)
        clip_gamma = trial.suggest_float("clip_gamma", 0.0, 1.0)

        return AGTHyperparameters(
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            patience=patience,
            clip_gamma=clip_gamma   
        )
