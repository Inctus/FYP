from dataclasses import dataclass

import abstract_gradient_training as agt
import optuna
from mechanism import BaseHyperparameters, BaseMechanism
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

    def train(self, hyperparameters: AGTHyperparameters):
        model = self.model_constructor()

        config = agt.AGTConfig(
            learning_rate=hyperparameters.learning_rate,
            n_epochs=hyperparameters.n_epochs,
            loss="cross_entropy",
            log_level="WARNING",
            device="cuda:0",
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
        

        # TODO: Calculate AUROC and Accuracy over validation set
        """s
        Example code for computing accuracy using SS
        
        Let's use this set of bounded models to for better private prediction using the smooth sensitivity mechanism.

        epsilon = 0.5  # privacy loss
        noise_free_acc = agt.test_metrics.test_accuracy(bounded_model_dict[0], x_test, y_test)[0]

        # compute accuracy using the smooth sensitivity Cauchy mechanism
        smooth_sens_noise_level = agt.privacy_utils.get_calibrated_noise_level(
            x_test, bounded_model_dict, epsilon, noise_type="cauchy"
        )
        smooth_sens_acc = agt.privacy_utils.noisy_test_accuracy(
            bounded_model_dict[0], x_test, y_test, noise_level=smooth_sens_noise_level, noise_type="cauchy"
        )

        # compute accuracy when using the global sensitivity mechanism
        global_sens_acc = agt.privacy_utils.noisy_test_accuracy(
            bounded_model_dict[0], x_test, y_test, noise_level=1.0 / epsilon
        )

        print(f"Noise Free Accuracy: {noise_free_acc:.2f}")
        print(f"Smooth Sensitivity Accuracy: {smooth_sens_acc:.2f}")
        print(f"Global Sensitivity Accuracy: {global_sens_acc:.2f}")
        """


    def predict(self):
        raise NotImplementedError("AGT prediction not yet implemented.")

    def save(self, path: str):
        """
        Save the AGT mechanism to a file.
        This method should serialize the mechanism's state, including the model and hyperparameters.
        """
        raise NotImplementedError("AGT save not yet implemented.")
    
    def load(self, path: str):
        """
        Load the AGT mechanism from a file.
        This method should deserialize the mechanism's state, including the model and hyperparameters.
        """
        raise NotImplementedError("AGT load not yet implemented.")

    def suggest_hyperparameters(self, trial: optuna.Trial):
        raise NotImplementedError("AGT hyperparameter suggestion not yet implemented.")