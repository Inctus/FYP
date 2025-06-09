import optuna
import json
from pathlib import Path
from typing import Type, Dict, Any

from mechanisms.mechanism import BaseMechanism, DPLearningMechanism
from datasets.dataset import BaseDataset

from util.constants import HYPERPARAMETER_RESULTS_DIR


class HyperparameterTuner:
    """
    A class to perform hyperparameter tuning for mechanisms using Optuna.
    It tunes both mechanism-specific hyperparameters and model architectural hyperparameters (specifically for MLP).
    """
    def __init__(self, mechanism_class: Type[BaseMechanism], model_class, dataset: BaseDataset, device: str):
        """
        Initializes the HyperparameterTuner.

        Args:
            mechanism_class: The class of the mechanism to tune (must be a subclass of BaseMechanism
                             or DPPredictionMechanism, but not DPLearningMechanism for this tuner's default setup).
            model_class: The class of the model to be used (e.g., MLP).
            dataset: The dataset instance to use for training and evaluation.
            device: The device to use for training (e.g., "cuda" or "cpu").
        """
        self.mechanism_class = mechanism_class
        self.model_class = model_class
        self.dataset = dataset
        self.device = device
        self.storage_base_dir = Path(HYPERPARAMETER_RESULTS_DIR)

    def objective(self, trial: optuna.Trial) -> float:
        """
        Optuna objective function for a single trial.
        """
        # 1. Suggest model architectural hyperparameters
        model_arch_params = self.model_class.suggest_hyperparameters(trial)

        # 2. Create model constructor with these trial parameters
        #    Input and output dimensions are determined by the dataset.
        input_dim = self.dataset.n_features

        def model_constructor_with_trial_params():
            return self.model_class(input_dim, model_arch_params)

        # 3. Instantiate the mechanism with the dataset and the trial-specific model constructor
        mechanism_instance = self.mechanism_class(
            model_constructor=model_constructor_with_trial_params,
            dataset=self.dataset
        )

        # 4. Suggest mechanism-specific training hyperparameters
        #    This uses the mechanism's own `suggest_hyperparameters` method.
        #    This method should return a dataclass like BaseHyperparameters.
        mechanism_train_hyperparams = mechanism_instance.suggest_hyperparameters(trial)

        # 5. Train the mechanism
        #    This tuner is designed for mechanisms with train(hyperparameters, device)
        try:
            # The train method should return TrainingResults
            training_results = mechanism_instance.train(
                hyperparameters=mechanism_train_hyperparams,
                device=self.device
            )
            print(f"Trial {trial.number} results: {training_results}")

            accuracy = training_results.accuracy # Optuna will maximize this
        except Exception as e:
            print(f"Trial {trial.number} failed with exception: {e}")
            # Prune trial if it fails (e.g., due to bad hyperparams leading to NaN loss, etc.)
            raise optuna.exceptions.TrialPruned()

        return accuracy

    def tune(self, n_trials: int, study_name: str) -> Dict[str, Any]:
        """
        Runs the hyperparameter tuning process.

        Args:
            n_trials: The number of trials to run.
            study_name: Optional name for the Optuna study.
            direction: "maximize" or "minimize" the objective.

        Returns:
            A dictionary containing the best hyperparameters found.
        """
        # Create an Optuna study
        study = optuna.create_study(direction="maximize", study_name=study_name)

        # Start optimization
        study.optimize(self.objective, n_trials=n_trials)

        best_trial = study.best_trial
        print(f"\nBest trial for {study_name}:")
        print(f"  Value accuracy: {best_trial.value}")
        print("  Best Hyperparameters (includes model and mechanism params):")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        # Save the best hyperparameters to a file
        self.save_hyperparameters(best_trial.params, study_name)

        return best_trial.params

    def save_hyperparameters(self, hyperparams: Dict[str, Any], study_name: str):
        """
        Saves the best hyperparameters to a JSON file.
        """
        self.storage_base_dir.mkdir(parents=True, exist_ok=True)

        # Include study_name in filename to distinguish different tuning runs if needed
        save_path = self.storage_base_dir / f"{study_name}.json"
        with open(save_path, 'w') as f:
            json.dump(hyperparams, f, indent=4)
        print(f"Best hyperparameters saved to {save_path}")
