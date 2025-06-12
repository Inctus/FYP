from dataclasses import dataclass

import abstract_gradient_training as agt
import optuna
from mechanisms.mechanism import BaseHyperparameters, DPPredictionMechanism, TrainingResults
from torch.utils.data import DataLoader, RandomSampler
import random
from torch.utils.data import Sampler
import torch

from datasets.dataset import BaseDataset
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
    lr_min: float
    lr_decay: float
    momentum: float

# Custom sampler ensuring at least one positive and one negative from each group per batch
class MinGroupBatchSampler(Sampler):
    def __init__(self, group0, group1, all_inds, batch_size, num_batches, labels):
        self.group0 = group0
        self.group1 = group1
        self.all_inds = all_inds
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.labels = labels
        
        # Split indices by both group and label
        self.group0_pos = [i for i in group0 if labels[i] == 1]
        self.group0_neg = [i for i in group0 if labels[i] == 0]
        self.group1_pos = [i for i in group1 if labels[i] == 1]
        self.group1_neg = [i for i in group1 if labels[i] == 0]
        
        # Validation for edge cases
        if len(group0) == 0:
            raise ValueError("Group 0 must have at least one sample")
        if len(group1) == 0:
            raise ValueError("Group 1 must have at least one sample")
        if batch_size < 4:
            raise ValueError("Batch size must be at least 4 to guarantee one positive and one negative from each group")
        
        # Check for equalized odds requirements
        if len(self.group0_pos) == 0:
            raise ValueError("Group 0 must have at least one positive sample (label=1) for equalized odds calculation")
        if len(self.group0_neg) == 0:
            raise ValueError("Group 0 must have at least one negative sample (label=0) for equalized odds calculation")
        if len(self.group1_pos) == 0:
            raise ValueError("Group 1 must have at least one positive sample (label=1) for equalized odds calculation")
        if len(self.group1_neg) == 0:
            raise ValueError("Group 1 must have at least one negative sample (label=0) for equalized odds calculation")
    
    def __iter__(self):
        for _ in range(self.num_batches):
            # Guarantee one positive and one negative from each group (4 samples minimum)
            idx_g0_pos = random.choice(self.group0_pos)
            idx_g0_neg = random.choice(self.group0_neg)
            idx_g1_pos = random.choice(self.group1_pos)
            idx_g1_neg = random.choice(self.group1_neg)
            
            batch = [idx_g0_pos, idx_g0_neg, idx_g1_pos, idx_g1_neg]
            
            # Fill remaining slots, avoiding duplicates within this batch
            available_indices = [i for i in self.all_inds if i not in batch]
            
            remaining_slots = self.batch_size - 4
            if remaining_slots > 0:
                if len(available_indices) >= remaining_slots:
                    # Sample without replacement from remaining indices
                    rest = random.sample(available_indices, remaining_slots)
                    batch.extend(rest)
                else:
                    # Edge case: not enough unique indices left
                    # This shouldn't happen if batch_size <= len(all_inds), but handle gracefully
                    rest = available_indices + random.choices(
                        self.all_inds, k=remaining_slots - len(available_indices)
                    )
                    batch.extend(rest)
            
            random.shuffle(batch)
            yield batch
    
    def __len__(self):
        return self.num_batches


class AGTMechanism(DPPredictionMechanism):
    """
    AGT (Abstract Gradient Training) Mechanism for Differentially Private Prediction.
    This mechanism is designed as a prediction-privacy mechanism.
    """
    def __init__(self, model_constructor, dataset: BaseDataset):
        super().__init__(model_constructor, dataset)
        self.k_values = [0, 1, 10, 20, 50, 100, 1000]
        self.bounded_model_dict = {}  # Dictionary to store bounded models for each k value

        print(f"AGT Mechanism initialized with k_values={self.k_values}")

    def train(self, hyperparameters: AGTHyperparameters, device: str):
        model = self.model_constructor().double().to(device)

        config = agt.AGTConfig(
            learning_rate=hyperparameters.learning_rate,
            n_epochs=hyperparameters.n_epochs,
            loss="binary_cross_entropy",
            log_level="INFO",
            device=device,
            clip_gamma=hyperparameters.clip_gamma,
            optimizer="SGDM",
            optimizer_kwargs={"momentum": hyperparameters.momentum, "nesterov": True},
            lr_decay=hyperparameters.lr_decay,
            lr_min=hyperparameters.lr_min,
        )

        # Prepare the dataset
        train_dataset, val_dataset, test_dataset = self.dataset.to_torch(make_float64=True)
        dataloader_train = DataLoader(train_dataset, batch_size=hyperparameters.batch_size, shuffle=True)
        dataloader_val = DataLoader(val_dataset, batch_size=hyperparameters.batch_size, shuffle=False)

        for k_private in self.k_values:
            config.k_private=k_private
            make_reproducible()
            
            bounded_model = agt.bounded_models.IntervalBoundedModel(model)
            bounded_model = agt.privacy_certified_training(bounded_model, config, dataloader_train, dataloader_val)
            self.bounded_model_dict[k_private] = bounded_model
            
            # as a metric, compute the number of predictions in the test set certified at this value of k_private
            certified_preds = agt.test_metrics.certified_predictions(bounded_model, test_dataset.features)
            print(f"Certified Predictions at k={k_private}: {certified_preds:.2f}")

        # After training, we can compute the accuracy on the validation set
        # using the smooth sensitivity Laplace mechanism
        test_epsilon = 2.0  # privacy loss
        test_delta = 1e-5  # privacy loss

        # compute accuracy using the smooth sensitivity Laplace mechanism
        smooth_sens_noise_level = agt.privacy_utils.get_calibrated_noise_level(
            val_dataset.features, self.bounded_model_dict, test_epsilon, test_delta, noise_type="laplace"
        )
        smooth_sens_acc = agt.privacy_utils.noisy_test_accuracy(
            self.bounded_model_dict[0], val_dataset.features, val_dataset.labels, noise_level=smooth_sens_noise_level, noise_type="laplace"
        )

        return TrainingResults(
            accuracy= smooth_sens_acc,
            mechanism_name="AGT SS Mechanism",
            hyperparameters=hyperparameters,
        )

    def predict(self, n_queries: int, privacy_budget: PrivacyBudget, device: str):
        """
        Predicts using the AGT mechanism. It repeatedly queries n_queries from the dataset
        using the entire privacy budget. This is to create enough data to predict population
        statistics from the sample size n_queries.
        Args:
            n_queries (int): Number of queries to make.
            privacy_budget (PrivacyBudget): The privacy budget to use for the queries.
            device (str): The device to run the predictions on (e.g., 'cpu' or 'cuda').
        Returns:
            A list of predictions for the n_queries samples.
        """
        _, _, test_dataset = self.dataset.to_torch(make_float64=True)
        len_test = len(test_dataset)
        if n_queries > len_test:
            raise ValueError(f"n_queries ({n_queries}) cannot be greater than the number of samples in the test dataset ({len_test}).")
        
        # Check minimum batch size for equalized odds
        if n_queries < 4:
            raise ValueError("n_queries must be at least 4 to guarantee samples from both groups and both labels for equalized odds calculation")

        make_reproducible()

        # Build lists of protected attribute values and labels
        protected_list = []
        labels_list = []
        for _, label, prot in test_dataset:
            val = prot.item() if isinstance(prot, torch.Tensor) and prot.numel()==1 else prot
            label_val = label.item() if isinstance(label, torch.Tensor) and label.numel()==1 else label
            protected_list.append(val)
            labels_list.append(label_val)
        
        # Split indices by attribute value
        group0_indices = [i for i, v in enumerate(protected_list) if v == 0]
        group1_indices = [i for i, v in enumerate(protected_list) if v == 1]
        all_indices = list(range(len_test))
        num_batches = 1000

        batch_sampler = MinGroupBatchSampler(
            group0_indices, group1_indices, all_indices, 
            n_queries, num_batches, labels_list
        )
        
        dataloader = DataLoader(test_dataset, batch_sampler=batch_sampler)

        model = self.bounded_model_dict[0]
        prediction_list = []

        # Run inference on each sampled point
        for features, labels, protected_attributes in dataloader:

            features = features.to(device)

            noise_level = agt.privacy_utils.get_calibrated_noise_level(
                features, self.bounded_model_dict, privacy_budget.epsilon, privacy_budget.delta, noise_type="laplace"
            )
            pred = agt.privacy_utils.noisy_predictions(
                model, features, labels, noise_level=noise_level, noise_type="laplace"
            )

            results = (
                pred.squeeze().cpu().numpy().tolist(),
                labels.squeeze().cpu().numpy().tolist(),
                protected_attributes.squeeze().cpu().numpy().tolist()
            )

            # Convert tensor to Python scalar if needed
            prediction_list.append(results)

        return prediction_list

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
        learning_rate = trial.suggest_float("learning_rate", 1e-2, 2e-1, log=True)
        n_epochs = trial.suggest_categorical("n_epochs", [5, 10, 15, 20])
        batch_size = trial.suggest_categorical("batch_size", [self.dataset.n_samples // 4, self.dataset.n_samples // 2, self.dataset.n_samples])
        patience = trial.suggest_categorical("patience", [50])
        clip_gamma = trial.suggest_float("clip_gamma", 0.04, 1.0)
        lr_min = trial.suggest_float("lr_min", 1e-4, 5e-4, log=True)
        lr_decay = trial.suggest_float("lr_decay", 1.0, 5.0, step=0.5)
        momentum = trial.suggest_float("momentum", 0.8, 0.99, step=0.01)

        return AGTHyperparameters(
            learning_rate=learning_rate,
            n_epochs=n_epochs,
            batch_size=batch_size,
            patience=patience,
            clip_gamma=clip_gamma,
            lr_min=lr_min,
            lr_decay=lr_decay,
            momentum=momentum
        )
