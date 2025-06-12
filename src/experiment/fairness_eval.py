from datasets.dataset import BaseDataset
from datasets.acs_income import ACSIncomeDataset
from datasets.adult import AdultDataset

from mechanisms.mechanism import BaseMechanism
from mechanisms.sgd import SGDMechanism, BaseHyperparameters
from mechanisms.dpsgd import DPSGDMechanism, DPSGDHyperparameters
from mechanisms.agt_ss import AGTMechanism, AGTHyperparameters

from models.mlp import BinaryClassificationMLP, MLPHyperparameters
from models.agt_mlp import AGTBCMLP, AGTBCMLPHyperparameters

from util.privacy import PrivacyBudget, split_privacy_budget

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import json

from itertools import product

import numpy as np


@dataclass
class HyperparameterResults:
    dataset: BaseDataset
    mechanism: BaseMechanism
    mechanism_name: str
    budget: Optional[PrivacyBudget] = None


DEVICE = "cuda:1"

HYPERPARAM_BASE_FOLDER = Path("/homes/hh2721/fyp/haashim_repo/hyperparameter_results")
RESULT_BASE_FOLDER = Path("/homes/hh2721/fyp/haashim_repo/results")

HYPERPARAM_RESULTS = {
    "acs_income_dataset/acs_income_agt.json": HyperparameterResults(
        dataset=ACSIncomeDataset,
        mechanism=AGTMechanism,
        mechanism_name="agt"
    ),
    "acs_income_dataset/acs_income_dpsgd_eps0.5_delta1e-06.json": HyperparameterResults(
        dataset=ACSIncomeDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=0.5, delta=1e-6)
    ),
    "acs_income_dataset/acs_income_dpsgd_eps1.0_delta1e-06.json": HyperparameterResults(
        dataset=ACSIncomeDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=1.0, delta=1e-6)
    ),
    "acs_income_dataset/acs_income_dpsgd_eps2.0_delta1e-06.json": HyperparameterResults(
        dataset=ACSIncomeDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=2.0, delta=1e-6)
    ),
    "acs_income_dataset/acs_income_dpsgd_eps4.0_delta1e-06.json": HyperparameterResults(
        dataset=ACSIncomeDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=4.0, delta=1e-6)
    ),
    "acs_income_dataset/acs_income_dpsgd_eps10.0_delta1e-06.json": HyperparameterResults(
        dataset=ACSIncomeDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=10.0, delta=1e-6)
    ),
    "acs_income_dataset/acs_income_sgd.json": HyperparameterResults(
        dataset=ACSIncomeDataset,
        mechanism=SGDMechanism,
        mechanism_name="sgd"
    ),
    "adult_dataset/adult_agt.json": HyperparameterResults(
        dataset=AdultDataset,
        mechanism=AGTMechanism,
        mechanism_name="agt"
    ),
    "adult_dataset/adult_dpsgd_eps0.5_delta1e-05.json": HyperparameterResults(
        dataset=AdultDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=0.5, delta=1e-5)
    ),
    "adult_dataset/adult_dpsgd_eps1.0_delta1e-05.json": HyperparameterResults(
        dataset=AdultDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=1.0, delta=1e-5)
    ),
    "adult_dataset/adult_dpsgd_eps2.0_delta1e-05.json": HyperparameterResults(
        dataset=AdultDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=2.0, delta=1e-5)
    ),
    "adult_dataset/adult_dpsgd_eps4.0_delta1e-05.json": HyperparameterResults(
        dataset=AdultDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=4.0, delta=1e-5)
    ),
    "adult_dataset/adult_dpsgd_eps10.0_delta1e-05.json": HyperparameterResults(
        dataset=AdultDataset,
        mechanism=DPSGDMechanism,
        mechanism_name="dpsgd",
        budget=PrivacyBudget(epsilon=10.0, delta=1e-5)
    ),
    "adult_dataset/adult_sgd.json": HyperparameterResults(
        dataset=AdultDataset,
        mechanism=SGDMechanism,
        mechanism_name="sgd"
    ),
}

def demographic_parity(
    y_pred: np.ndarray,
    sensitive_attribute: np.ndarray,
) -> Optional[float]:
    """
    Calculates demographic parity: the absolute difference in positive prediction rate
    between two groups defined by a binary sensitive attribute.
    
    Args:
        y_pred: 1D array of predicted probabilities or binary class predictions (0/1).
        sensitive_attribute: 1D array of binary values (0 or 1) indicating group membership.
    
    Returns:
        The percentage difference in positive rate (0-100), or None if one group is empty.
    """
    if y_pred.shape != sensitive_attribute.shape:
        raise ValueError("`y_pred` and `sensitive_attribute` must have the same shape.")
    
    if not np.all(np.isin(sensitive_attribute, [0, 1])):
        raise ValueError("sensitive_attribute must contain only 0 and 1 values")
    
    y_hat = (y_pred == 1).astype(int)  # More explicit than astype(bool)
    
    # Masks for each group
    mask1 = sensitive_attribute == 1
    mask0 = sensitive_attribute == 0
    
    # If one group is empty, we can't compute parity
    if not mask1.any() or not mask0.any():
        raise ValueError(
            "Both groups defined by the sensitive attribute must have at least one member."
        )
    
    rate1 = y_hat[mask1].mean()
    rate0 = y_hat[mask0].mean()
    
    # Return absolute difference as a percentage
    return abs(rate1 - rate0) * 100


def equalised_odds(
    y_pred: np.ndarray,
    y_gt: np.ndarray,
    sensitive_attribute: np.ndarray
) -> Optional[float]:
    """
    Calculates the equalized odds metric: the sum of absolute differences in
    true positive rate and false positive rate between the two groups
    defined by a binary sensitive attribute.
    
    Args:
        y_pred: 1D array of predicted probabilities OR binary class predictions (0/1).
        y_gt: 1D array of true labels (0 or 1).
        sensitive_attribute: 1D array of sensitive attribute values (0 or 1).
    
    Returns:
        The percentage difference in TPR + FPR between groups (a float in [0, 200]),
        or None if one of the groups has no members in either the positive- or
        negative-ground-truth subsets.
    """
    # Make sure inputs align
    if not (y_pred.shape == y_gt.shape == sensitive_attribute.shape):
        raise ValueError("All inputs must have the same shape.")
    
    if not np.all(np.isin(sensitive_attribute, [0, 1])):
        raise ValueError("sensitive_attribute must contain only 0 and 1 values")
    
    # Check if either sensitive group is empty (same as demographic_parity)
    mask1 = sensitive_attribute == 1
    mask0 = sensitive_attribute == 0
    if not mask1.any() or not mask0.any():
        raise ValueError(
            "Both groups defined by the sensitive attribute must have at least one member."
        )
    
    y_hat = (y_pred == 1).astype(int)
    
    # Helper to compute a rate-difference for a given ground-truth mask
    def rate_diff(gt_mask: np.ndarray) -> Optional[float]:
        if not gt_mask.any():  # No samples with this ground truth label
            raise ValueError(
                "Ground truth mask must have at least one member for both groups."
            )
            
        sa = sensitive_attribute[gt_mask]
        preds = y_hat[gt_mask]

        p1 = preds[sa == 1].mean()
        p0 = preds[sa == 0].mean()
        return abs(p1 - p0)
    
    # True Positive Rate difference
    tpr_diff = rate_diff(y_gt == 1)
    if tpr_diff is None:
        return None
    
    # False Positive Rate difference  
    fpr_diff = rate_diff(y_gt == 0)
    if fpr_diff is None:
        return None
    
    # Return sum of diffs as percentage (FIXED: was missing * 100)
    return (tpr_diff + fpr_diff) * 100


def get_fairness_results(hyperparam_path: str) -> HyperparameterResults:
    if hyperparam_path not in HYPERPARAM_RESULTS:
        raise ValueError(f"Hyperparameter path {hyperparam_path} not found in results.")

    results = HYPERPARAM_RESULTS[hyperparam_path]
    hyperparameters_full_path = HYPERPARAM_BASE_FOLDER / hyperparam_path
    if not hyperparameters_full_path.exists():
        raise FileNotFoundError(f"Hyperparameter file {hyperparameters_full_path} does not exist.")

    with open(hyperparameters_full_path, 'r') as file:
        hyperparameters = json.load(file)

    dataset = results.dataset()

    if results.dataset == ACSIncomeDataset:
        delta = 1e-6
    else:
        delta = 1e-5

    if results.mechanism_name == "agt":
        mechanism_hyperparams = AGTHyperparameters(
            learning_rate=hyperparameters["learning_rate"],
            n_epochs=hyperparameters["n_epochs"],
            batch_size=hyperparameters["batch_size"],
            patience=hyperparameters["patience"],
            clip_gamma=hyperparameters["clip_gamma"],
            lr_min=hyperparameters["lr_min"],
            lr_decay=hyperparameters["lr_decay"],
            momentum=hyperparameters["momentum"],
        )

        model_hyperparams = AGTBCMLPHyperparameters(
            [
                hyperparameters["mlp_layer_0"],
                hyperparameters["mlp_layer_1"],
            ]
        )

        model_constructor = lambda: AGTBCMLP(
            n_features=dataset.n_features,
            hyperparameters=model_hyperparams
        )
    elif results.mechanism_name == "dpsgd":
        mechanism_hyperparams = DPSGDHyperparameters(
            learning_rate=hyperparameters["learning_rate"],
            n_epochs=hyperparameters["num_epochs"],
            batch_size=hyperparameters["batch_size"],
            patience=hyperparameters["patience"],
            max_grad_norm=hyperparameters["max_grad_norm"],
        )
        
        model_hyperparams = MLPHyperparameters(
            [
                hyperparameters["mlp_hidden_dim_l0"],
                hyperparameters["mlp_hidden_dim_l1"],
            ],
            p_dropout=hyperparameters["mlp_dropout_p"],
        )

        model_constructor = lambda: BinaryClassificationMLP(
            n_features=dataset.n_features,
            hyperparameters=model_hyperparams
        )
    elif results.mechanism_name == "sgd":
        mechanism_hyperparams = BaseHyperparameters(
            learning_rate=hyperparameters["learning_rate"],
            n_epochs=hyperparameters["n_epochs"],
            batch_size=hyperparameters["batch_size"],
            patience=hyperparameters["patience"],
        )

        model_hyperparams = MLPHyperparameters(
            [
                hyperparameters["mlp_hidden_dim_l0"],
                hyperparameters["mlp_hidden_dim_l1"],
            ],
            p_dropout=hyperparameters["mlp_dropout_p"],
        )

        model_constructor = lambda: BinaryClassificationMLP(
            n_features=dataset.n_features,
            hyperparameters=model_hyperparams
        )
    else:
        raise ValueError(f"Unknown mechanism name: {results.mechanism_name}")
    
    mechanism = results.mechanism(
        model_constructor=model_constructor,
        dataset=dataset
    )

    if results.mechanism_name == "agt" or results.mechanism_name == "sgd":
        mechanism.train(
            mechanism_hyperparams, DEVICE
        )
    elif results.mechanism_name == "dpsgd":
        mechanism.train(
            mechanism_hyperparams, results.budget, DEVICE
        )
    else:
        raise ValueError(f"Unknown mechanism name: {results.mechanism_name}")

    print("Training complete.")

    if results.mechanism_name == "agt":
        queries_numbers = [5, 10, 50, 100, 200, 500, 1000]
        eps_budgets = [0.5, 1.0, 2.0, 4.0, 10.0]

        outputs = {}
        
        for n_queries, eps_budget in product(queries_numbers, eps_budgets):
            slack_ratio = 1/n_queries
            per_query_budget = split_privacy_budget(
                PrivacyBudget(epsilon=eps_budget, delta=delta),
                n_queries,
                slack_ratio=slack_ratio,
            )
            print(f"Using slack_ratio={slack_ratio}")
            # per_query_budget = PrivacyBudget(epsilon=eps_budget, delta=delta)
            print(f"Evaluating for n_query={n_queries}, eps_budget={eps_budget}, per query budget {per_query_budget}")
            dem_parity_list = []
            equal_odds_list = []
            accuracy_list = []

            skipped_predictions = 0
            predictions = mechanism.predict(n_queries, per_query_budget, DEVICE)
            for prediction in predictions:
                preds, true_labels, prot_attrs = prediction
                preds, true_labels, prot_attrs = (
                    np.array(preds),
                    np.array(true_labels),
                    np.array(prot_attrs)
                )
                demo_parity = demographic_parity(preds, prot_attrs)
                equal_odds = equalised_odds(preds, true_labels, prot_attrs)
                accuracy = (preds == true_labels).mean()

                if demo_parity is not None and equal_odds is not None:
                    dem_parity_list.append(demo_parity)
                    equal_odds_list.append(equal_odds)
                    accuracy_list.append(accuracy)
                else:
                    skipped_predictions += 1

            print(f"Skipped {skipped_predictions}/{len(predictions)} predictions due to empty groups.")
            dem_parity_mean = float(np.mean(dem_parity_list))
            dem_parity_std = float(np.std(dem_parity_list))
            equal_odds_mean = float(np.mean(equal_odds_list))
            equal_odds_std = float(np.std(equal_odds_list))
            accuracy_mean = float(np.mean(accuracy_list))
            accuracy_std = float(np.std(accuracy_list))

            print(f"Demographic Parity: {dem_parity_mean} ± {dem_parity_std}")
            print(f"Equalised Odds: {equal_odds_mean} ± {equal_odds_std}")
            print(f"Accuracy: {accuracy_mean} ± {accuracy_std}")

            outputs[f"{n_queries}_{eps_budget}"] = {
                "demographic_parity": {
                    "mean": dem_parity_mean,
                    "std": dem_parity_std
                },
                "equalised_odds": {
                    "mean": equal_odds_mean,
                    "std": equal_odds_std
                },
                "accuracy": {
                    "mean": accuracy_mean,
                    "std": accuracy_std
                }
            }


    elif results.mechanism_name == "dpsgd" or results.mechanism_name == "sgd":
        predictions, true_labels, protected_attrs = mechanism.predict(DEVICE)

        print(len(predictions), len(true_labels), len(protected_attrs))
        print(predictions[:5], true_labels[:5], protected_attrs[:5])

        outputs = {}

        # convert to numpy
        predictions, true_labels, protected_attrs = (
            np.array(predictions),
            np.array(true_labels),
            np.array(protected_attrs)
        )

        outputs["demographic_parity"] = demographic_parity(predictions, protected_attrs)
        outputs["equalised_odds"] = equalised_odds(predictions, true_labels, protected_attrs)
        outputs["accuracy"] = (predictions == true_labels).mean()

        print(f"Demographic Parity: {outputs['demographic_parity']}"
              f"\nEqualised Odds: {outputs['equalised_odds']}"
              f"\nAccuracy: {outputs['accuracy']}")

    save_path = RESULT_BASE_FOLDER / hyperparam_path
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to json file
    with open(save_path, "w") as f:
        json.dump(outputs, f)

    print("Done")
