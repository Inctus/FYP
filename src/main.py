import argparse

from datasets.adult import AdultDataset
from datasets.acs_travel import ACSTravelTimeDataset
from datasets.acs_income import ACSIncomeDataset

from mechanisms.sgd import SGDMechanism
from mechanisms.dpsgd import DPSGDMechanism
from mechanisms.agt_ss import AGTMechanism
from models.mlp import BinaryClassificationMLP
from models.agt_mlp import AGTBCMLP

from experiment.hyperparameter_tuner import HyperparameterTuner


DATASET_CHOICES = {
    "adult": AdultDataset,
    "acs_travel": ACSTravelTimeDataset,
    "acs_income": ACSIncomeDataset,
}

MECHANISM_CHOICES = {
    "sgd": SGDMechanism,
    "dpsgd": DPSGDMechanism,
    "agt": AGTMechanism
}

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for specified dataset.")
    parser.add_argument(
        "dataset_name",
        type=str,
        choices=DATASET_CHOICES.keys(),
        help="Name of the dataset to use."
    )
    parser.add_argument(
        "device",
        type=str,
        choices=["cpu", "cuda:0", "cuda:1"],
        help="Device to run the tuning on (e.g., 'cpu', 'cuda:0', 'cuda:1')."
    )
    parser.add_argument(
        "mechanism",
        type=str,
        choices=["sgd", "dpsgd", "agt"],
        help="Mechanism to use for training (e.g., 'sgd' or 'dpsgd' or 'agt')."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
        help="Privacy budget epsilon for differential privacy mechanisms"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=None,
        help="Privacy budget delta for differential privacy mechanisms"
    )

    args = parser.parse_args()

    if args.mechanism == "dpsgd" and (args.eps is None or args.delta is None):
        parser.error("--eps and --delta are required for DPSGD mechanism.")

    dataset_class = DATASET_CHOICES[args.dataset_name]
    dataset = dataset_class()
    print(f"Loaded dataset: {args.dataset_name} with n_features: {dataset.n_features}")

    if args.eps is not None and args.delta is not None:
        from util.privacy import PrivacyBudget
        privacy_budget = PrivacyBudget(epsilon=args.eps, delta=args.delta)
        study_name = f"{args.dataset_name}_{args.mechanism}_eps{args.eps}_delta{args.delta}"
    else:
        privacy_budget = None
        study_name = f"{args.dataset_name}_{args.mechanism}"

    print(f"Starting tuning with study name: {study_name}")

    if args.mechanism == "dpsgd":
        tuner = HyperparameterTuner(
            mechanism_class=MECHANISM_CHOICES[args.mechanism],
            model_class=BinaryClassificationMLP,
            dataset=dataset,
            device=args.device,  # Use the device specified in the command line arguments
            privacy_budget=privacy_budget
        )
        tuner.tune(50, study_name)
    elif args.mechanism == "sgd":
        tuner = HyperparameterTuner(
            mechanism_class=MECHANISM_CHOICES[args.mechanism],
            model_class=BinaryClassificationMLP,
            dataset=dataset,
            device=args.device,  # Use the device specified in the command line arguments
        )
        tuner.tune(
            n_trials=200,  # You might want to make this configurable too
            study_name=study_name
        )
    elif args.mechanism == "agt":
        tuner = HyperparameterTuner(
            mechanism_class=MECHANISM_CHOICES[args.mechanism],
            model_class=AGTBCMLP,
            dataset=dataset,
            device=args.device,  # Use the device specified in the command line arguments
        )
        tuner.tune(
            n_trials=100,  # You might want to make this configurable too
            study_name=study_name
        )
    
    print(f"Finished tuning for {study_name}")

if __name__ == "__main__":
    main()