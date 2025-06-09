
import math
from typing import Tuple, List

def per_run_budget_rdp(target_eps: float,
                       target_delta: float,
                       gamma: float,
                       delta_run: float | None = None,
                       alphas: List[float] | None = None,
                       tol: float = 1e-6,
                       max_iter: int = 100) -> Tuple[float, float]:
    """
    Allocate a per-run (ε_run, δ_run) so that the *whole*
    best-of-K routine (K ~ logarithmic, η = 0) satisfies
    (target_eps, target_delta)-DP            - Theorem 2, η = 0.

    Returns
    -------
    eps_run, delta_run
    """
    if delta_run is None:
        delta_run = target_delta

    # Opacus’ default RDP α-grid (1.1, 1.2, …, 10.0, 12, …, 63)
    if alphas is None:
        alphas = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64))

    log1g = math.log(1 / gamma)
    EK    = (1 / gamma - 1) / log1g           # mean of log-series

    def overall_eps(eps_run: float) -> float:
        best = float("inf")
        for lam in alphas:
            if lam <= 1:
                continue
            # Theorem 2: add γ-dependent RDP penalties
            eps_total_rdp = eps_run + log1g / lam + math.log(EK) / (lam - 1)
            # RDP → (ε,δ)  (Mironov 2017, Lemma 3)
            eps_dp = eps_total_rdp + math.log(1 / target_delta) / (lam - 1)
            best = min(best, eps_dp)
        return best

    # binary-search for the largest ε_run that keeps overall ε ≤ target_eps
    lo, hi = 1e-6, target_eps
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if overall_eps(mid) > target_eps:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2, delta_run

import optuna
import json
from pathlib import Path
from typing import Dict, Any

from mechanisms.mechanism import DPLearningMechanism
from datasets.dataset import BaseDataset

from util.constants import HYPERPARAMETER_RESULTS_DIR, RANDOM_SEED
from util.privacy import PrivacyBudget

from scipy.stats import logser   # SciPy ≥ 1.8


MAX_TRIALS = 100   # hard cap on Optuna trials  – improves runtime *and* privacy


class DPHyperparameterTuner:
    """
    Differentially-private hyper-parameter sweep using the logarithmic series
    distribution (η = 0).  The per-run privacy budget is computed *automatically*
    from the overall (ε,δ) and γ, accounting for the extra privacy loss due to
    repeating the run K ~ D_{0,γ}.
    """
    # ────────────────────────────────────────────────────────────────
    def __init__(self,
                 mechanism_class: DPLearningMechanism,
                 dataset: BaseDataset,
                 device: str,
                 privacy_budget: PrivacyBudget):
        self.mechanism_class   = mechanism_class
        self.dataset           = dataset
        self.device            = device
        self.storage_base_dir  = Path(HYPERPARAMETER_RESULTS_DIR)

        # keep the *global* budget – per-run allocation happens inside tune()
        self.global_privacy_budget = privacy_budget
        self.privacy_budget: PrivacyBudget | None = None   # set later

    # ────────────────────────────────────────────────────────────────
    def _sample_logarithmic_distribution(self, gamma: float) -> int:
        """
        One draw K ~ D_{0,γ}.  The SciPy log-series parameter p = 1 - γ.
        """
        k = logser.rvs(1 - gamma)
        if k > MAX_TRIALS:
            print(f"Sampled n_trials {k} exceeds MAX_TRIALS {MAX_TRIALS}. Clamping.")
            k = MAX_TRIALS
        elif k < 1:
            raise ValueError(f"Sampled n_trials {k} < 1.  Check γ or MAX_TRIALS.")
        return k

    def _set_per_run_budget(self, gamma: float):
        """Compute and cache the per-run PrivacyBudget for this sweep."""
        eps_run, delta_run = per_run_budget_rdp(
            target_eps   = self.global_privacy_budget.epsilon,
            target_delta = self.global_privacy_budget.delta,
            gamma        = gamma,
        )
        print(f"Per-run budget: ε = {eps_run:.4f}, δ = {delta_run}")
        self.privacy_budget = PrivacyBudget(epsilon=eps_run, delta=delta_run)

    def objective(self, trial: optuna.Trial) -> float:
        """One Optuna trial under the *per-run* DP budget."""
        if self.privacy_budget is None:
            raise RuntimeError("Per-run privacy budget not initialised.")

        mechanism_instance = self.mechanism_class(dataset=self.dataset)
        mechanism_hparams  = mechanism_instance.suggest_hyperparameters(trial)

        try:
            results  = mechanism_instance.train(
                hyperparameters = mechanism_hparams,
                privacy_budget  = self.privacy_budget,
                device          = self.device,
            )
            accuracy = results.accuracy
            print(f"Trial {trial.number} finished: {accuracy=:.4f}")
            return accuracy                                  # maximise
        except Exception as exc:
            print(f"Trial {trial.number} pruned: {exc}")
            raise optuna.exceptions.TrialPruned()

    def tune(self, gamma: float, study_name: str) -> Dict[str, Any]:
        """
        Run the DP sweep.  γ controls the logarithmic-series tail
        (smaller γ ⇒ more expected repeats ⇒ larger privacy penalty).
        """
        if not (0 < gamma < 1):
            raise ValueError(f"Invalid γ = {gamma}. Must be in (0, 1).")
        self._set_per_run_budget(gamma)

        n_trials = self._sample_logarithmic_distribution(gamma)
        print(f"Running {n_trials} DP trials (γ = {gamma})")

        sampler = optuna.samplers.RandomSampler(seed=RANDOM_SEED)
        study   = optuna.create_study(direction="maximize",
                                      study_name=study_name,
                                      sampler=sampler)

        study.optimize(self.objective, n_trials=n_trials)

        best_trial = study.best_trial
        print(f"\nBest trial for '{study_name}': val-acc = {best_trial.value:.4f}")
        self.save_hyperparameters(best_trial.params, study_name)
        return best_trial.params

    def save_hyperparameters(self, hyperparams: Dict[str, Any], study_name: str):
        self.storage_base_dir.mkdir(parents=True, exist_ok=True)
        save_path = self.storage_base_dir / f"{study_name}.json"
        with open(save_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
        print(f"Best hyper-parameters saved to {save_path}")