from dataclasses import dataclass

from typing import Tuple

import numpy as np


@dataclass
class PrivacyBudget:
    """
    Represents a privacy budget for differential privacy.
    
    Attributes:
        epsilon (float): The privacy budget parameter, typically denoted as ε.
        delta (float): The privacy budget parameter, typically denoted as δ.
    """
    epsilon: float
    delta: float

    def __post_init__(self):
        if self.epsilon < 0:
            raise ValueError("Epsilon must be non-negative.")
        
        if self.delta < 0 or self.delta > 1:
            raise ValueError("Delta must be in the range [0, 1].")
    
    def split_epsilon(self, factor: float) -> Tuple["PrivacyBudget", "PrivacyBudget"]:
        """
        Splits the privacy budget into two parts based on a factor.
        
        Args:
            factor (float): The factor by which to split the budget. 
                            Must be between 0 and 1.
        
        Returns:
            Tuple[PrivacyBudget, PrivacyBudget]: Two new PrivacyBudget instances.
        """
        if not (0 < factor < 1):
            raise ValueError("Factor must be between 0 and 1.")
        
        new_epsilon = self.epsilon * factor

        return PrivacyBudget(new_epsilon, self.delta), PrivacyBudget(self.epsilon - new_epsilon, self.delta)
    


def __total_delta_safe(spent_budget, slack):
    delta_spend = [slack]
    for _, delta in spent_budget:
        delta_spend.append(delta)
    delta_spend.sort()

    # (1 - a) * (1 - b) = 1 - (a + b - a * b)
    prod = 0
    for delta in delta_spend:
        prod += delta - prod * delta

    return prod

def total(spent_budget, slack):
        epsilon_sum, epsilon_exp_sum, epsilon_sq_sum = 0, 0, 0

        for epsilon, _ in spent_budget:
            epsilon_sum += epsilon
            epsilon_exp_sum += (1 - np.exp(-epsilon)) * epsilon / (1 + np.exp(-epsilon))
            epsilon_sq_sum += epsilon ** 2

        total_epsilon_naive = epsilon_sum
        total_delta = __total_delta_safe(spent_budget, slack)

        if slack == 0:
            return PrivacyBudget(total_epsilon_naive, total_delta)

        total_epsilon_drv = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum * np.log(1 / slack))
        total_epsilon_kov = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum *
                                                      np.log(np.exp(1) + np.sqrt(epsilon_sq_sum) / slack))

        return PrivacyBudget(min(total_epsilon_naive, total_epsilon_drv, total_epsilon_kov), total_delta)


def split_privacy_budget(
    privacy_budget: PrivacyBudget,
    k: int = 1,
    *,                       # keyword-only
    slack_ratio: float = 0.05
) -> PrivacyBudget:
    """
    Evenly splits an (ε, δ) budget across *k* compositions, reserving
    `slack_ratio·δ` for the “slack” term used in the advanced-composition
    theorem (Dwork & Roth, 2014).

    Returns
    -------
    PrivacyBudget
        (ε_i, δ_i) that can be used once; k copies compose to ≤
        (privacy_budget.epsilon, privacy_budget.delta).
    """
    if not (0 < slack_ratio < 1):
        raise ValueError("slack_ratio must be in (0, 1).")

    slack   = privacy_budget.delta * slack_ratio      # δ′
    delta_i = (privacy_budget.delta - slack) / k      # per-step δ

    # ---- binary-search ε_i --------------------------------------------------
    lower, upper, tol = 0.0, privacy_budget.epsilon, 1e-12
    while upper - lower > tol:
        mid  = (upper + lower) / 2
        cost = total([(mid, delta_i)] * k, slack=slack)
        if cost.epsilon > privacy_budget.epsilon:
            upper = mid
        else:
            lower = mid

    eps_i = (upper + lower) / 2
    return PrivacyBudget(eps_i, delta_i)
