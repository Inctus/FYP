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


def split_privacy_budget(privacy_budget: PrivacyBudget, k=1):
        delta = 1 - (1 - privacy_budget.delta) ** (1 / k)
        # delta = 1 - np.exp((np.log(1 - self.delta) - np.log(1 - spent_delta)) / k)

        lower = 0
        upper = privacy_budget.epsilon
        old_interval_size = (upper - lower) * 2

        while old_interval_size > upper - lower:
            old_interval_size = upper - lower
            mid = (upper + lower) / 2

            spent_budget = [(mid, 0)] * k
            cost = total(spent_budget=spent_budget, slack=privacy_budget.delta/2)

            if cost.epsilon >= privacy_budget.epsilon:
                upper = mid
            else:
                lower = mid

        epsilon = (upper + lower) / 2

        return PrivacyBudget(epsilon, delta)
