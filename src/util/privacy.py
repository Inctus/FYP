from dataclasses import dataclass

from typing import Tuple


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