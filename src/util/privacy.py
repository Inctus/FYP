from dataclasses import dataclass


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