from dataclasses import dataclass, asdict, field
from typing import List

import numpy as np
import pandas as pd

from util.constants import RANDOM_SEED


@dataclass(frozen=True)
class DatasetConfig:
    n_samples: int = int(1e6)
    n_features: int = 5
    group_values: List[int] = field(default_factory=lambda: [0, 1])
    group_probabilities: List[float] = field(default_factory=lambda: [0.55, 0.45])
    group_target_probabilities: List[float] = field(default_factory=lambda: [0.5, 0.3])
    neg_to_pos_target_flip_prob: List[float] = field(default_factory=lambda: [0.15, 0.1])
    pos_to_neg_target_flip_prob: List[float] = field(default_factory=lambda: [0.1, 0.15])
    eps: List[float] = field(default_factory=lambda: list(np.arange(5) / 10.0))
    random_seed: int = RANDOM_SEED


class DatasetGenerator:
    def __init__(self, config: DatasetConfig):
        self.config = config

    def generate_dataset(self):
        rnd = np.random.RandomState(self.config.random_seed)
        # Generate an array of sensitive attribute values
        if sum(self.config.group_probabilities) != 1:
            raise ValueError('Sum of group_probabilities should be equal to 1')

        sensitive_attributes = rnd.choice(self.config.group_values, size=self.config.n_samples,
                                          p=self.config.group_probabilities)
        # Generate outcomes based on sensitive attribute values
        group_probabilities = dict(zip(self.config.group_values, self.config.group_target_probabilities))
        outcomes = rnd.rand(self.config.n_samples)
        for group in self.config.group_values:
            group_mask = sensitive_attributes == group
            outcomes[group_mask] = np.where(outcomes[group_mask] < group_probabilities[group], 1, 0)

        # Flip outcomes based on sensitive attribute values
        flipped_outcomes = outcomes.copy()
        flip_rnd_value = rnd.rand(self.config.n_samples)
        neg_mask = outcomes == 0
        pos_mask = ~neg_mask
        for group in self.config.group_values:
            group_mask = sensitive_attributes == group

            neg_threshold = self.config.neg_to_pos_target_flip_prob[self.config.group_values.index(group)] # Assuming order matches
            flipped_outcomes[group_mask & neg_mask & (flip_rnd_value < neg_threshold)] = 1

            pos_threshold = self.config.pos_to_neg_target_flip_prob[self.config.group_values.index(group)] # Assuming order matches
            flipped_outcomes[group_mask & pos_mask & (flip_rnd_value < pos_threshold)] = 0

        # Generate additional features
        # copy outcome n_feature times in features array
        features = np.tile(flipped_outcomes, (self.config.n_features, 1)).T
        rnd_value = rnd.rand(self.config.n_samples, self.config.n_features)
        # copy eps n_outcomes times in eps array
        eps_array = np.tile(self.config.eps, (self.config.n_samples, 1)) # Corrected: use self.config.eps
        equal_mask = rnd_value < (
                0.5 + eps_array)  # derived from Y_i with a probability of 1/2 + eps_i, and its complement(1 −Y_i ) with the remaining probability
        complement_mask = ~equal_mask
        features[complement_mask] = 1 - features[complement_mask]

        # Create DataFrame
        data = np.column_stack((features, flipped_outcomes, sensitive_attributes))
        columns = [f'Feature_{i + 1}' for i in range(self.config.n_features)] + ['Outcome'] + ['Sensitive Attribute']
        # dtypes int
        df = pd.DataFrame(data, columns=columns, dtype=int)

        return df
