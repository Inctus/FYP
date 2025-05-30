"""
A collection of constants used throughout the project.
"""

# For reproducibility, we set a random seed.
RANDOM_SEED = 69420

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
# The splits should sum to 1.0
assert TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT == 1.0, \
    "Train, validation, and test splits must sum to 1.0"

