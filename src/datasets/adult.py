from aif360.datasets import AdultDataset as Aif360AdultDataset

from datasets.dataset import BaseDataset


class AdultDataset(BaseDataset):
    """
    Adult dataset class for loading and preprocessing Adult dataset.
    
    This class handles the specific configuration needed for the Adult/Census Income
    dataset, including proper handling of categorical features and protected attributes.
    """

    def __init__(self):
        super().__init__()  # This calls BaseDataset.__init__() which calls load_data()

    def load_data(self):
        """
        Load and configure the Adult dataset from AIF360.

        The configuration here follows fairness research best practices:
        - 'sex' is the primary protected attribute for fairness evaluation
        - Males are considered the privileged group
        - Categorical features are properly encoded
        - Missing values (marked as '?') are handled by dropping rows
        """
        adult_ds = Aif360AdultDataset(
            protected_attribute_names=['sex'],  # Primary protected attribute
            privileged_classes=[['Male']],      # Privileged group definition
            categorical_features=['workclass', 'education', 'marital-status',
                                'occupation', 'relationship', 'race', 'native-country'],
            features_to_keep=['age', 'workclass', 'education', 'education-num',
                            'marital-status', 'occupation', 'relationship', 'race',
                            'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                            'native-country'],
            na_values=['?'],  # Handle missing values
            custom_preprocessing=lambda df: df.dropna()  # Simple approach: drop missing values
        )

        return adult_ds