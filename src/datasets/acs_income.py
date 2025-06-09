import os
from folktables import ACSDataSource, ACSIncome

from aif360.datasets import StandardDataset
from datasets.dataset import BaseDataset # Assuming this is in the parent directory or PYTHONPATH
from util.constants import FOLKTABLES_DATA_PATH

import pandas as pd

class ACSIncomeDataset(BaseDataset):
    """
    Simplified ACS Income Dataset class for loading and preprocessing data from Folktables.

    This class is configured for:
    - Survey Year: 2018
    - Horizon: 1-Year
    - Survey: Person
    - Sensitive Attribute: Sex (Male privileged)
    - Data Path: /vol/bitbucket/hh2721/folktables
    - Download Data: False (assumes data is pre-downloaded)

    It handles loading ACS Income data and applying standard preprocessing steps
    like one-hot encoding via AIF360's StandardDataset.

    Folktables features for ACSIncome:
    ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
    Target: PINCP (Income) > $50,000 (binary)
    Protected Attribute: SEX (1 for Male, 2 for Female. Privileged is Male (1.0)).
    """

    _STATES = ["CA"]
    _SURVEY_YEAR = '2018'
    _HORIZON = '1-Year'
    _SURVEY = 'person'
    _SENSITIVE_ATTRIBUTE_NAME = 'SEX' # Folktables column name
    _DOWNLOAD_DATA = False

    def __init__(self):
        if not os.path.exists(FOLKTABLES_DATA_PATH):
            # If data is not downloaded and download is False, this will likely fail later.
            # Consider adding a more robust check or instruction.
            print(f"Warning: Folktables data path {FOLKTABLES_DATA_PATH} does not exist. "
                  "Data loading might fail as download_data is False.")
        
        self.features_dim = None # Will be populated after load_data

        super().__init__() # Calls BaseDataset.__init__ -> self.load_data()
        
        if self._aif360_dataset_original:
            self.features_dim = self._aif360_dataset_original.features.shape[1]

    def load_data(self) -> StandardDataset:
        """
        Loads and preprocesses ACS Income data using Folktables and AIF360 StandardDataset
        with fixed configuration (2018, 1-Year, 'sex' as SA).
        """
        data_source = ACSDataSource(survey_year=self._SURVEY_YEAR,
                                    horizon=self._HORIZON,
                                    survey=self._SURVEY,
                                    root_dir=FOLKTABLES_DATA_PATH)
        
        try:
            acs_data = data_source.get_data(states=self._STATES, download=self._DOWNLOAD_DATA)
        except Exception as e:
            print(f"Error loading Folktables data: {e}")
            print(f"Please ensure data for states {self._STATES}, year {self._SURVEY_YEAR}, horizon {self._HORIZON} "
                  f"is available at {FOLKTABLES_DATA_PATH} (download is set to False).")
            raise

        features_df, labels_series, _ = ACSIncome.df_to_pandas(acs_data)

        # Print imbalance in labels
        print("Label distribution:")
        print(labels_series.value_counts(normalize=True))
        print("Label imbalance (should be around 0.5 for balanced data):")
        print(labels_series.value_counts(normalize=True).get(1, 0.0),
                labels_series.value_counts(normalize=True).get(0, 0.0))
        
        df = features_df.copy()
        # We need to map from Folktables to AIF360's expected format:
        # Folktables uses 1 for Male and 2 for Female but AIF360 expects
        # 1.0 for privileged and 0.0 for unprivileged. Since this is a numeric column it isn't mapped
        # automatically by AIF360, we need to do it manually.
        df[self._SENSITIVE_ATTRIBUTE_NAME] = df[self._SENSITIVE_ATTRIBUTE_NAME].map({1: 1.0, 2: 0.0})

        label_name = 'PINCP_GT_50K'
        df[label_name] = labels_series.astype(float)
        favorable_classes = [1.0]

        # Protected attribute is 'SEX' (1.0=Male, 0.0=Female) since we mapped it above.
        # Male (1.0) is privileged.
        protected_attribute_names = [self._SENSITIVE_ATTRIBUTE_NAME]
        privileged_classes = [[1.0]] 

        # ACSIncome.features: ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
        # 'SEX' is the protected attribute. 'RAC1P' will be a regular categorical feature.
        categorical_features_for_encoding = ['COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'RAC1P']
        new_ohe_columns_data = {} # Dictionary to hold new column Series

        for col_name in categorical_features_for_encoding:
            if col_name not in df.columns: # Check if original column exists
                # This check was already there, good for robustness
                raise ValueError(f"Expected categorical column '{col_name}' not found in data.")
            
            original_series = df[col_name] # Get the original column Series
            top_cats = original_series.value_counts().nlargest(20).index
            
            for cat_value in top_cats:
                print(f"OHE category '{cat_value}' for column '{col_name}'")
                new_ohe_col_name = f"{col_name}_{cat_value}"
                new_ohe_columns_data[new_ohe_col_name] = (original_series == cat_value).astype(float)
            
            other_series_mask = ~original_series.isin(top_cats)
            if other_series_mask.any(): # Check if any row falls into the 'other' category
                other_col_name = f"{col_name}_other"
                new_ohe_columns_data[other_col_name] = other_series_mask.astype(float)
                print(f"Created 'other' category for column '{col_name}' with {other_series_mask.sum()} entries.")
            else:
                print(f"No 'other' category needed for column '{col_name}' as all values are in top 20.")

        # Create a new DataFrame from all the generated OHE columns
        if new_ohe_columns_data: # Check if any OHE columns were generated
            df_ohe_new = pd.DataFrame(new_ohe_columns_data, index=df.index)

            # Drop the original categorical columns that were encoded
            df.drop(columns=categorical_features_for_encoding, inplace=True)

            # Concatenate the original DataFrame (now without raw categorical features) 
            # with the new DataFrame of OHE features
            df = pd.concat([df, df_ohe_new], axis=1)
            
        # Ensure all columns exist
        all_expected_cols = protected_attribute_names + ['AGEP', 'WKHP']
        for col in all_expected_cols:
            if col not in df.columns:
                raise ValueError(f"Expected column '{col}' not found in ACSIncome features for the given configuration.")

        default_metadata = self._get_default_metadata()

        print(df.columns)

        dataset = StandardDataset(
            df=df,
            label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=None,
            categorical_features=[],
            features_to_keep=[], 
            features_to_drop=[],
            na_values=[], 
            custom_preprocessing=None, 
            metadata=default_metadata
        )
        
        return dataset

    def _get_default_metadata(self):
        """Provides default metadata for labels and the 'sex' protected attribute."""
        label_map = [{1.0: 'Income > $50K', 0.0: 'Income <= $50K'}]
        # For 'SEX': Male (1.0) is privileged. Female (2.0) is unprivileged.
        # AIF360's StandardDataset handles numeric protected attributes by keeping original values
        # but using privileged_classes to define groups. The metadata map reflects the
        # conceptual 0/1 mapping for fairness metrics.
        protected_attribute_map = [{1.0: 'Male', 0.0: 'Female'}] 

        return {
            'label_maps': label_map,
            'protected_attribute_maps': protected_attribute_map
        }
