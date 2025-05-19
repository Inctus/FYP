from aif360.datasets import AdultDataset as Aif360AdultDataset

from dataset import BaseDataset

class AdultDataset(BaseDataset):
    """
    Adult dataset class for loading and preprocessing Adult dataset
    """

    def __init__(self):
        super().__init__()

    def load_data(self):
        """
        Load the Adult dataset
        """
        
        return Aif360AdultDataset(
            ... # Need to choose KWARGS but Idk yet
        )