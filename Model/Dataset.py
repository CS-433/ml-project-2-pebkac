"""
This script defines a Dataset object, which is used in the training phase
"""

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        """
        Initializes the dataset with the specified fold and set type.

        Parameters:
        - data_dict (dict): The data dictionary containing the folds
        """
        self.Z = data_dict["Z"].transpose(0, 1) 
        self.W = data_dict["W"] 

    def __len__(self):
        """
        Returns the number of sequences in the dataset.
        """
        return len(self.Z)
    
    def __getitem__(self, idx):
        """
        Returns the sequence and its corresponding weight for a given index.

        Parameters:
        - idx (int): Index of the sequence to retrieve

        Returns:
        - tuple: (sequence, weight) at the specified index
        """
        return self.Z[idx], self.W[idx]