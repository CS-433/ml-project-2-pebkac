################
#Based on the article https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
################

import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, fold_num, train_or_test):
        """
        Initializes the dataset with the specified fold and set type.

        Parameters:
        - data_dict (dict): The data dictionary containing the folds
        - fold_num (int): The fold number to use (1-based index)
        - train_or_test (str): Whether to use the 'Train' or 'Test' set
        """
        self.Z = data_dict[str(fold_num)][train_or_test]["Z"]  # Alignment matrix
        self.W = data_dict[str(fold_num)][train_or_test]["W"]  # Weight vector

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