################
#Based on the article https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
################
"""
Cet objet n'est pas super utile, utilise un objet Dataloader pytorch directement
"""


import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_ids, file_path, weights):
        ### Faut il ajouter le type de dataset qu'on cree ex 
        self.list_ids = list_ids
        self.weights = weights
        self.file_path = file_path


    def __len__(self):
        return len(self.list_ids)
    
    def __getitem__(self, index):
        id = self.list_ids[index]
        X = torch.load(self.file_path + id + ".pt")
        weight = self.weights[id]

        return X, weight