from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MovielensDataset(Dataset):
    def __init__(self, path, index, columns, values, train_ratio=.9, test_ratio=.1, train = None):
        super().__init__()
        # self.index = index
        # self.columns = columns
        # self.values = values
        self.rating_df = pd.read_csv(path)
        self.inter_matrix = self.rating_df.pivot(index=index, columns=columns, values=values).fillna(0).to_numpy()

        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

        self.total_indices = np.arange(len(self.inter_matrix))
        self.test_indices = np.random.choice(self.total_indices, size=(int(len(self.inter_matrix) * self.test_ratio),), replace=False)
        self.train_indices = np.array(list(set(self.total_indices)-set(self.test_indices)))

        if train != None:
            if train == True:
                self.inter_matrix = self.inter_matrix[self.train_indices]
            elif train == False:
                self.inter_matrix = self.inter_matrix[self.test_indices] 
    def __len__(self):
        return len(self.inter_matrix)
    def __getitem__(self, index: int):
        """
        get rating vector of one item(or user) : [0., 0., 4., 3., ..., 0., 0.]
        """
        return torch.Tensor(self.inter_matrix[index]).to(device)