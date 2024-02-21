from torch.utils.data import TensorDataset, DataLoader, random_split
import torch

import os

class AnglesDataset:
    
    def __init__(self, data: torch.Tensor) -> None:
        """Initialize the dataset from a .pt file that contains a tensor"""

        # Shape: (n_samples, (x,y,[z])), i.e. (n_samples, 2 or 3)
        self.dataset = TensorDataset(data)

    def split(self, train_size: float = .8) -> None:
        """Split the datasets into train and test"""

        self.train_dataset, self.test_dataset = random_split(self.dataset, lengths=(train_size, 1 - train_size))

    def get_dataloaders(self, batch_size: int = 100) -> None:
        """Get a training and testing dataloader from the split dataset"""
        
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=batch_size)
