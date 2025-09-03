import torch
import pandas as pd

from typing import List
from typing_custom import DataType


class CreditDataset:
    """Dataset for `CreditRNNModel`"""
    def __init__(self, 
                 data: List, 
                 flags: List = None,
                 category: str = 'train'):
        """Initializing class object

        Args:
            data (List): _description_
            flags (List, optional): _description_. Defaults to None.
            category (str, optional): _description_. Defaults to 'train'.
        """
        self.data = data
        self.flags = flags
        self.category = category

    def __getitem__(self, 
                    idx: int) -> DataType:
        """Get data by one ID from dataset.

        Args:
            idx (int): Id

        Returns:
            DataType: If category is 'train' or 'eval', returns 
                a tuple of two elements: torch.Tensor with features 
                and np.array of targets. If category is 'test', 
                returns only one torch.Tensor with test features.
        """
        if self.category in ('train', 'eval'):
            return (
                torch.tensor(self.data[idx]),
                self.flags[idx]
            )
        elif self.category == 'test':
            return torch.tensor(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)
