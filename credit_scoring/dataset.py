import torch
import pandas as pd

from typing import Dict

class CreditDataset:
    def __init__(self, 
                 data: Dict, 
                 flags: pd.DataFrame = None,
                 category: str ='train'):
        self.data = data
        self.flags = flags
        self.category = category

    def __getitem__(self, 
                    idx: int):
        if self.category in ('train', 'eval'):
            return (
                torch.tensor(self.data[idx]),
                self.flags[idx]
            )
        elif self.category == 'test':
            return torch.tensor(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)
