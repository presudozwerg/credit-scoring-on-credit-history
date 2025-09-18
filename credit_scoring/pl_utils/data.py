from omegaconf import DictConfig
from typing import List

from dvc_utils.dvc_load import download_dvc_data, only_dvc_in_dir
import os
import pl_utils.dataloader_utils as loader_utils
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .preprocessing import Preprocesser
from .typing_custom import DataType

class CreditDataset:
    """Dataset for `CreditRNNModel`"""

    def __init__(self, data: List, flags: List = None, category: str = "train"):
        """Initializing class object

        Args:
            data (List): _description_
            flags (List, optional): _description_. Defaults to None.
            category (str, optional): _description_. Defaults to 'train'.
        """
        self.data = data
        self.flags = flags
        self.category = category

    def __getitem__(self, idx: int) -> DataType:
        """Get data by one ID from dataset.

        Args:
            idx (int): Id

        Returns:
            DataType: If category is 'train' or 'eval', returns
                a tuple of two elements: torch.Tensor with features
                and np.array of targets. If category is 'test',
                returns only one torch.Tensor with test features.
        """
        if self.category in ("train", "eval"):
            return (torch.tensor(self.data[idx]), self.flags[idx])
        elif self.category == "test":
            return torch.tensor(self.data[idx])

    def __len__(self) -> int:
        return len(self.data)


class CreditDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config.train
        self.load_conf = config.data_load
        self.preproc = Preprocesser(config.preprocess)

    def prepare_data(self):
        if only_dvc_in_dir(self.load_conf.dvc_data):
            print(f'Downloading data from {self.load_conf.gdrive_id}...\n')
            os.system(
                f"gdown {self.load_conf.gdrive_id} -O {self.load_conf.zip_path} --quiet"
            )
            os.system(f"unzip {self.load_conf.zip_path} -d {self.load_conf.raw_data}")
            os.system(f"rm -rf {self.load_conf.zip_path}")
            download_dvc_data(self.load_conf.dvc_data)
        pass

    def setup(self, stage: str):
        if stage == "fit":
            all_features = self.preproc.read_train_test_features("train")
            tr_target = self.preproc.read_train_target()

            tr_features, val_features, tr_target, val_target = train_test_split(
                list(all_features.values()), 
                tr_target["flag"].values, 
                test_size=self.config.val_size
            )
            del all_features

            self.train_dataset = CreditDataset(tr_features, tr_target)
            self.val_dataset = CreditDataset(val_features, val_target)
        if stage == "predict":
            te_features = self.preproc.read_train_test_features("test")
            self.pred_ids = te_features.keys()
            self.pred_dataset = CreditDataset(list(te_features.values()), category="test")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=loader_utils.collate_fn_tr,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=loader_utils.collate_fn_tr,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.pred_dataset,
            collate_fn=loader_utils.collate_fn_te,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
