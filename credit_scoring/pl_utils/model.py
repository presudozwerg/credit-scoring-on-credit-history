from typing import Any

import pytorch_lightning as pl
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR

from pl_utils.credit_rnn_model import CreditRNNModel


class CreditModel(pl.LightningModule):
    """Module for training and evaluation
    models for the scoring task
    """

    def __init__(
        self, model: CreditRNNModel, lr: float, sched_step_size: int, sched_gamma: float
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = lr
        self.sched_step_size = sched_step_size
        self.sched_gamma = sched_gamma
        self.loss_fn = torch.nn.BCELoss()
        self.criterion = roc_auc_score

    def forward(self, data):
        return self.model(data)

    def training_step(self, batch: Any, batch_idx: int):
        data, logits = batch
        batch_idx += 0
        preds = self(data).squeeze()
        targets = logits.to(torch.float32)
        loss = self.loss_fn(preds, targets)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        data, logits = batch
        batch_idx += 0
        preds = self(data).squeeze()
        targets = logits.to(torch.float32)
        loss = self.loss_fn(preds, targets)
        acc = ((preds >= 0.5) == logits).float().mean()
        roc = self.criterion(logits, preds)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_roc", roc, prog_bar=True, on_epoch=True)
        return {"val_loss": loss, "val_acc": acc, "val_roc": roc}

    def predict_step(self, batch: Any, batch_idx: int):
        preds = self(batch).squeeze().numpy()
        return preds

    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = StepLR(
            optimizer, step_size=self.sched_step_size, gamma=self.sched_gamma
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_roc",
                "frequency": 1,
                "name": "lr_scheduler",
            },
        }
