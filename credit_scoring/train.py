import fire
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader
from typing import Callable
from tqdm import tqdm

from constants import *
from data_preprocessing import PathsDict, Preprocesser
import dataloader_utils
from dataset import CreditDataset
from model import CreditRNNModel

def save_model(model: CreditRNNModel,
               type: str) -> None:
    raw_datetime = str(np.datetime64('now'))
    date, _ = raw_datetime.split('T')
    trans_table = str.maketrans({'-': '', ':': ''})
    date = date.translate(trans_table)
    file_name = f'{date}_{type}_model_checkpoint.pt'
    torch.save(model, CHECKPOINTS_PATH / file_name)


def train_model(model: CreditRNNModel,
                train_loader: DataLoader,
                eval_loader: DataLoader,
                loss_criterion: Callable,
                opt: Optimizer,
                sched: LRScheduler = None,
                num_epochs: int = N_EPOCHS):
    """Training loop of one instance `CreditRNNModel`

    Args:
        model (credit_scoring.CreditRNNModel): Instance of one model
        train_loader (torch.utils.data.DataLoader): Dataloader with train data
        eval_loader (torch.utils.data.DataLoader): Dataloader with evaluation data
        loss_criterion (Callable[[float, float], float]): Loss function
        opt (Optimizer): Optimizer of loss function
        sched (LRScheduler, optional): LR Scheduler. Defaults to None.
        num_epochs (int, optional): Number of epochs to train. Defaults to 10.

    Returns:
        Tuple[Dict, Dict]: Checkpoint with best evaluation metrics and training history
    """
    max_roc = 0
    roc_list = []
    loss_list = []
    model = model.to(DEVICE)

    for epoch in tqdm(range(num_epochs)):
        epoch_losses = []
        model.train()
        for batch in train_loader:
            opt.zero_grad()
            X_batch = batch[0].to(DEVICE)
            y_batch = batch[1].to(DEVICE) # [batch_size]
            probs = model(X_batch).squeeze() # [batch_size]
            targets = y_batch.to(torch.float32)
            loss = loss_criterion(probs, targets)
            loss.backward()
            opt.step()
            epoch_losses.append(loss)
        sched.step()
        epoch_loss = np.array(epoch_losses).mean()
        loss_list.append(epoch_loss)
        roc_auc = evaluate(model, roc_auc_score, eval_loader)
        roc_list.append(roc_auc)
        if roc_auc > max_roc:
            max_roc = roc_auc
            best_checkpoint = {
                'epoch': epoch,
                'roc_auc': roc_auc,
                'loss': epoch_loss
            }
            save_model(best_checkpoint, 'best')
        history = {
            'losses': loss_list,
            'roc-auc': roc_list
        }
        print(f'Trained epoch {epoch} with loss {epoch_loss:.4f} and roc-auc {roc_auc:.4f}')

    return best_checkpoint, history

def evaluate(model: CreditRNNModel,
             criterion: Callable,
             dataloader: DataLoader) -> float:
    model.eval()
    roc_auc = []
    with torch.no_grad():
        for batch in dataloader:
            X_batch = batch[0].to(DEVICE)
            y_batch = batch[1].to('cpu')
            probs = model(X_batch).squeeze().to('cpu') # [batch_size]
            loss = criterion(y_batch, probs)
            roc_auc.append(loss.item())

    roc_auc = np.mean(np.array(roc_auc))
    return roc_auc

def main(data_root: Path | str = DATA_ROOT,
         train_folder: str = TRAIN_FILES_FOLDER,
         train_target: str = TRAIN_TARGET_FILE,
         test_folder: str = TEST_FILES_FOLDER,
         test_target: str = TEST_TARGET_FILE,
         file_submission: str = FILE_SUBMISSION,
         n_epochs: int = N_EPOCHS,
         agg_type: str = AGG_TYPE,
         rnn_type: str = RNN_TYPE):
    # Preprocessing
    paths_dict = PathsDict().make(
        data_root,
        train_folder,
        train_target,
        test_folder,
        test_target,
        file_submission
    )
    preproc = Preprocesser(paths_dict)

    all_features = preproc.combine_train_test_features('train')
    n_features = all_features[min(all_features.keys())].shape[1]
    tr_target = preproc.read_train_test_target('train_target')

    tr_features, val_features, tr_target, val_target = train_test_split(
        list(all_features.values()),
        tr_target['flag'].values,
        test_size=VAL_SIZE
    )

    del all_features

    # Dataset and dataloader create
    train_dataset = CreditDataset(tr_features, tr_target)
    val_dataset = CreditDataset(val_features, val_target)
    
    train_loader = DataLoader(
        train_dataset,
        collate_fn=dataloader_utils.collate_fn_tr,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        collate_fn=dataloader_utils.collate_fn_tr,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = CreditRNNModel(seq_len=n_features,
                           aggregation_type=agg_type,
                           rnn_type=rnn_type)
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    
    best_checkpoint, history = train_model(
        model, 
        train_loader, 
        val_loader,
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=n_epochs
    )

if __name__ == "__main__":
    fire.Fire(main)