import fire
import inspect
import numpy as np
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
from typing import Dict

def save_model(model: CreditRNNModel,
               type: str,
               metrics: Dict = None) -> None:
    raw_datetime = str(np.datetime64('now'))
    date, _ = raw_datetime.split('T')
    trans_table = str.maketrans({'-': '', ':': ''})
    date = date.translate(trans_table)

    # Saving checkpoint
    file_name = f'{date}_{type}_model_checkpoint.pt'
    model_path = CHECKPOINTS_PATH / file_name
    with open(model_path, 'w') as handle:
        torch.save(model, CHECKPOINTS_PATH / file_name)

    # Saving metrics history
    if metrics:
        file_name = f'{date}_{type}_model_history.pt'
        model_path = model_path = CHECKPOINTS_PATH / file_name
        with open(model_path, 'w') as handle:
            torch.save(metrics, CHECKPOINTS_PATH / file_name)
        if type == 'last':
            msg = (f"\nModel and history of training successfully "
                   f"saved at: \n{CHECKPOINTS_PATH}")
            print(msg)


def train_model(model: CreditRNNModel,
                train_loader: DataLoader,
                eval_loader: DataLoader,
                loss_criterion: Callable,
                opt: Optimizer,
                sched: LRScheduler = None,
                num_epochs: int = 10):
    """Training loop of one instance `CreditRNNModel`

    Args:
        model (credit_scoring.CreditRNNModel): Instance of 
            one model.
        train_loader (torch.utils.data.DataLoader): Dataloader 
            with train data.
        eval_loader (torch.utils.data.DataLoader): Dataloader 
            with evaluation data.
        loss_criterion (Callable[[float, float], float]): 
            Loss function
        opt (Optimizer): Optimizer of loss function
        sched (LRScheduler, optional): LR Scheduler. 
            Defaults to None.
        num_epochs (int, optional): Number of epochs to train. 
            Defaults to 10.

    Returns:
        Tuple[Dict, Dict]: Checkpoint with best evaluation 
            metrics and training history.
    """
    max_roc = 0
    roc_list = []
    loss_list = []
    model = model.to(DEVICE)

    msg = (f"Starting training of the model on "
           f"{num_epochs} epochs.\n")
    print(msg)

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
            epoch_losses.append(loss.item())
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
            save_model(model, 'best')
        history = {
            'losses': loss_list,
            'roc-auc': roc_list
        }
        msg = (f"Trained epoch {epoch} with loss "
               f"{epoch_loss:.4f} and roc-auc {roc_auc:.4f}")
        print(msg)
    save_model(model, 'last', history)
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
            roc_auc.append(loss)

    roc_auc = np.mean(np.array(roc_auc))
    return roc_auc

def train_pipeline(data_root: Path | str = DATA_ROOT,
                   train_folder: str = TRAIN_FILES_FOLDER,
                   train_target: str = TRAIN_TARGET_FILE,
                   test_folder: str = TEST_FILES_FOLDER,
                   n_epochs: int = 10,
                   agg_type: str = 'last',
                   rnn_type: str = 'rnn'):
    """Training pipeline

    Args:
        data_root (Path | str, optional): Path to root folder, 
            where all the data is placed. Defaults to DATA_ROOT.
        train_folder (str, optional): Name of the folder in 
            `data_root_path`, with training data. Defaults to 
            TRAIN_FILES_FOLDER.
        train_target (str, optional): Name of the file in 
            `data_root_path` folder, with train target values. 
            Should have `.csv` extension. Defaults to TRAIN_TARGET_FILE.
        test_folder (str, optional): Name of the folder in 
            `data_root_path`, with test data. Defaults to 
            TEST_FILES_FOLDER.
        n_epochs (int, optional): Number of training epochs. 
            Defaults to 10.
        agg_type (str, optional): RNN aggergation type. Can be 
            'max', 'mean' or 'last'. Defaults to 'last'.
        rnn_type (str, optional): Type of RNN block. Can be 'rnn', 
            'gru' or 'lstm'. Defaults to 'rnn'.
    """
    if type(data_root) is str:
        data_root = Path(data_root)

    # Preprocessing
    paths_dict = PathsDict().make(
        data_root,
        train_folder,
        train_target,
        test_folder
    )
    preproc = Preprocesser(paths_dict)

    all_features = preproc.read_train_test_features('train')
    n_features = all_features[min(all_features.keys())].shape[1]
    tr_target = preproc.read_train_target()

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
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)
    
    history = train_model(
        model, 
        train_loader, 
        val_loader,
        criterion, 
        optimizer, 
        scheduler, 
        num_epochs=n_epochs
    )
    del history

if __name__ == "__main__":
    fire.Fire(train_pipeline)