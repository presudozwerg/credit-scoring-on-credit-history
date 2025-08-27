from data_preprocessing import PathsDict, Preprocesser
from dataset import CreditDataset
from model import CreditRNNModel
from dataloader_utils import collate_fn_te
from constants import *

import fire
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

def find_new_checkpoint(location: Path):
    return sorted(list(location.iterdir()))[-1]

def prediction(model: CreditRNNModel, 
               dataloader: DataLoader) -> np.array:
    model.eval()
    preds = np.array([])
    with torch.no_grad():
        for batch in tqdm(dataloader):
            X_batch = batch.to(DEVICE)
            probs = model(X_batch).squeeze().to('cpu').numpy()
            preds = np.hstack((preds, probs))
    return preds

def main(data_root: Path | str = DATA_ROOT,
         test_folder: str = TEST_FILES_FOLDER,
         prediction_path: Path | str = DATA_ROOT):
    
    paths_dict = PathsDict().make(
        data_root,
        test_folder
    )
    preproc = Preprocesser(paths_dict)
    te_features = preproc.combine_train_test_features('test')
    test_ids = pd.DataFrame(te_features.keys(), columns=[ID_COLUMN_NAME])
    test_dataset = CreditDataset(list(te_features.values()), category='test')

    test_loader = DataLoader(
        test_dataset,
        collate_fn=collate_fn_te,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    model_instance = torch.load(
        find_new_checkpoint(CHECKPOINTS_PATH),
        weights_only=False
    )

    preds = prediction(model_instance, test_loader)
    preds = pd.DataFrame(preds, columns=['prediction'])

    
    prediction_result = pd.concat((test_ids, preds), axis=1)
    prediction_result.to_csv(prediction_path / 'predict_probas.csv', index=False)
    print(f"Predicted values successfully saved at file: {prediction_path / 'predict_probas.csv'}")

if __name__ == '__main__':
    fire.Fire(main)

