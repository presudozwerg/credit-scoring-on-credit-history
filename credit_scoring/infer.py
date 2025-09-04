from pathlib import Path

import fire
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import (
    BATCH_SIZE,
    CHECKPOINTS_PATH,
    DATA_ROOT,
    DEVICE,
    ID_COLUMN_NAME,
    TEST_FILES_FOLDER,
)
from data_preprocessing import PathsDict, Preprocesser
from dataloader_utils import collate_fn_te
from dataset import CreditDataset
from model import CreditRNNModel


def find_new_checkpoint(location: Path):
    f_paths = list(location.iterdir())
    f_names = [path for path in f_paths if "best" in path.name]
    return sorted(f_names)[-1]


def prediction(model: CreditRNNModel, dataloader: DataLoader) -> npt.NDArray:
    """Inference function

    Args:
        model (CreditRNNModel): Model instance (should be trained).
        dataloader (DataLoader): Dataloader with test data

    Returns:
        npt.NDArray: Prediction probabilites of credit default
    """
    model.eval()
    preds = np.array([])

    with torch.no_grad():
        for batch in tqdm(dataloader):
            X_batch = batch.to(DEVICE)
            probs = model(X_batch).squeeze().to("cpu").numpy()
            preds = np.hstack((preds, probs))
    return preds


def infer_pipeline(
    data_root: Path | str = DATA_ROOT,
    test_folder: str = TEST_FILES_FOLDER,
    prediction_path: Path | str = DATA_ROOT,
):
    """Inderence pipeline

    Args:
        data_root (Path | str, optional): Path to root folder,
            where test data is placed. Defaults to DATA_ROOT.
        test_folder (str, optional): Directory with test data.
            Defaults to TEST_FILES_FOLDER.
        prediction_path (Path | str, optional): Directory where
            to place predictions. Defaults to DATA_ROOT.
    """
    if type(data_root) is str:
        data_root = Path(data_root)
    if type(prediction_path) is str:
        prediction_path = Path(prediction_path)

    paths_dict = PathsDict().make(data_root, test_folder)
    preproc = Preprocesser(paths_dict)
    te_features = preproc.read_train_test_features("test")
    test_ids = pd.DataFrame(te_features.keys(), columns=[ID_COLUMN_NAME])
    test_dataset = CreditDataset(list(te_features.values()), category="test")

    test_loader = DataLoader(
        test_dataset, collate_fn=collate_fn_te, batch_size=BATCH_SIZE, shuffle=False
    )
    model_instance = torch.load(find_new_checkpoint(CHECKPOINTS_PATH), weights_only=False)

    print("\nStarting inference of the model.\n")
    preds = prediction(model_instance, test_loader)
    preds = pd.DataFrame(preds, columns=["prediction"])

    prediction_result = pd.concat((test_ids, preds), axis=1)
    prediction_result.to_csv(prediction_path / "predict_probas.csv", index=False)
    print(
        f"\nPredicted values successfully saved at file: \n"
        f"{prediction_path / 'predict_probas.csv'}"
    )


if __name__ == "__main__":
    fire.Fire(infer_pipeline)
