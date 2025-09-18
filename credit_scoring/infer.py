from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig

from credit_scoring.pl_utils.data import CreditDataModule
from credit_scoring.pl_utils.model import CreditModel


# def choose_checkpoint(dir_path, typ):
#     chkpt_dir = Path(dir_path)
#     file_paths_list = sorted(list(chkpt_dir.glob("**/*.ckpt")))
#     if typ == "last":
#         return file_paths_list[-1]
#     else:
#         decimals = [str(path).split(".")[-2] for path in file_paths_list]
#         max_roc_idx = max(enumerate(decimals), key=lambda x: x[1])[0]
#         return file_paths_list[max_roc_idx]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def infer_pipeline(config: DictConfig):
    data_module = CreditDataModule(config)

    # if not checkpoint_path:
    #     chkp_dir = Path(CHECKPOINTS_DIR) / f"{model_type}-model"
    #     checkpoint_path = choose_checkpoint(chkp_dir, which_checkpoint)

    model = CreditModel.load_from_checkpoint(config.infer.checkpoint_path)

    trainer = pl.Trainer(
        accelerator=config.train.device.accelerator,
        devices=config.infer.predict_devices,
    )

    preds = trainer.predict(model, datamodule=data_module)
    preds = pd.DataFrame(np.hstack(preds), columns=[config.infer.prediction_column_name])
    ids = pd.DataFrame(data_module.pred_ids, columns=[config.infer.id_column_name])

    prediction_result = pd.concat((ids, preds), axis=1)
    prediction_result.to_csv(config.infer.prediction_path, index=False)
    print(
        f"\nPredicted values successfully saved at file: \n"
        f"{Path(config.infer.prediction_path).resolve()}"
    )


if __name__ == "__main__":
    infer_pipeline()
