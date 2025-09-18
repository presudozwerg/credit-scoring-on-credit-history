import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

import credit_scoring.dvc_utils.dvc_load as dvc_load 
from credit_scoring.pl_utils.experiment import create_new_run
from credit_scoring.pl_utils.credit_rnn_model import CreditRNNModel
from credit_scoring.pl_utils.data import CreditDataModule
from credit_scoring.pl_utils.model import CreditModel


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def training_pipeline(config: DictConfig):
    pl.seed_everything(config.seed)

    data_module = CreditDataModule(config)
    rnn_model = CreditRNNModel(config.model)

    model = CreditModel(
        rnn_model, config.train.lr, config.train.sched_step, config.train.sched_gamma
    )

    if dvc_load.only_dvc_in_dir(config.data_load.checkpoint.dvc_data):
        dvc_load.dvc_prepare_pipeline(**config.data_load.checkpoint)

    run_title = create_new_run(
        config.train.chkp_root, 
        f"{config.model.rnn_type}-model"
    )

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=config.logger.experiment,
            run_name=run_title,
            save_dir=config.logger.save_dir,
            tracking_uri=config.logger.ip,
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(**config.logger.callbacks.lr_monitor),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(**config.logger.callbacks.rich_model_sum),
    ]

    checkpoints_path = f"{config.train.chkp_root}/{run_title}"

    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            **config.train.checkpoints,
            dirpath=checkpoints_path)
    )
    trainer = pl.Trainer(
        **config.train.device,
        **config.train.process,
        max_epochs=config.train.max_epochs,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)

    dvc_load.upload_dvc_data(config.train.chkp_root, run_title, 'model')

if __name__ == "__main__":
    training_pipeline()
