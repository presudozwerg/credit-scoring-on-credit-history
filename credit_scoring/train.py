import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig

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

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=config.logger.experiment,
            run_name=f"{config.model.rnn_type}-model",
            save_dir=config.logger.save_dir,
            tracking_uri=config.logger.ip,
        ),
    ]

    callbacks = [
        pl.callbacks.LearningRateMonitor(**config.logger.callbacks.lr_monitor),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(**config.logger.callbacks.rich_model_sum),
    ]

    callbacks.append(pl.callbacks.ModelCheckpoint(**config.train.checkpoints))

    trainer = pl.Trainer(
        **config.train.device,
        **config.train.process,
        max_epochs=config.train.max_epochs,
        logger=loggers,
        callbacks=callbacks,
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    training_pipeline()
