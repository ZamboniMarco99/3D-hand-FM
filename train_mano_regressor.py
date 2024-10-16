"""Module containing the main function for training."""

import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from data.h2o_datamodule import H2ODataModule
from models.video_mano_regressor import VideoMANORegressor


@hydra.main(config_path="configs", config_name="train_mano_regressor.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Train model using PyTorch Lightning with Weights & Biases logging and Hydra configuration."""
    # Set up logging
    logging.basicConfig(level=cfg.log_level)
    logger = logging.getLogger(__name__)

    # Create dataset and dataloaders
    datamodule = H2ODataModule(
        dataset_prefix=cfg.data.dataset_prefix,
        cameras=cfg.data.cameras,
        max_width=cfg.data.max_width,
        max_height=cfg.data.max_height,
        num_frames=cfg.data.num_frames,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        crop=cfg.data.pretrained,
    )
    logger.info("DataModule initialized")

    if cfg.data.resolution == "16x9":
        width = cfg.data.max_width
        height = int(cfg.data.max_width * 9 / 16)
    else:
        width = cfg.data.max_width
        height = cfg.data.max_height

    # Initialize the model
    model = VideoMANORegressor(
        num_frames=cfg.data.num_frames,
        height=height,
        width=width,
        mano_root=os.environ.get("MANO_ROOT"),
        learning_rate=cfg.model.learning_rate,
        pretrained=cfg.data.pretrained,
    )
    logger.info("Model initialized")

    # Setup Weights & Biases logger
    wandb_logger = WandbLogger(
        name=cfg.logger.name,
        save_dir=cfg.logger.save_dir,
        group=cfg.logger.group,
        log_model=cfg.logger.log_model,
    )
    logger.info("WandbLogger initialized")

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=wandb_logger,
        log_every_n_steps=16,
    )
    logger.info("Trainer initialized")

    # Train the model
    logger.info("Starting model training")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()
