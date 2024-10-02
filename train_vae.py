"""Module containing the main function for training."""

import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from data.h2o_datamodule import H2ODataModule
from models.vae.vae import VideoVAE


@hydra.main(config_path="configs", config_name="train_vae.yaml", version_base="1.1")
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
    )
    logger.info("DataModule initialized")

    if cfg.data.resolution == "16x9":
        width = cfg.data.max_width
        height = int(cfg.data.max_width * 9 / 16)
    else:
        width = cfg.data.max_width
        height = cfg.data.max_height

    # Initialize the model
    model = VideoVAE(
        num_frames=cfg.data.num_frames,
        height=height,
        width=width,
        learning_rate=cfg.model.learning_rate,
        kld_weight=cfg.model.kld_weight,
        mse_weight=cfg.model.mse_weight,
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
        log_every_n_steps=1,
    )
    logger.info("Trainer initialized")

    # Train the model
    logger.info("Starting model training")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()
