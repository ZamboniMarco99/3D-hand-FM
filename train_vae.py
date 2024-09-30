"""Module containing the main function for training."""

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from data.h2o_datamodule import H2ODataModule
from models.vae.vae import VideoVAE


@hydra.main(config_path="configs", config_name="train_vae.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Train model using PyTorch Lightning with Weights & Biases logging and Hydra configuration."""
    # Create dataset and dataloaders
    datamodule = H2ODataModule(
        dataset_prefix=cfg.data.dataset_prefix,
        cameras=cfg.data.cameras,
        max_width=cfg.data.max_width,
        max_height=cfg.data.max_height,
        num_frames=cfg.data.num_frames,
        batch_size=cfg.data.loader.batch_size,
    )

    # Initialize the model
    model = VideoVAE(
        num_frames=cfg.data.num_frames,
        height=cfg.data.max_height,
        width=cfg.data.max_width,
        learning_rate=cfg.model.learning_rate,
        kld_weight=cfg.model.kld_weight,
        mse_weight=cfg.model.mse_weight,
    )

    # Setup Weights & Biases logger
    logger = WandbLogger(
        name=cfg.logger.name,
        save_dir=cfg.logger.save_dir,
        group=cfg.logger.group,
        log_model=cfg.logger.log_model,
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        log_every_n_steps=1,
    )

    # Train the model
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
