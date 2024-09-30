"""Module containing the main function for training."""

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data.dummy_dataset import DummyDataset
from models.vae.vae import VideoVAE


@hydra.main(config_path="configs", config_name="train_vae.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Train model using PyTorch Lightning with Weights & Biases logging and Hydra configuration."""
    # Create dataset and dataloaders
    train_dataset = DummyDataset(
        time=cfg.data.time,
        height=cfg.data.height,
        width=cfg.data.width,
        frame_rate=cfg.data.frame_rate,
    )
    val_dataset = DummyDataset(
        time=cfg.data.time,
        height=cfg.data.height,
        width=cfg.data.width,
        frame_rate=cfg.data.frame_rate,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.train_loader.batch_size,
        shuffle=True,
        num_workers=cfg.data.train_loader.num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.val_loader.batch_size,
        shuffle=False,
        num_workers=cfg.data.val_loader.num_workers,
        persistent_workers=True,
    )

    # Initialize the model
    model = VideoVAE(
        num_frames=cfg.data.time * cfg.data.frame_rate,
        height=cfg.data.height,
        width=cfg.data.width,
        learning_rate=cfg.model.learning_rate,
    )

    # Setup Weights & Biases logger
    logger = WandbLogger(
        name=cfg.logger.name,
        save_dir=cfg.logger.save_dir,
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
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
