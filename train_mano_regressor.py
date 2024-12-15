"""Module containing the main function for training."""

import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from data.h2o_datamodule import H2ODataModule
from data.transforms import VideoColorJitter, VideoMirror, VideoRandomRotation
from models.video_mano_regressor import VideoMANORegressor


@hydra.main(config_path="configs", config_name="train_mano_regressor.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Train model using PyTorch Lightning with Weights & Biases logging and Hydra configuration."""
    # Set up logging
    logging.basicConfig(level=cfg.log_level)
    logger = logging.getLogger(__name__)

    transforms = []
    if cfg.transforms is not None:
        if "VideoColorJitter" in cfg.transforms:
            transforms.append(VideoColorJitter())
        if "VideoRandomRotation" in cfg.transforms:
            transforms.append(VideoRandomRotation())
        if "VideoMirror" in cfg.transforms:
            transforms.append(VideoMirror())
    else:
        transforms = None

    # Create dataset and dataloaders
    datamodule = H2ODataModule(
        dataset_prefix=cfg.data.dataset_prefix,
        cameras=cfg.data.cameras,
        num_frames=cfg.data.num_frames,
        fps=cfg.data.framerate,
        batch_size=cfg.data.loader.batch_size,
        num_workers=cfg.data.loader.num_workers,
        crop_size=cfg.data.crop_size,
        padding_factor=cfg.data.padding_factor,
        transforms=transforms,
    )
    logger.info("DataModule initialized")

    height = cfg.data.crop_size
    width = cfg.data.crop_size

    # Initialize the model
    model = VideoMANORegressor(
        num_frames=cfg.data.num_frames,
        height=height,
        width=width,
        mano_root=os.environ.get("MANO_ROOT"),
        learning_rate=cfg.model.learning_rate,
        loss_weights=cfg.model.loss_weights,
        pretrained=cfg.data.pretrained,
        mano_params=cfg.model.mano_params,
        sixd=cfg.model.sixd,
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

    # Save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        filename="epoch{epoch:02d}-val{val_loss:.2f}",
        save_top_k=1,
        verbose=True,
    )

    # Initialize the trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=wandb_logger,
        log_every_n_steps=16,
        callbacks=[checkpoint_callback],
    )
    logger.info("Trainer initialized")

    # Train the model
    logger.info("Starting model training")
    trainer.fit(model, datamodule=datamodule)
    logger.info("Model training completed")


if __name__ == "__main__":
    main()
