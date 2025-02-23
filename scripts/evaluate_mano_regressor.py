"""Script to evaluate the VideoManoRegressor model on the H2O validation set."""

import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from data.h2o import H2ODataModule
from models.video_mano_regressor import VideoMANORegressor


@hydra.main(config_path="configs", config_name="train_mano_regressor.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """Evaluate the VideoManoRegressor model on the validation set."""
    # Set up logging
    logging.basicConfig(level=cfg.log_level)
    logger = logging.getLogger(__name__)

    torch.set_float32_matmul_precision("high")

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
        transforms=None,  # No augmentations during evaluation
        dataset_type="last_frame" if cfg.model.last_frame_only else "sequence",
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
        mean_mano_params_location=cfg.model.mean_mano_params_location,
        last_frame_only=cfg.model.last_frame_only,
    )
    logger.info("Model initialized")

    # Load the checkpoint
    if not hasattr(cfg, "checkpoint_path"):
        msg = "Please specify the checkpoint_path in the config file"
        raise ValueError(msg)

    checkpoint_path = cfg.checkpoint_path
    if not Path(checkpoint_path).exists():
        msg = f"Checkpoint not found at {checkpoint_path}"
        raise FileNotFoundError(msg)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    logger.info(f"Loaded checkpoint from {checkpoint_path}")

    # Initialize the trainer for evaluation
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,  # Use single device for evaluation
    )
    logger.info("Trainer initialized")

    # Run evaluation
    logger.info("Starting evaluation...")
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
