"""PyTorch Lightning DataModule for the H2O dataset.

This module provides the H2ODataModule class, which handles data loading and preparation
for the H2O dataset, including setting up train and validation datasets.
Training and validation scenes are defined as the original repo:
https://github.com/taeinkwon/h2odataset.

Example usage:
    datamodule = H2ODataModule(
        dataset_prefix='/path/to/dataset',
        cameras=['cam1', 'cam2'],
        batch_size=16,
        num_frames=200
    )
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=datamodule)
"""

import logging
from typing import Literal

import pytorch_lightning as pl
from torch import nn
from torch.utils.data import DataLoader

from data.h2o.dataset import H2ODataset
from data.h2o.last_frame_dataset import H2OLastFrameDataset


class H2ODataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the H2O dataset.

    This class handles the data loading and preparation for the H2O dataset,
    including setting up train and validation datasets.
    Training and validation scenes are defined as the original repo:
    https://github.com/taeinkwon/h2odataset.

    Attributes:
        train_scenes (tuple): A tuple of scene names used for training.
        val_scenes (tuple): A tuple of scene names used for validation.

    Args:
        dataset_prefix (str): The prefix path to the dataset.
        cameras (list[str]): List of camera names to include in the dataset.
        batch_size (int, optional): The batch size for DataLoaders. Defaults to 32.
        num_frames (int, optional): Number of frames to include per video. Defaults to 300.
        dataset_type (str, optional): Type of dataset to use. Either 'sequence' or 'last_frame'.
            Defaults to 'sequence'.

    Example:
        datamodule = H2ODataModule(
            dataset_prefix='/path/to/dataset',
            cameras=['cam1', 'cam2'],
            batch_size=16,
            num_frames=200
        )
        trainer = pl.Trainer()
        trainer.fit(model, datamodule=datamodule)

    """

    train_scenes = (
        "subject1/h1",
        "subject1/h2",
        "subject1/k1",
        "subject1/k2",
        "subject1/o1",
        "subject1/o2",
        "subject2/h1",
        "subject2/h2",
        "subject2/k1",
        "subject2/k2",
        "subject2/o1",
        "subject2/o2",
        "subject3/h1",
        "subject3/h2",
        "subject3/k1",
    )

    val_scenes = ("subject3/k2", "subject3/o1", "subject3/o2")

    def __init__(
        self,
        dataset_prefix: str,
        cameras: list[str],
        batch_size: int = 32,
        num_frames: int = 300,
        fps: float = 7.5,
        num_workers: int = 8,
        transforms: list[nn.Module] | None = None,
        crop_size: int = 224,
        padding_factor: float = 1.2,
        dataset_type: Literal["sequence", "last_frame"] = "sequence",
    ) -> None:
        """Initialize the H2ODataModule.

        Args:
            dataset_prefix (str): The prefix path to the dataset.
            cameras (list[str]): List of camera names to include in the dataset.
            batch_size (int, optional): The batch size for DataLoaders. Defaults to 32.
            num_frames (int, optional): Number of frames to include per video. Defaults to 300.
            fps (float, optional): Desired frames per second. Defaults to 7.5.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 8.
            transforms (list[nn.Module] | None, optional): List of video transform modules to apply to training data.
                Each transform should take (video, mano_left, mano_right, intrinsic_matrix) as input and return the same
                tuple with transformed tensors. Defaults to None.
            crop_size (int, optional): Size of the output square crop in pixels. Defaults to 224.
            padding_factor (float, optional): Factor to increase the crop size by. Defaults to 1.2.
            dataset_type (str, optional): Type of dataset to use. Either 'sequence' or 'last_frame'.
                Defaults to 'sequence'.

        """
        super().__init__()
        self.dataset_prefix = dataset_prefix
        self.cameras = cameras
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.fps = fps
        self.num_workers = num_workers
        self.transforms = transforms
        self.crop_size = crop_size
        self.padding_factor = padding_factor
        self.dataset_type = dataset_type

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        """Set up the train and validation datasets.

        This method initializes the train and validation datasets using the H2ODataset class.
        It is called automatically by PyTorch Lightning.

        Args:
            stage (str | None, optional): The stage of training ('fit', 'validate', 'test', or 'predict').
                                          This parameter is not used in the current implementation.

        The method creates two dataset attributes:
        - self.train_dataset: An H2ODataset instance for training data.
        - self.val_dataset: An H2ODataset instance for validation data.

        """
        dataset_class = H2OLastFrameDataset if self.dataset_type == "last_frame" else H2ODataset

        self.train_dataset = dataset_class(
            dataset_prefix=self.dataset_prefix,
            scenes=self.train_scenes,
            cameras=self.cameras,
            num_frames=self.num_frames,
            fps=self.fps,
            cache=False,
            transforms=self.transforms,
            crop_size=self.crop_size,
            padding_factor=self.padding_factor,
        )
        self.val_dataset = dataset_class(
            dataset_prefix=self.dataset_prefix,
            scenes=self.val_scenes,
            cameras=self.cameras,
            num_frames=self.num_frames,
            fps=self.fps,
            cache=False,
            transforms=None,
            crop_size=self.crop_size,
            padding_factor=self.padding_factor,
        )

        logging.info(f"Train dataset size: {len(self.train_dataset)}")
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Create and return the DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader configured for the training dataset.

        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=4,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader configured for the validation dataset.

        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=4,
        )
