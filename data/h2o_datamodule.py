"""Data module for H2O dataset.

This module provides the H2ODataModule class, which is designed to handle
multiple video files for machine learning tasks. It utilizes the VideoReader
class to efficiently load and process video frames.

H20 dataset has videos that range from 257 to 1239 frames

Example usage:
    datamodule = H2ODataModule(
        video_paths=['video1.mp4', 'video2.mp4'],
        frame_dir_paths=['frames1', 'frames2'],
        max_width=640,
        max_height=480,
        num_frames=100,
        batch_size=32,
        num_workers=4
    )
    trainer = pl.Trainer()
    trainer.fit(model, datamodule=datamodule)

"""

import logging
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from data.video_reader import VideoReader


class H2ODataset(Dataset):
    """A dataset class for handling H2O video data.

    This class is designed to work with the H2O dataset, which consists of multiple
    video sequences across different scenes and camera views. It utilizes the VideoReader
    class to efficiently load and process video frames.

    Attributes:
        video_readers (list): A list of VideoReader instances, one for each video in the dataset.
        num_frames (int | None): The number of frames to include per video. If None, all frames are included.

    Args:
        dataset_prefix (str): The root directory path of the dataset.
        scenes (list[str]): List of scene names to include in the dataset.
        cameras (list[str]): List of camera names to include in the dataset.
        max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
        max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
        num_frames (int | None, optional): Number of frames to include per video. Defaults to None.

    The dataset is constructed by creating VideoReader instances for each combination
    of scene and camera, using the provided dataset_prefix to construct the full path.

    """

    def __init__(
        self,
        dataset_prefix: str,
        scenes: list[str],
        cameras: list[str],
        max_width: int | None = None,
        max_height: int | None = None,
        num_frames: int | None = None,
    ) -> None:
        """Initialize the H2ODataset.

        Args:
            dataset_prefix (str): The root directory path of the dataset.
            scenes (list[str]): List of scene names to include in the dataset.
            cameras (list[str]): List of camera names to include in the dataset.
            max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
            max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
            num_frames (int | None, optional): Number of frames to include per video. Defaults to None.

        The dataset is constructed by creating VideoReader instances for each combination
        of scene and camera, using the provided dataset_prefix to construct the full path.

        """
        self.video_readers = []
        self.num_frames = num_frames

        scene_path_pattern = "{dataset_prefix}/{scene}"
        for scene in scenes:
            scene_path = scene_path_pattern.format(dataset_prefix=dataset_prefix, scene=scene)
            for directory in os.listdir(scene_path):
                for camera in cameras:
                    frame_dir_path = Path(scene_path) / directory / camera / "rgb"
                    self.video_readers.append(
                        VideoReader(
                            video_path=None,
                            frame_dir_path=frame_dir_path,
                            max_width=max_width,
                            max_height=max_height,
                            fmt_frame_fn=lambda x: f"{x:06d}.png",
                        ),
                    )

    def __len__(self) -> int:
        """Get the total number of videos in the dataset.

        Returns:
            int: The number of videos in the dataset.

        """
        return len(self.video_readers)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieve a video as a tensor of frames from the dataset.

        This method loads frames from a single video specified by the index.
        If the video is shorter than the desired number of frames, it extends
        the video with zero-filled frames.

        Args:
            idx (int): The index of the video to retrieve.

        Returns:
            torch.Tensor: A tensor containing frames of the video.
                          Shape: (T, C, H, W), where T is the number of frames,
                          C is the number of channels, H is the height, and W is the width.

        Raises:
            IndexError: If the provided index is out of range.

        """
        if idx >= len(self.video_readers):
            msg = f"Index {idx} out of range. Total videos: {len(self.video_readers)}"
            raise IndexError(msg)

        reader = self.video_readers[idx]

        if len(reader) < self.num_frames:
            # If video is shorter than the desired number of frames extend with zeros
            frames = reader.get_frames(list(range(len(reader))))
            frames.extend(np.zeros((self.num_frames - len(reader), *frames[0].shape)))
        else:
            frames = reader.get_frames(list(range(self.num_frames)))

        # Convert list of numpy arrays to a PyTorch tensor
        return torch.from_numpy(np.stack(frames))


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
        max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
        max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
        batch_size (int, optional): The batch size for DataLoaders. Defaults to 32.
        num_frames (int, optional): Number of frames to include per video. Defaults to 300.

    Example:
        datamodule = H2ODataModule(
            dataset_prefix='/path/to/dataset',
            cameras=['cam1', 'cam2'],
            max_width=640,
            max_height=480,
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
        max_width: int | None = None,
        max_height: int | None = None,
        batch_size: int = 32,
        num_frames: int = 300,
    ) -> None:
        """Initialize the H2ODataModule.

        Args:
            dataset_prefix (str): The prefix path to the dataset.
            cameras (list[str]): List of camera names to include in the dataset.
            max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
            max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
            batch_size (int, optional): The batch size for DataLoaders. Defaults to 32.
            num_frames (int, optional): Number of frames to include per video. Defaults to 300.

        """
        super().__init__()
        self.dataset_prefix = dataset_prefix
        self.cameras = cameras
        self.max_width = max_width
        self.max_height = max_height
        self.batch_size = batch_size
        self.num_frames = num_frames

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
        self.train_dataset = H2ODataset(
            dataset_prefix=self.dataset_prefix,
            scenes=self.train_scenes,
            cameras=self.cameras,
            max_width=self.max_width,
            max_height=self.max_height,
            num_frames=self.num_frames,
        )
        self.val_dataset = H2ODataset(
            dataset_prefix=self.dataset_prefix,
            scenes=self.val_scenes,
            cameras=self.cameras,
            max_width=self.max_width,
            max_height=self.max_height,
            num_frames=self.num_frames,
        )

        logging.info(f"Train dataset size: {len(self.train_dataset)}")
        logging.info(f"Validation dataset size: {len(self.val_dataset)}")

    def train_dataloader(self) -> DataLoader:
        """Create and return the DataLoader for the training dataset.

        Returns:
            DataLoader: A PyTorch DataLoader configured for the training dataset.

        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create and return the DataLoader for the validation dataset.

        Returns:
            DataLoader: A PyTorch DataLoader configured for the validation dataset.

        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)
