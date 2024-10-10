"""Data module for H2O dataset.

This module provides the H2ODataModule class, which is designed to handle
multiple video sequences from the H2O dataset for machine learning tasks.
It utilizes the VideoReader class to efficiently load and process video frames
across different scenes and camera views.

The H2O dataset consists of videos ranging from 257 to 1239 frames in length.

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
        num_clips (int): The total number of clips in the dataset.
        stride (int): The number of frames to skip between consecutive clips.
        start_video_idx (dict): Maps the first dataset index to the corresponding video reader.

    Args:
        dataset_prefix (str): The root directory path of the dataset.
        scenes (list[str]): List of scene names to include in the dataset.
        cameras (list[str]): List of camera names to include in the dataset.
        max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
        max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
        num_frames (int | None, optional): Number of frames to include per video. Defaults to None.
        stride (int | None, optional): Number of frames to skip between consecutive clips. Defaults to num_frames.

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
        stride: int | None = None,
    ) -> None:
        """Initialize the H2ODataset.

        Args:
            dataset_prefix (str): The root directory path of the dataset.
            scenes (list[str]): List of scene names to include in the dataset.
            cameras (list[str]): List of camera names to include in the dataset.
            max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
            max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
            num_frames (int | None, optional): Number of frames to include per video. Defaults to None.
            stride (int | None, optional): Number of frames to skip between consecutive clips. Defaults to num_frames.

        The dataset is constructed by creating VideoReader instances for each combination
        of scene and camera, using the provided dataset_prefix to construct the full path.

        """
        self.video_readers = []
        self.num_clips = 0
        self.num_frames = num_frames

        if stride is None:
            self.stride = num_frames
        else:
            self.stride = stride

        # Maps the first dataset index to the corresponding video
        self.start_video_idx = {}

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
                    self.start_video_idx[self.num_clips] = self.video_readers[-1]
                    num_frames = len(self.video_readers[-1])
                    self.num_clips += (num_frames - self.num_frames) // self.stride + 1

    def __len__(self) -> int:
        """Get the total number of clips in the dataset.

        Returns:
            int: The number of clips in the dataset.

        """
        return self.num_clips

    def _get_video_and_start_frame(self, idx: int) -> tuple[VideoReader, int]:
        """Get the video reader and start frame for a given dataset index.

        Args:
            idx (int): The dataset index.

        Returns:
            tuple[VideoReader, int]: A tuple containing the VideoReader instance
                                     and the start frame index of the clip.

        """
        # Find the corresponding video
        video_idx, video_reader = max((i, video) for i, video in self.start_video_idx.items() if i <= idx)

        # Calculate the start frame of the clip within the video
        clip_offset = idx - video_idx
        start_frame = clip_offset * self.stride

        return video_reader, start_frame

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Retrieve a video clip as a tensor of frames from the dataset.

        This method loads frames from a single video clip specified by the index.

        Args:
            idx (int): The index of the video clip to retrieve.

        Returns:
            torch.Tensor: A tensor containing frames of the video clip.
                          Shape: (T, C, H, W), where T is the number of frames,
                          C is the number of channels, H is the height, and W is the width.

        Raises:
            IndexError: If the provided index is out of range.

        """
        if idx >= len(self):
            msg = f"Index {idx} out of range. Total clips: {len(self)}"
            raise IndexError(msg)

        video_reader, start_frame = self._get_video_and_start_frame(idx)

        frames = video_reader.get_frames(list(range(start_frame, start_frame + self.num_frames)))

        # Normalize frames from uint8 to float32 with values between 0 and 1
        normalized_frames = [frame.astype(np.float32) / 255 for frame in frames]
        # Change shape from [H, W, C] to [C, H, W]
        normalized_frames = [np.transpose(frame, (2, 0, 1)) for frame in normalized_frames]

        # Convert list of numpy arrays to a PyTorch tensor
        return torch.from_numpy(np.stack(normalized_frames))


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
        num_workers: int = 8,
    ) -> None:
        """Initialize the H2ODataModule.

        Args:
            dataset_prefix (str): The prefix path to the dataset.
            cameras (list[str]): List of camera names to include in the dataset.
            max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
            max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
            batch_size (int, optional): The batch size for DataLoaders. Defaults to 32.
            num_frames (int, optional): Number of frames to include per video. Defaults to 300.
            num_workers (int, optional): Number of worker processes for data loading. Defaults to 8.

        """
        super().__init__()
        self.dataset_prefix = dataset_prefix
        self.cameras = cameras
        self.max_width = max_width
        self.max_height = max_height
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_workers = num_workers

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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
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
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
