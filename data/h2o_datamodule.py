"""Data module for H2O dataset.

This module provides the H2ODataModule class, which is designed to handle
multiple video sequences and corresponding MANO hand pose parameters from the H2O dataset
for machine learning tasks. It utilizes the VideoReader class to efficiently load and
process video frames, and the ManoReader class to load MANO parameters, across different
scenes and camera views.

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
from functools import cache
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F  # noqa: N812

from data.mano_reader import ManoReader
from data.video_reader import VideoReader


class H2ODataset(Dataset):
    """A dataset class for handling H2O video data and MANO parameters.

    This class is designed to work with the H2O dataset, which consists of multiple
    video sequences and corresponding MANO hand pose parameters across different
    scenes and camera views. It utilizes the VideoReader class to efficiently load
    and process video frames, and the ManoReader class to load MANO parameters.

    Attributes:
        video_readers (list): A list of VideoReader instances, one for each video in the dataset.
        mano_readers (list): A list of ManoReader instances, one for each MANO sequence in the dataset.
        num_frames (int | None): The number of frames to include per video clip. If None, all frames are included.
        num_clips (int): The total number of clips in the dataset.
        clip_to_data (dict): Maps clip indices to corresponding video reader, MANO reader, and start frame.

    Args:
        dataset_prefix (str): The root directory path of the dataset.
        scenes (list[str]): List of scene names to include in the dataset.
        cameras (list[str]): List of camera names to include in the dataset.
        max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
        max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
        num_frames (int | None, optional): Number of frames to include per video clip. Defaults to None.

    The dataset is constructed by creating VideoReader and ManoReader instances for each
    combination of scene and camera, using the provided dataset_prefix to construct the full path.
    It handles both video frame data and corresponding MANO hand pose parameters.

    """

    def __init__(
        self,
        dataset_prefix: str,
        scenes: list[str],
        cameras: list[str],
        max_width: int | None = None,
        max_height: int | None = None,
        num_frames: int | None = None,
        crop: bool = False,
        cache: bool = True,
        transforms: list[nn.Module] | None = None,
    ) -> None:
        """Initialize the H2ODataset.

        Args:
            dataset_prefix (str): The root directory path of the dataset.
            scenes (list[str]): List of scene names to include in the dataset.
            cameras (list[str]): List of camera names to include in the dataset.
            max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
            max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
            num_frames (int | None, optional): Number of frames to include per video. Defaults to None.
            crop (bool, optional): If True, crop videos to exact max_width and max_height sizes. Defaults to False.
            cache (bool, optional): If True, enable caching of video frames. Defaults to True.
            transforms (list[nn.Module] | None, optional): List of video transform modules to apply. Each transform
                should take (video, mano_left, mano_right, intrinsic_matrix) as input and return the same tuple
                with transformed tensors. Defaults to None.

        The dataset is constructed by creating VideoReader and ManoReader instances for each combination
        of scene and camera, using the provided dataset_prefix to construct the full path.
        If crop is True, videos will be cropped to the exact max_width and max_height sizes.
        If cache is True, data will be cached in memory for faster access.
        If transforms is provided, the transforms will be applied sequentially to the video and parameters.

        """
        self.video_readers = []
        self.mano_readers = []
        self.camera_intrinsics = []
        self.num_clips = 0
        self.num_frames = num_frames
        self.cache = cache
        self.transforms = transforms
        # Maps the first dataset index to the corresponding video reader and MANO reader
        self.clip_to_data = {}

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
                            fmt_frame_fn=lambda x: f"{x:06d}.png",
                        ),
                    )

                    intrinsics_path = Path(scene_path) / directory / camera / "cam_intrinsics.txt"
                    [fx, fy, cx, cy, w, h] = np.loadtxt(intrinsics_path)
                    # Calculate the scale factor while maintaining aspect ratio
                    scale = max(max_width / w, max_height / h) if max_width and max_height else 1
                    new_w, new_h = int(w * scale), int(h * scale)

                    # Calculate cropping for centering
                    crop_x = (new_w - max_width) // 2
                    crop_y = (new_h - max_height) // 2

                    intrinsics = np.array(
                        [
                            [fx * scale, 0, cx * scale - crop_x],
                            [0, fy * scale, cy * scale - crop_y],
                            [0, 0, 1],
                        ],
                    )
                    self.camera_intrinsics.append(intrinsics)

                    mano_dir_path = Path(scene_path) / directory / camera / "hand_pose_mano"
                    self.mano_readers.append(
                        ManoReader(
                            mano_dir_path=mano_dir_path,
                            assumed_fps=30,
                            fmt_frame_fn=lambda x: f"{x:06d}.txt",
                        ),
                    )

                    self.clip_to_data[self.num_clips] = (
                        self.video_readers[-1],
                        self.mano_readers[-1],
                        self.camera_intrinsics[-1],
                    )
                    self.num_clips += len(self.video_readers[-1]) // self.num_frames

    def __len__(self) -> int:
        """Get the total number of clips in the dataset.

        Returns:
            int: The number of clips in the dataset.

        """
        return self.num_clips

    def _get_clip_data(self, clip_idx: int) -> tuple[VideoReader, ManoReader, np.ndarray, int]:
        """Get the video reader, MANO reader, camera intrinsics and start frame for a given clip index.

        Args:
            clip_idx (int): The index of the clip in the dataset.

        Returns:
            tuple[VideoReader, ManoReader, np.ndarray, int]: A tuple containing:
                - The VideoReader instance for the clip.
                - The ManoReader instance for the clip.
                - The camera intrinsic matrix with shape (3, 3).
                - The start frame index of the clip within its video.

        """
        # Find the corresponding video and MANO readers
        video_idx, video_reader, mano_reader, intrinsics = max(
            (i, video, mano, intrinsics) for i, (video, mano, intrinsics) in self.clip_to_data.items() if i <= clip_idx
        )

        # Calculate the start frame of the clip within the video
        clip_idx_in_video = clip_idx - video_idx
        start_frame = self.num_frames * clip_idx_in_video

        return video_reader, mano_reader, intrinsics, start_frame

    @staticmethod
    @cache
    def _get_video_frames(video_reader: VideoReader, start_frame: int, num_frames: int) -> list[np.ndarray]:
        """Get video frames from the video reader for a given clip.

        Args:
            video_reader (VideoReader): The video reader instance.
            start_frame (int): The start frame index.
            num_frames (int): The number of frames to retrieve.

        Returns:
            list[np.ndarray]: A list of video frames with shape (H, W, C).

        """
        frames = video_reader.get_frames(list(range(start_frame, start_frame + num_frames)))

        # Normalize frames from uint8 to float32 with values between 0 and 1
        normalized_frames = [frame.astype(np.float32) / 255 for frame in frames]
        # Change shape from [H, W, C] to [C, H, W]
        return [np.transpose(frame, (2, 0, 1)) for frame in normalized_frames]

    @staticmethod
    @cache
    def _get_mano_params(
        mano_reader: ManoReader,
        start_frame: int,
        num_frames: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get MANO parameters from the MANO reader for a given clip.

        Args:
            mano_reader (ManoReader): The MANO reader instance.
            start_frame (int): The start frame index.
            num_frames (int): The number of frames to retrieve.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists of numpy arrays:
                - The first list contains MANO parameters for the left hand for each frame.
                - The second list contains MANO parameters for the right hand for each frame.

        """
        return mano_reader.get_mano_sequence(list(range(start_frame, start_frame + num_frames)))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a video clip and corresponding MANO parameters and camera intrinsics from the dataset.

        This method loads frames, MANO parameters for both hands, and camera intrinsics from a single video clip
        specified by the index.

        Args:
            idx (int): The index of the video clip to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of video frames with shape (T, C, H, W), where T is the number of frames,
                  C is the number of channels, H is the height, and W is the width. Values are normalized
                  with mean (0.45, 0.45, 0.45) and std (0.225, 0.225, 0.225).
                - A tensor of left hand MANO parameters with shape (T, 61), where T is the number of frames.
                  Contains translation (3), pose (45) and shape (10) parameters.
                - A tensor of right hand MANO parameters with shape (T, 61), where T is the number of frames.
                  Contains translation (3), pose (45) and shape (10) parameters.
                - A tensor of camera intrinsic parameters with shape (3, 3).

        Raises:
            IndexError: If the provided index is out of range.

        """
        if idx >= len(self):
            msg = f"Index {idx} out of range. Total clips: {len(self)}"
            raise IndexError(msg)

        video_reader, mano_reader, intrinsics, start_frame = self._get_clip_data(idx)

        if self.cache:
            frames = self._get_video_frames(video_reader, start_frame, self.num_frames)
            mano_params_left, mano_params_right = self._get_mano_params(mano_reader, start_frame, self.num_frames)
        else:
            frames = self._get_video_frames.__wrapped__(video_reader, start_frame, self.num_frames)
            mano_params_left, mano_params_right = self._get_mano_params.__wrapped__(
                mano_reader,
                start_frame,
                self.num_frames,
            )

        # Convert list of numpy arrays to PyTorch tensors
        clip = F.normalize(torch.from_numpy(np.stack(frames)), mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225))
        mano_left = torch.from_numpy(np.stack(mano_params_left))
        mano_right = torch.from_numpy(np.stack(mano_params_right))

        intrinsics = torch.from_numpy(intrinsics).to(torch.float32)

        if self.transforms is not None:
            for transform in self.transforms:
                clip, mano_left, mano_right, intrinsics = transform(clip, mano_left, mano_right, intrinsics)

        return clip, mano_left, mano_right, intrinsics


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
        crop: bool = False,
        transforms: list[nn.Module] | None = None,
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
            crop (bool, optional): Whether to crop the frames to exact max_width and max_height. Defaults to False.
            transforms (list[nn.Module] | None, optional): List of video transform modules to apply to training data.
                Each transform should take (video, mano_left, mano_right, intrinsic_matrix) as input and return the same
                tuple with transformed tensors. Defaults to None.

        """
        super().__init__()
        self.dataset_prefix = dataset_prefix
        self.cameras = cameras
        self.max_width = max_width
        self.max_height = max_height
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.num_workers = num_workers
        self.crop = crop
        self.transforms = transforms

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
            crop=self.crop,
            cache=False,
            transforms=self.transforms,
        )
        self.val_dataset = H2ODataset(
            dataset_prefix=self.dataset_prefix,
            scenes=self.val_scenes,
            cameras=self.cameras,
            max_width=self.max_width,
            max_height=self.max_height,
            num_frames=self.num_frames,
            crop=self.crop,
            cache=False,
            transforms=None,
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
