"""Dataset class for H2O dataset.

This module provides the H2ODataset class, which is designed to handle
multiple video sequences and corresponding MANO hand pose parameters from the H2O dataset
for machine learning tasks. It utilizes the VideoReader class to efficiently load and
process video frames, and the ManoReader class to load MANO parameters, across different
scenes and camera views.

The H2O dataset consists of videos ranging from 257 to 1239 frames in length.

Example usage:
    dataset = H2ODataset(
        dataset_prefix='/path/to/dataset',
        scenes=['scene1', 'scene2'],
        cameras=['cam1', 'cam2'],
        num_frames=100,
        fps=7.5
    )
"""

import os
from functools import cache
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.transforms import functional as F  # noqa: N812

from data.bbox_reader import BboxReader
from data.joints_reader import JointsReader
from data.mano_reader import ManoReader
from data.transforms import CropHand, VideoMirror
from data.video_reader import VideoReader
from models.utils import project_joints_to_2d


class H2ODataset(torch.utils.data.Dataset):
    """A dataset class for handling H2O video data and MANO parameters.

    This class is designed to work with the H2O dataset, which consists of multiple
    video sequences and corresponding MANO hand pose parameters across different
    scenes and camera views. It utilizes the VideoReader class to efficiently load
    and process video frames, and the ManoReader class to load MANO parameters.

    Attributes:
        video_readers (list): A list of VideoReader instances, one for each video in the dataset.
        mano_readers (list): A list of ManoReader instances, one for each MANO sequence in the dataset.
        bbox_readers (list): A list of BboxReader instances, one for each Bbox sequence in the dataset.
        camera_intrinsics (list): A list of camera intrinsic matrices, one for each camera in the dataset.
        num_frames (int | None): The number of frames to include per video clip. If None, all frames are included.
        num_clips (int): The total number of clips in the dataset.
        clip_to_data (dict): Maps clip indices to corresponding video reader, MANO reader, bbox reader, and start frame.

    Args:
        dataset_prefix (str): The root directory path of the dataset.
        scenes (list[str]): List of scene names to include in the dataset.
        cameras (list[str]): List of camera names to include in the dataset.
        num_frames (int | None, optional): Number of frames to include per video clip. Defaults to None.

    """

    def __init__(
        self,
        dataset_prefix: str,
        scenes: list[str],
        cameras: list[str],
        num_frames: int | None = None,
        fps: float = 7.5,
        cache: bool = True,
        transforms: list[nn.Module] | None = None,
        crop_size: int = 224,
        padding_factor: float = 1.2,
    ) -> None:
        """Initialize the H2ODataset.

        Args:
            dataset_prefix (str): The root directory path of the dataset.
            scenes (list[str]): List of scene names to include in the dataset.
            cameras (list[str]): List of camera names to include in the dataset.
            num_frames (int | None, optional): Number of frames to include per video. Defaults to None.
            fps (float, optional): Desired frames per second. Defaults to 7.5.
            cache (bool, optional): If True, enable caching of video frames. Defaults to True.
            transforms (list[nn.Module] | None, optional): List of video transform modules to apply. Each transform
                should take (video, mano_left, mano_right, intrinsic_matrix) as input and return the same tuple
                with transformed tensors. Defaults to None.
            crop_size (int, optional): Size of the output square crop in pixels. Defaults to 224.
            padding_factor (float, optional): Factor to increase the crop size by. Defaults to 1.2.

        """
        self.video_readers = []
        self.mano_readers = []
        self.bbox_readers = []
        self.joints_readers = []
        self.camera_intrinsics = []
        self.num_clips = 0
        self.num_frames = num_frames
        self.fps = fps
        self.base_framerate = 30
        self.cache = cache
        self.transforms = transforms
        # Maps the first dataset index to the corresponding video reader and MANO reader
        self.clip_to_data = {}
        self.crop_transform = CropHand(output_size=crop_size, padding_factor=padding_factor)
        self.mirror_transform = VideoMirror(p=1)
        self.crop_size = crop_size

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
                    intrinsics = np.array(
                        [
                            [fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1],
                        ],
                        dtype=np.float32,
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

                    bbox_dir_path = Path(scene_path) / directory / camera / "predicted_bboxes.txt"
                    self.bbox_readers.append(
                        BboxReader(
                            bbox_path=bbox_dir_path,
                            single_file=True,
                            use_kalman=True,
                            measurement_noise=0.1,
                            min_bbox_diagonal=10,
                        ),
                    )

                    joints_dir_path = Path(scene_path) / directory / camera / "joints"
                    self.joints_readers.append(
                        JointsReader(
                            joints_dir_path=joints_dir_path,
                            fmt_frame_fn=lambda x: f"{x:06d}.json",
                        ),
                    )

                    self.clip_to_data[self.num_clips] = (
                        self.video_readers[-1],
                        self.mano_readers[-1],
                        self.bbox_readers[-1],
                        self.joints_readers[-1],
                        self.camera_intrinsics[-1],
                    )

                    # Calculate the number of clips based on the desired fps
                    step = int(self.base_framerate // self.fps)
                    full_starts = int(len(self.video_readers[-1]) // (step * self.num_frames))
                    partials = max(
                        (len(self.video_readers[-1]) + step - (full_starts + 1) * step * self.num_frames),
                        0,
                    )
                    total_len = full_starts * step + partials
                    self.num_clips += total_len

    def __len__(self) -> int:
        """Get the total number of clips in the dataset.

        Returns:
            int: The number of clips in the dataset.

        """
        return 2 * self.num_clips

    def _get_clip_data(
        self,
        clip_idx: int,
    ) -> tuple[VideoReader, ManoReader, BboxReader, JointsReader, np.ndarray, int]:
        """Get the readers, camera intrinsics and start frame for a given clip index.

        Args:
            clip_idx (int): The index of the clip in the dataset.

        Returns:
            tuple[VideoReader, ManoReader, BboxReader, JointsReader, np.ndarray, int]: A tuple containing:
                - The VideoReader instance for the clip.
                - The ManoReader instance for the clip.
                - The BboxReader instance for the clip.
                - The JointsReader instance for the clip.
                - The camera intrinsic matrix with shape (3, 3).
                - The start frame index of the clip within its video.

        """
        # Find the corresponding readers
        video_idx, video, mano, bbox, joints, intrinsics = max(
            (i, video, mano, bbox, joints, intrinsics)
            for i, (video, mano, bbox, joints, intrinsics) in self.clip_to_data.items()
            if i <= clip_idx
        )

        # Calculate the start frame of the clip within the video
        clip_idx_in_video = clip_idx - video_idx
        step = int(self.base_framerate // self.fps)
        full = clip_idx_in_video // step
        partial = clip_idx_in_video % step
        start_frame = full * (step * self.num_frames) + partial

        return video, mano, bbox, joints, intrinsics, start_frame

    @staticmethod
    @cache
    def _get_video_frames(video_reader: VideoReader, start_frame: int, num_frames: int, step: int) -> list[np.ndarray]:
        """Get video frames from the video reader for a given clip.

        Args:
            video_reader (VideoReader): The video reader instance.
            start_frame (int): The start frame index.
            num_frames (int): The number of frames to retrieve.
            step (int): The step size between frames.

        Returns:
            list[np.ndarray]: A list of video frames with shape (H, W, C).

        """
        frames = video_reader.get_frames(list(range(start_frame, start_frame + num_frames * step, step)))

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
        step: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, bool]]]:
        """Get MANO parameters from the MANO reader for a given clip.

        Args:
            mano_reader (ManoReader): The MANO reader instance.
            start_frame (int): The start frame index.
            num_frames (int): The number of frames to retrieve.
            step (int): The step size between frames.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[dict[str, bool]]]: A tuple containing:
                - The first list contains MANO parameters for the left hand for each frame.
                - The second list contains MANO parameters for the right hand for each frame.
                - The third list contains dictionaries indicating availability of each hand for each frame.

        """
        return mano_reader.get_mano_sequence(list(range(start_frame, start_frame + num_frames * step, step)))

    @staticmethod
    @cache
    def _get_bbox_data(
        bbox_reader: BboxReader,
        start_frame: int,
        num_frames: int,
        step: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get bounding box data from the bbox reader for a given clip.

        Args:
            bbox_reader (BboxReader): The bbox reader instance.
            start_frame (int): The start frame index.
            num_frames (int): The number of frames to retrieve.
            step (int): The step size between frames.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists of numpy arrays:
                - The first list contains bbox coordinates for the left hand for each frame.
                - The second list contains bbox coordinates for the right hand for each frame.

        """
        return bbox_reader.get_bbox_sequence(list(range(start_frame, start_frame + num_frames * step, step)))

    @staticmethod
    @cache
    def _get_joints_data(
        joints_reader: JointsReader,
        start_frame: int,
        num_frames: int,
        step: int,
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get joints data from the joints reader for a given clip.

        Args:
            joints_reader (JointsReader): The joints reader instance.
            start_frame (int): The start frame index.
            num_frames (int): The number of frames to retrieve.
            step (int): The step size between frames.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists of numpy arrays:
                - The first list contains joints coordinates for the left hand for each frame.
                - The second list contains joints coordinates for the right hand for each frame.

        """
        return joints_reader.get_joints_sequence(list(range(start_frame, start_frame + num_frames * step, step)))

    def _load_data(
        self,
        video_reader: VideoReader,
        mano_reader: ManoReader,
        bbox_reader: BboxReader,
        joints_reader: JointsReader,
        start_frame: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, bool]],
    ]:
        """Load and preprocess data from readers.

        Args:
            video_reader (VideoReader): Video reader instance.
            mano_reader (ManoReader): MANO reader instance.
            bbox_reader (BboxReader): Bbox reader instance.
            joints_reader (JointsReader): Joints reader instance.
            start_frame (int): Start frame index.

        Returns:
            tuple containing:
                - Video frames tensor
                - Left MANO parameters tensor
                - Right MANO parameters tensor
                - Left bbox tensor
                - Right bbox tensor
                - Left joints tensor
                - Right joints tensor
                - Hand availability

        """
        step = int(self.base_framerate // self.fps)
        if self.cache:
            frames = self._get_video_frames(video_reader, start_frame, self.num_frames, step)
            mano_params_left, mano_params_right, hand_availables = self._get_mano_params(
                mano_reader,
                start_frame,
                self.num_frames,
                step,
            )
            bbox_left, bbox_right = self._get_bbox_data(bbox_reader, start_frame, self.num_frames, step)
            joints_left, joints_right = self._get_joints_data(joints_reader, start_frame, self.num_frames, step)
        else:
            frames = self._get_video_frames.__wrapped__(video_reader, start_frame, self.num_frames, step)
            mano_params_left, mano_params_right, hand_availables = self._get_mano_params.__wrapped__(
                mano_reader,
                start_frame,
                self.num_frames,
                step,
            )
            bbox_left, bbox_right = self._get_bbox_data.__wrapped__(
                bbox_reader,
                start_frame,
                self.num_frames,
                step,
            )
            joints_left, joints_right = self._get_joints_data.__wrapped__(
                joints_reader,
                start_frame,
                self.num_frames,
                step,
            )

        # Convert list of numpy arrays to PyTorch tensors
        clip = torch.from_numpy(np.stack(frames))
        mano_left = torch.from_numpy(np.stack(mano_params_left))
        mano_right = torch.from_numpy(np.stack(mano_params_right))
        bbox_left = torch.from_numpy(np.stack(bbox_left))
        bbox_right = torch.from_numpy(np.stack(bbox_right))
        joints_left = torch.from_numpy(np.stack(joints_left))
        joints_right = torch.from_numpy(np.stack(joints_right))

        return clip, mano_left, mano_right, bbox_left, bbox_right, joints_left, joints_right, hand_availables

    def _process_original_clip(
        self,
        clip: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process the original clip by center cropping and resizing.

        Args:
            clip (torch.Tensor): Original video clip tensor.
            intrinsics (torch.Tensor): Original camera intrinsics.

        Returns:
            tuple containing:
                - Processed video clip tensor
                - Adjusted camera intrinsics

        """
        clip_original = clip.clone()
        intrinsics_original = intrinsics.clone()

        # Center crop to square shape
        _, _, H, W = clip_original.shape  # noqa: N806
        if H > W:
            start = (H - W) // 2
            clip_original = clip_original[:, :, start : start + W, :]
            intrinsics_original[1, 2] -= start  # Adjust y-offset
        elif W > H:
            start = (W - H) // 2
            clip_original = clip_original[:, :, :, start : start + H]
            intrinsics_original[0, 2] -= start  # Adjust x-offset

        # Scale to target size
        current_size = clip_original.shape[-1]
        scale = self.crop_size / current_size
        intrinsics_original[:2, :] = intrinsics_original[:2, :] * scale
        clip_original = F.resize(clip_original, size=[self.crop_size, self.crop_size], antialias=True)

        return clip_original, intrinsics_original

    def _get_hand_data(
        self,
        return_right_hand: bool,
        mano_left: torch.Tensor,
        mano_right: torch.Tensor,
        bbox_left: torch.Tensor,
        bbox_right: torch.Tensor,
        joints_left: torch.Tensor,
        joints_right: torch.Tensor,
        hand_availables: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get data for the specified hand.

        Args:
            return_right_hand (bool): Whether to return right hand data.
            mano_left (torch.Tensor): Left hand MANO parameters.
            mano_right (torch.Tensor): Right hand MANO parameters.
            bbox_left (torch.Tensor): Left hand bounding boxes.
            bbox_right (torch.Tensor): Right hand bounding boxes.
            joints_left (torch.Tensor): Left hand joints.
            joints_right (torch.Tensor): Right hand joints.
            hand_availables (torch.Tensor): Hand availability flags.

        Returns:
            tuple containing:
                - MANO parameters for selected hand
                - Bounding boxes for selected hand
                - Joints for selected hand
                - Hand availability flags

        """
        if return_right_hand:
            return (
                mano_right,
                bbox_right,
                joints_right,
                torch.tensor([h["right"] for h in hand_availables], dtype=torch.float32),
            )
        return (
            mano_left,
            bbox_left,
            joints_left,
            torch.tensor([h["left"] for h in hand_availables], dtype=torch.float32),
        )

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a video clip and corresponding MANO parameters, joints and 2D joints from the dataset.

        This method loads frames, MANO parameters, 3D joint coordinates, and 2D joint coordinates
        from a single video clip specified by the index. The frames are cropped around either
        the left or right hand based on the index. If idx >= num_clips, the right hand is processed,
        otherwise the left hand.

        Args:
            idx (int): The index of the video clip to retrieve. Values [0, num_clips-1] process left hand,
                      values [num_clips, 2*num_clips-1] process right hand.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                - A tensor of cropped video frames with shape (T, C, H, W), where T is the number of frames,
                  C is the number of channels (3), and H=W=output_size. Values are normalized with
                  mean (0.45, 0.45, 0.45) and std (0.225, 0.225, 0.225).
                - A tensor of MANO parameters with shape (T, 61), where T is the number of frames.
                  Contains translation (3), pose (45) and shape (10) parameters.
                - A tensor of 3D joint coordinates with shape (T, J, 3), where T is the number of frames
                  and J is the number of joints.
                - A tensor of 2D joint coordinates with shape (T, J, 2), where T is the number of frames
                  and J is the number of joints.
                - A tensor of hand availability flags with shape (T,), where T is the number of frames.
                  1 indicates the hand is available, 0 indicates it is not.
                - A tensor of original non-cropped video frames with shape (T, C, H, W), normalized with
                  mean (0.45, 0.45, 0.45) and std (0.225, 0.225, 0.225).
                - A tensor of camera intrinsics for the original resized view with shape (3, 3).

        Raises:
            IndexError: If the provided index is out of range [0, 2*num_clips-1].

        """
        if idx >= len(self):
            msg = f"Index {idx} out of range. Total clips: {len(self)}"
            raise IndexError(msg)

        return_right_hand = idx >= self.num_clips
        if return_right_hand:
            idx = idx - self.num_clips

        # Get readers and start frame
        video_reader, mano_reader, bbox_reader, joints_reader, intrinsics, start_frame = self._get_clip_data(idx)

        # Load all data
        clip, mano_left, mano_right, bbox_left, bbox_right, joints_left, joints_right, hand_availables = (
            self._load_data(
                video_reader,
                mano_reader,
                bbox_reader,
                joints_reader,
                start_frame,
            )
        )
        intrinsics = torch.from_numpy(np.stack(intrinsics))

        # Get data for the selected hand
        mano_current, bbox_current, joints_current, hand_available = self._get_hand_data(
            return_right_hand,
            mano_left,
            mano_right,
            bbox_left,
            bbox_right,
            joints_left,
            joints_right,
            hand_availables,
        )

        # Process original clip
        clip_original, intrinsics_original = self._process_original_clip(clip, intrinsics.to(torch.float32))

        # Project 2D joints for original view
        mano_trans = mano_current[..., :3].unsqueeze(1).clone() * 1000  # Scale to millimeters
        joints_2d_original = project_joints_to_2d(
            (joints_current + mano_trans).unsqueeze(0),
            intrinsics_original,
        ).squeeze(0)

        # Apply hand crop transform
        clip_current, _ = self.crop_transform(clip, bbox_current, joints_2d_original)

        # Mirror transform for left hand
        if not return_right_hand:
            clip_current, mano_current, joints_current, _ = self.mirror_transform(
                clip_current,
                mano_current,
                joints_current,
                None,
            )
            # Also mirror the original clip and 2D joints
            clip_original, _, _, joints_2d_original = self.mirror_transform(
                clip_original,
                None,  # Unused
                None,  # Unused
                joints_2d_original,
            )

        # Apply additional transforms
        if self.transforms is not None:
            for transform in self.transforms:
                clip_current, mano_current, intrinsics = transform(clip_current, mano_current, intrinsics)

        # Normalize both clips
        clip_current = F.normalize(clip_current, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225))
        clip_original = F.normalize(clip_original, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225))

        return (
            clip_current,
            mano_current,
            joints_current,
            joints_2d_original,
            hand_available,
            clip_original,
            intrinsics_original,
        )
