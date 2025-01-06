"""Dataset that returns a video clip as input and the last frame's parameters as target.

This dataset inherits from H2ODataset and modifies its behavior to return a video clip
as input (x) and the MANO parameters, joints, and 2D joints of the last frame in that
clip as the target (y). Unlike the parent class which has one item per clip, this dataset
has one item for each possible frame that could be the last frame of a clip.
"""

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from data.bbox_reader import BboxReader
from data.h2o_datamodule import H2ODataset
from data.joints_reader import JointsReader
from data.mano_reader import ManoReader
from data.video_reader import VideoReader
from models.utils import project_joints_to_2d


class H2OLastFrameDataset(H2ODataset):
    """Dataset that returns a video clip and the last frame's parameters.

    This dataset inherits from H2ODataset and modifies its behavior to return:
    - x: a video clip ending at frame t
    - y: the MANO parameters, joints, and 2D joints of frame t

    The dataset is useful for training models that predict hand pose parameters
    for the last frame of a video sequence. Unlike the parent class which has
    one item per clip, this dataset has one item for each possible frame that
    could be the last frame of a clip.
    """

    def __init__(
        self,
        dataset_prefix: str,
        scenes: list[str],
        cameras: list[str],
        num_frames: int = 300,
        fps: float = 7.5,
        cache: bool = True,
        transforms: list[nn.Module] | None = None,
        crop_size: int = 224,
        padding_factor: float = 1.2,
    ) -> None:
        """Initialize the dataset.

        Args:
            dataset_prefix (str): The prefix path to the dataset.
            scenes (list[str]): List of scene names to include in the dataset.
            cameras (list[str]): List of camera names to include in the dataset.
            num_frames (int, optional): Number of frames to include per video. Defaults to 300.
            fps (float, optional): Desired frames per second. Defaults to 7.5.
            cache (bool, optional): If True, enable caching of video frames. Defaults to True.
            transforms (list[nn.Module] | None, optional): List of video transform modules to apply.
            crop_size (int, optional): Size of the output square crop in pixels. Defaults to 224.
            padding_factor (float, optional): Factor to increase the crop size by. Defaults to 1.2.

        """
        super().__init__(
            dataset_prefix=dataset_prefix,
            scenes=scenes,
            cameras=cameras,
            num_frames=num_frames,
            fps=fps,
            cache=cache,
            transforms=transforms,
            crop_size=crop_size,
            padding_factor=padding_factor,
        )

        # Calculate total number of frames during initialization
        total_frames = 0
        for video_reader, *_ in self.clip_to_data.values():
            step = int(self.base_framerate // self.fps)
            usable_frames = max(0, len(video_reader) - (self.num_frames - 1) * step)
            total_frames += usable_frames

        # Multiply by 2 because we process both left and right hands
        self._length = 2 * total_frames

    def __len__(self) -> int:
        """Get the total number of possible target frames in the dataset.

        Returns:
            int: The number of frames that could be targets (twice the number of frames
                because we process both left and right hands).

        """
        return self._length

    def _get_clip_data(
        self,
        idx: int,
    ) -> tuple[VideoReader, ManoReader, BboxReader, JointsReader, torch.Tensor, int]:
        """Get the readers, camera intrinsics and start frame for a given target frame index.

        This method maps a target frame index to the corresponding video and frame number.
        It ensures that there are enough previous frames to form a complete clip ending
        at the target frame.

        Args:
            idx (int): Global index identifying a specific target frame.

        Returns:
            tuple: Contains the readers, camera intrinsics, and the start frame such that
                  the clip will end at the target frame.

        """
        # First, map the index to a specific video and frame
        frames_so_far = 0
        target_video_idx = None
        target_frame = None
        self.return_right_hand = False

        if idx >= self._length // 2:
            idx = idx - self._length // 2
            self.return_right_hand = True

        for video_idx, (video_reader, *_) in self.clip_to_data.items():
            step = int(self.base_framerate // self.fps)
            usable_frames = max(0, len(video_reader) - (self.num_frames - 1) * step)

            if frames_so_far + usable_frames > idx:
                target_video_idx = video_idx
                # The target frame is the frame at this index in this video
                target_frame = idx - frames_so_far + (self.num_frames - 1) * step
                break

            frames_so_far += usable_frames

        if target_video_idx is None or target_frame is None:
            msg = f"Index {idx} is out of range"
            raise IndexError(msg)

        # Get the video reader and other data for this video
        video_reader, mano_reader, bbox_reader, joints_reader, intrinsics = self.clip_to_data[target_video_idx]

        # Calculate the start frame such that the clip will end at the target frame
        start_frame = target_frame - (self.num_frames - 1) * step

        return video_reader, mano_reader, bbox_reader, joints_reader, intrinsics, start_frame

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve a video clip and corresponding parameters of its last frame.

        This method returns a video clip ending at frame t and the parameters of frame t.

        Args:
            idx (int): The index identifying a specific target frame.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - A tensor of video frames with shape (T, C, H, W), where T is the number of frames,
                  C is the number of channels (3), and H=W=output_size.
                - A tensor of MANO parameters with shape (1, 61) for the target frame, containing
                  translation (3), pose (45) and shape (10) parameters.
                - A tensor of 3D joint coordinates with shape (1, J, 3) for the target frame,
                  where J is the number of joints.
                - A tensor of 2D joint coordinates with shape (1, J, 2) for the target frame,
                  where J is the number of joints.

        """
        # Get the readers and frame information
        video_reader, mano_reader, bbox_reader, joints_reader, intrinsics, start_frame = self._get_clip_data(idx)

        # Calculate step and target frame
        step = int(self.base_framerate // self.fps)

        # Get frames and data for the clip
        if self.cache:
            frames = self._get_video_frames(video_reader, start_frame, self.num_frames, step)
            mano_left, mano_right = self._get_mano_params(mano_reader, start_frame, self.num_frames, step)
            bbox_left, bbox_right = self._get_bbox_data(bbox_reader, start_frame, self.num_frames, step)
            joints_left, joints_right = self._get_joints_data(joints_reader, start_frame, self.num_frames, step)
        else:
            frames = self._get_video_frames.__wrapped__(video_reader, start_frame, self.num_frames, step)
            mano_left, mano_right = self._get_mano_params.__wrapped__(mano_reader, start_frame, self.num_frames, step)
            bbox_left, bbox_right = self._get_bbox_data.__wrapped__(bbox_reader, start_frame, self.num_frames, step)
            joints_left, joints_right = self._get_joints_data.__wrapped__(
                joints_reader,
                start_frame,
                self.num_frames,
                step,
            )

        # Convert to tensors
        clip = torch.from_numpy(np.stack(frames))
        mano_sequence = torch.from_numpy(np.stack(mano_right if self.return_right_hand else mano_left))
        bbox_current = torch.from_numpy(np.stack(bbox_right if self.return_right_hand else bbox_left))
        joints_sequence = torch.from_numpy(np.stack(joints_right if self.return_right_hand else joints_left))
        intrinsics = torch.from_numpy(intrinsics).to(torch.float32)

        # Project joints to 2D for all frames
        mano_trans = mano_sequence[..., :3].unsqueeze(1).clone()
        # Scale to milimeters
        mano_trans = mano_trans * 1000
        joints_2d_sequence = project_joints_to_2d(
            (joints_sequence + mano_trans).unsqueeze(0),
            intrinsics,
        ).squeeze(0)

        # Apply CropHand transform for the current hand only
        clip_current, joints_2d_sequence = self.crop_transform(
            clip,
            bbox_current,
            joints_2d_sequence,
        )

        # Mirror if left hand
        if not self.return_right_hand:
            clip_current, mano_sequence, joints_sequence, joints_2d_sequence = self.mirror_transform(
                clip_current,
                mano_sequence,
                joints_sequence,
                joints_2d_sequence,
            )

        # Apply additional transforms if provided
        if self.transforms is not None:
            for transform in self.transforms:
                clip_current, mano_sequence, intrinsics = transform(
                    clip_current,
                    mano_sequence,
                    intrinsics,
                )

        # Normalize the cropped clip
        clip_current = F.normalize(clip_current, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225))

        # Get only the last frame's parameters for the target
        mano_current = mano_sequence[-1:]  # Shape: (1, 61)
        joints_current = joints_sequence[-1:]  # Shape: (1, J, 3)
        joints_2d_current = joints_2d_sequence[-1:]  # Shape: (1, J, 2)

        return clip_current, mano_current, joints_current, joints_2d_current
