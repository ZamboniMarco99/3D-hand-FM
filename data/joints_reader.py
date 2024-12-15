"""JointsReader class for efficient reading and processing of hand joint files.

This module provides functionality to:
- Read joint coordinates from a directory of JSON files
- Support frame indexing for flexible joint data processing
- Process joint data for both left and right hands
- Each file contains "joints_left" and "joints_right" data
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np


class JointsReader:
    """A class for reading joint coordinate files efficiently.

    This class provides methods to read joint coordinates from a directory
    containing individual JSON files. Each file contains coordinates for left and right hands.

    Attributes:
        joints_dir_path (Path): Path to directory containing joint files.
        fmt_frame_fn (Callable[[int], str] | None): Function to transform frame indexes into filenames.
        joints_len (int): Total number of joint files in the directory.

    """

    def __init__(
        self,
        joints_dir_path: str | Path,
        fmt_frame_fn: Callable[[int], str] | None = None,
    ) -> None:
        """Initialize the JointsReader.

        Args:
            joints_dir_path (str | Path): Path to directory containing joint files.
            fmt_frame_fn (Callable[[int], str] | None, optional): Function to transform frame indexes into filenames.
                If None, a default naming convention will be used. Defaults to None.

        """
        self.joints_dir_path = Path(joints_dir_path)
        self.fmt_frame_fn = fmt_frame_fn
        self.joints_len = len(list(self.joints_dir_path.glob("*.json")))

    @staticmethod
    def decode_joints(joints_data: dict) -> tuple[np.ndarray, np.ndarray]:
        """Decode joint coordinates into separate arrays for left and right hands.

        Args:
            joints_data (dict): Dictionary containing joint coordinates for both hands
                with keys "joints_left" and "joints_right".

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array contains joint coordinates for the left hand.
                - The second array contains joint coordinates for the right hand.

        """
        joints_left = np.array(joints_data["left_joints"][0], dtype=np.float32)
        joints_right = np.array(joints_data["right_joints"][0], dtype=np.float32)
        return joints_left, joints_right

    def get_joints(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve joint coordinates for a single frame.

        Args:
            frame_idx (int): The index of the frame to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array contains joint coordinates for the left hand.
                - The second array contains joint coordinates for the right hand.

        Raises:
            FileNotFoundError: If the joints file for the specified frame is not found.

        """
        file_name = self.fmt_frame_fn(frame_idx)
        file_path = self.joints_dir_path / file_name
        if not file_path.exists():
            message = f"Joints file at {file_path} not found for frame {frame_idx}"
            raise FileNotFoundError(message)

        with file_path.open() as f:
            joints_data = json.load(f)
        return self.decode_joints(joints_data)

    def get_joints_sequence(self, frame_idxs: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Retrieve joint coordinates for a sequence of frames.

        Args:
            frame_idxs (list[int]): List of frame indices to retrieve.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists:
                - The first list contains joint coordinates for the left hand for each frame.
                - The second list contains joint coordinates for the right hand for each frame.

        """
        left_hands_joints = []
        right_hands_joints = []
        for frame_idx in frame_idxs:
            left, right = self.get_joints(frame_idx)
            left_hands_joints.append(left)
            right_hands_joints.append(right)
        return left_hands_joints, right_hands_joints

    def __len__(self) -> int:
        """Get the total number of joint files.

        Returns:
            int: The number of joint files in the directory.

        """
        return self.joints_len

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve joint coordinates for a single frame.

        This method allows the JointsReader to be used as an iterable,
        returning joint coordinates based on the given index.

        Args:
            idx (int): The index of the frame to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array contains joint coordinates for the left hand.
                - The second array contains joint coordinates for the right hand.

        Raises:
            FileNotFoundError: If the joints file for the specified frame is not found.

        """
        return self.get_joints(idx)
