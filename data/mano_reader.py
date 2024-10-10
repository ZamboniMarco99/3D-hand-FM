"""ManoReader class for efficient reading and processing of MANO hand model files.

This module provides functionality to:
- Read MANO (hand model) pose and shape parameters from a directory of files
- Support virtual frame indexing for flexible MANO data processing
- Convert between real and virtual frame indices based on assumed frame rates
- Decode MANO parameters into a structured format for both left and right hands

The ManoReader class offers an efficient way to handle MANO data in conjunction with
video processing tasks, allowing for synchronization between video frames and hand pose data.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np


class ManoReader:
    """A class for reading MANO files efficiently.

    This class provides methods to read MANO (hand model) pose and shape parameters from a directory
    containing individual MANO files. It supports virtual frame indexing for flexible MANO data processing.

    Attributes:
        mano_dir_path (Path): Path to directory containing MANO files.
        assumed_fps (float): Assumed frames per second for virtual frame conversion.
        fmt_frame_fn (Callable[[int], str] | None): Function to transform frame indexes into filenames.
        mano_len (int): Total number of MANO files in the directory.

    """

    def __init__(
        self,
        mano_dir_path: str | Path,
        assumed_fps: float | None = None,
        fmt_frame_fn: Callable[[int], str] | None = None,
    ) -> None:
        """Initialize the ManoReader.

        Args:
            mano_dir_path (str | Path): Path to directory containing MANO files.
            assumed_fps (float | None, optional): Assumed frames per second for virtual frame conversion.
                If None, no frame rate conversion will be performed. Defaults to None.
            fmt_frame_fn (Callable[[int], str] | None, optional): Function to transform frame indexes into filenames.
                If None, a default naming convention will be used. Defaults to None.

        """
        self.mano_dir_path = Path(mano_dir_path)
        self.assumed_fps = assumed_fps
        self.fmt_frame_fn = fmt_frame_fn

        self.mano_len = len(list(self.mano_dir_path.glob("*.txt")))

    @staticmethod
    def decode_mano(mano_params: np.ndarray) -> dict:
        """Decode MANO parameters from a numpy array.

        Args:
            mano_params (np.ndarray): Numpy array containing MANO parameters for both hands.

        Returns:
            dict: Dictionary containing decoded MANO parameters for left and right hands,
                  including translation, pose, and shape for each hand.

        """
        return {
            "left_tran": np.expand_dims(mano_params[1:4], 0).astype(np.float32),
            "left_pose": np.expand_dims(mano_params[4:52], 0).astype(np.float32),
            "left_shape": np.expand_dims(mano_params[52:62], 0).astype(np.float32),
            "right_tran": np.expand_dims(mano_params[63:66], 0).astype(np.float32),
            "right_pose": np.expand_dims(mano_params[66:114], 0).astype(np.float32),
            "right_shape": np.expand_dims(mano_params[114:124], 0).astype(np.float32),
        }

    def get_mano(self, frame_idx: int) -> dict:
        """Retrieve MANO parameters for a single frame.

        Args:
            frame_idx (int): The index of the frame to retrieve.

        Returns:
            dict: Dictionary containing raw MANO parameters for the specified frame.

        Raises:
            FileNotFoundError: If the MANO file for the specified frame is not found.

        """
        file_name = self.fmt_frame_fn(frame_idx)
        file_path = self.mano_dir_path / file_name
        if not file_path.exists():
            message = f"MANO file at {file_path} not found for frame {frame_idx}"
            raise FileNotFoundError(message)
        return np.loadtxt(file_path, dtype=np.float32)

    def get_mano_sequence(self, frame_idxs: list[int]) -> list[np.ndarray]:
        """Retrieve MANO parameters for a sequence of frames.

        Args:
            frame_idxs (list[int]): List of frame indices to retrieve.

        Returns:
            list[np.ndarray]: List of MANO parameters for the specified frames.

        """
        return [self.get_mano(frame_idx) for frame_idx in frame_idxs]

    def __len__(self) -> int:
        """Get the total number of MANO files.

        Returns:
            int: The number of MANO files in the directory.

        """
        return self.mano_len

    def __getitem__(self, idx: int) -> np.ndarray:
        """Retrieve decoded MANO parameters for a single frame.

        This method allows the ManoReader to be used as an iterable,
        returning decoded MANO parameters based on the given index.

        Args:
            idx (int): The index of the frame to retrieve.

        Returns:
            np.ndarray: A 1D numpy array containing concatenated MANO parameters
                        for both hands (left and right) for the specified frame.
                        The array includes translation, pose, and shape parameters
                        for each hand in the following order:
                        [left_tran, left_pose, left_shape, right_tran, right_pose, right_shape]

        Raises:
            FileNotFoundError: If the MANO file for the specified frame is not found.

        """
        return np.concatenate(self.get_mano(idx).values(), axis=1)
