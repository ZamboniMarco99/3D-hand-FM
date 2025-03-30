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
        data_format: str | None = None,
    ) -> None:
        """Initialize the ManoReader.

        Args:
            mano_dir_path (str | Path): Path to directory containing MANO files.
            assumed_fps (float | None, optional): Assumed frames per second for virtual frame conversion.
                If None, no frame rate conversion will be performed. Defaults to None.
            fmt_frame_fn (Callable[[int], str] | None, optional): Function to transform frame indexes into filenames.
                If None, a default naming convention will be used. Defaults to None.
            data_format (str | None, optional): Format of the MANO data files. Defaults to None.

        """
        self.mano_dir_path = Path(mano_dir_path)
        self.assumed_fps = assumed_fps
        self.fmt_frame_fn = fmt_frame_fn
        self.data_format = data_format

        self.mano_len = len(list(self.mano_dir_path.glob("*.txt")))

    @staticmethod
    def decode_mano(mano_params: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, bool]]:
        """Decode MANO parameters from a numpy array into separate dictionaries for left and right hands.

        Args:
            mano_params (np.ndarray): Numpy array containing MANO parameters for both hands.
                Expected shape: (124,) where indices correspond to:
                - 0: left hand availability flag (0 if unavailable)
                - 1-3: left hand translation
                - 4-51: left hand pose (48 parameters)
                - 52-61: left hand shape (10 parameters)
                - 62: right hand availability flag (0 if unavailable)
                - 63-65: right hand translation
                - 66-113: right hand pose (48 parameters)
                - 114-123: right hand shape (10 parameters)

        Returns:
            tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, bool]]: A tuple containing:
                - The first dictionary contains MANO parameters for the left hand.
                - The second dictionary contains MANO parameters for the right hand.
                Each dictionary has keys 'tran', 'pose', and 'shape' with corresponding numpy array values.
                - The third dictionary contains availability flags for each hand with keys 'left' and 'right'.

        """
        # Extract availability flags
        hand_available = {
            "left": bool(mano_params[0]),
            "right": bool(mano_params[62]),
        }

        mano_params = {
            "left_tran": mano_params[1:4].astype(np.float32),
            "left_pose": mano_params[4:52].astype(np.float32),
            "left_shape": mano_params[52:62].astype(np.float32),
            "right_tran": mano_params[63:66].astype(np.float32),
            "right_pose": mano_params[66:114].astype(np.float32),
            "right_shape": mano_params[114:124].astype(np.float32),
        }

        mano_params_left = {
            "tran": mano_params["left_tran"],
            "pose": mano_params["left_pose"],
            "shape": mano_params["left_shape"],
        }
        mano_params_right = {
            "tran": mano_params["right_tran"],
            "pose": mano_params["right_pose"],
            "shape": mano_params["right_shape"],
        }

        return mano_params_left, mano_params_right, hand_available

    def get_mano(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray, dict[str, bool]]:
        """Retrieve MANO parameters for a single frame.

        Args:
            frame_idx (int): The index of the frame to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray, dict[str, bool]]: A tuple containing:
                - The first array contains MANO parameters for the left hand.
                - The second array contains MANO parameters for the right hand.
                Each array has shape (61,) where the first 3 elements are translation,
                the next 45 are pose parameters, and the last 10 are shape parameters.
                - A dictionary indicating availability of each hand with keys 'left' and 'right'.

        Raises:
            FileNotFoundError: If the MANO file for the specified frame is not found.

        """
        file_name = self.fmt_frame_fn(frame_idx)
        file_path = self.mano_dir_path / file_name
        if not file_path.exists():
            message = f"MANO file at {file_path} not found for frame {frame_idx}"
            raise FileNotFoundError(message)
        mano_params_left, mano_params_right, hand_available = self.decode_mano(np.loadtxt(file_path, dtype=np.float32))
        return (
            np.concatenate(list(mano_params_left.values())),
            np.concatenate(list(mano_params_right.values())),
            hand_available,
        )

    def get_mano_sequence(
        self,
        frame_idxs: list[int],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, bool]]]:
        """Retrieve MANO parameters for a sequence of frames.

        Args:
            frame_idxs (list[int]): List of frame indices to retrieve.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray], list[dict[str, bool]]]: A tuple containing:
                - The first list contains MANO parameters for the left hand for each frame.
                - The second list contains MANO parameters for the right hand for each frame.
                Each numpy array in the lists has shape (61,) where the first 3 elements are translation,
                the next 45 are pose parameters, and the last 10 are shape parameters.
                - The third list contains dictionaries indicating availability of each hand for each frame.

        """
        left_hands = []
        right_hands = []
        hand_availables = []
        for frame_idx in frame_idxs:
            left, right, hand_available = self.get_mano(frame_idx)
            left_hands.append(left)
            right_hands.append(right)
            hand_availables.append(hand_available)
        return left_hands, right_hands, hand_availables

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
        return self.get_mano(idx).values()
