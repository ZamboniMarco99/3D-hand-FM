"""BboxReader class for efficient reading and processing of hand bounding box files.

This module provides functionality to:
- Read bounding box coordinates from a directory of files
- Support frame indexing for flexible bbox data processing
- Process bbox data for both left and right hands
- Each file contains two lines with [x_min, y_min, x_max, y_max] coordinates
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np


class BboxReader:
    """A class for reading bounding box files efficiently.

    This class provides methods to read bounding box coordinates from a directory
    containing individual bbox files. Each file contains coordinates for left and right hands.

    Attributes:
        bbox_dir_path (Path): Path to directory containing bbox files.
        fmt_frame_fn (Callable[[int], str] | None): Function to transform frame indexes into filenames.
        bbox_len (int): Total number of bbox files in the directory.

    """

    def __init__(
        self,
        bbox_dir_path: str | Path,
        fmt_frame_fn: Callable[[int], str] | None = None,
    ) -> None:
        """Initialize the BboxReader.

        Args:
            bbox_dir_path (str | Path): Path to directory containing bbox files.
            fmt_frame_fn (Callable[[int], str] | None, optional): Function to transform frame indexes into filenames.
                If None, a default naming convention will be used. Defaults to None.

        """
        self.bbox_dir_path = Path(bbox_dir_path)
        self.fmt_frame_fn = fmt_frame_fn
        self.bbox_len = len(list(self.bbox_dir_path.glob("*.txt")))

    @staticmethod
    def decode_bbox(bbox_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode bbox coordinates into separate arrays for left and right hands.

        Args:
            bbox_data (np.ndarray): Numpy array containing bbox coordinates for both hands.
                Expected shape: (2, 4) where:
                - First row: left hand [x_min, y_min, x_max, y_max]
                - Second row: right hand [x_min, y_min, x_max, y_max]

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array contains bbox coordinates for the left hand.
                - The second array contains bbox coordinates for the right hand.
                Each array has shape (4,) representing [x_min, y_min, x_max, y_max].

        """
        return bbox_data[0].astype(np.float32), bbox_data[1].astype(np.float32)

    def get_bbox(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve bbox coordinates for a single frame.

        Args:
            frame_idx (int): The index of the frame to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array contains bbox coordinates for the left hand.
                - The second array contains bbox coordinates for the right hand.
                Each array has shape (4,) representing [x_min, y_min, x_max, y_max].

        Raises:
            FileNotFoundError: If the bbox file for the specified frame is not found.

        """
        file_name = self.fmt_frame_fn(frame_idx)
        file_path = self.bbox_dir_path / file_name
        if not file_path.exists():
            message = f"Bbox file at {file_path} not found for frame {frame_idx}"
            raise FileNotFoundError(message)
        bbox_data = np.loadtxt(file_path, dtype=np.float32)
        return self.decode_bbox(bbox_data)

    def get_bbox_sequence(self, frame_idxs: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Retrieve bbox coordinates for a sequence of frames.

        Args:
            frame_idxs (list[int]): List of frame indices to retrieve.

        Returns:
            tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists:
                - The first list contains bbox coordinates for the left hand for each frame.
                - The second list contains bbox coordinates for the right hand for each frame.
                Each numpy array in the lists has shape (4,) representing [x_min, y_min, x_max, y_max].

        """
        left_hands_bboxes = []
        right_hands_bboxes = []
        for frame_idx in frame_idxs:
            left, right = self.get_bbox(frame_idx)
            left_hands_bboxes.append(left)
            right_hands_bboxes.append(right)
        return left_hands_bboxes, right_hands_bboxes

    def __len__(self) -> int:
        """Get the total number of bbox files.

        Returns:
            int: The number of bbox files in the directory.

        """
        return self.bbox_len

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve bbox coordinates for a single frame.

        This method allows the BboxReader to be used as an iterable,
        returning bbox coordinates based on the given index.

        Args:
            idx (int): The index of the frame to retrieve.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays:
                - The first array contains bbox coordinates for the left hand.
                - The second array contains bbox coordinates for the right hand.
                Each array has shape (4,) representing [x_min, y_min, x_max, y_max].

        Raises:
            FileNotFoundError: If the bbox file for the specified frame is not found.

        """
        return self.get_bbox(idx)
