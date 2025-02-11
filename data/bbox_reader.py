"""BboxReader class for efficient reading and processing of hand bounding box files.

This module provides functionality to:
- Read bounding box coordinates from a directory of files or a single file
- Support frame indexing for flexible bbox data processing
- Process bbox data for both left and right hands
- Filter and predict missing bboxes using Kalman filtering
- Each file contains two lines with [x_min, y_min, x_max, y_max] coordinates
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from filterpy.kalman import KalmanFilter


class BboxReader:
    """A class for reading bounding box files efficiently.

    This class provides methods to read bounding box coordinates from either a directory
    containing individual bbox files or a single file with one bbox per line.
    It supports Kalman filtering for tracking and predicting missing bboxes.

    Attributes:
        bbox_dir_path (Path): Path to directory containing bbox files.
        fmt_frame_fn (Callable[[int], str] | None): Function to transform frame indexes into filenames.
        bbox_len (int): Total number of bbox files in the directory.
        single_file (bool): Whether reading from a single file with one bbox per line.
        use_kalman (bool): Whether to use Kalman filtering for bbox tracking.
        filtered_left (np.ndarray): Preprocessed and filtered bboxes for left hand.
        filtered_right (np.ndarray): Preprocessed and filtered bboxes for right hand.

    """

    def __init__(
        self,
        bbox_path: str | Path,
        fmt_frame_fn: Callable[[int], str] | None = None,
        single_file: bool = False,
        use_kalman: bool = False,
        process_noise: float = 1e-4,
        measurement_noise: float = 3e-1,
    ) -> None:
        """Initialize the BboxReader.

        Args:
            bbox_path (str | Path): Path to directory containing bbox files or single bbox file.
            fmt_frame_fn (Callable[[int], str] | None, optional): Function to transform frame indexes into filenames.
                If None, a default naming convention will be used. Defaults to None.
            single_file (bool, optional): Whether reading from a single file with one bbox per line.
                Defaults to False.
            use_kalman (bool, optional): Whether to use Kalman filtering for bbox tracking.
                Defaults to False.
            process_noise (float, optional): Process noise parameter for Kalman filter.
                Defaults to 1e-4.
            measurement_noise (float, optional): Measurement noise parameter for Kalman filter.
                Defaults to 3e-1.

        """
        self.bbox_path = Path(bbox_path)
        self.fmt_frame_fn = fmt_frame_fn
        self.single_file = single_file
        self.use_kalman = use_kalman
        self.filtered_left = None
        self.filtered_right = None

        # Load all bboxes
        if single_file:
            # Count lines in the file for total number of frames
            with self.bbox_path.open() as f:
                self.bbox_len = sum(1 for _ in f)
            # Load all bboxes at once for single file mode
            bboxes = np.loadtxt(self.bbox_path, dtype=np.float32)
            if len(bboxes.shape) == 1:
                # If only one bbox, reshape to 2D array
                bboxes = bboxes.reshape(1, -1)
            # Split into left and right hands
            self.bboxes_left = bboxes[:, :4]
            self.bboxes_right = bboxes[:, 4:]
        else:
            # Load all bboxes from directory
            self.bbox_len = len(list(self.bbox_path.glob("*.txt")))
            self.bboxes_left = np.zeros((self.bbox_len, 4), dtype=np.float32)
            self.bboxes_right = np.zeros((self.bbox_len, 4), dtype=np.float32)
            try:
                for i in range(self.bbox_len):
                    file_name = self.fmt_frame_fn(i)
                    file_path = self.bbox_path / file_name
                    if file_path.exists():
                        bbox_data = np.loadtxt(file_path, dtype=np.float32)
                        if len(bbox_data.shape) == 1:
                            # Single bbox, split in half for left and right hands
                            mid = len(bbox_data) // 2
                            self.bboxes_left[i] = bbox_data[:mid]
                            self.bboxes_right[i] = bbox_data[mid:]
                        else:
                            self.bboxes_left[i] = bbox_data[0]
                            self.bboxes_right[i] = bbox_data[1]
            except (FileNotFoundError, ValueError, IndexError):
                # Keep zeros for missing or invalid bboxes
                pass

        if use_kalman:
            # Preprocess all bboxes with Kalman filtering
            self._preprocess_with_kalman(process_noise, measurement_noise)

    def _filter_sequence(
        self,
        kf: KalmanFilter,
        bboxes: np.ndarray,
        filtered: np.ndarray,
        forward: bool = True,
        average: bool = False,
    ) -> np.ndarray:
        """Apply Kalman filtering to a sequence of bboxes.

        Args:
            kf (KalmanFilter): Kalman filter instance.
            bboxes (np.ndarray): Input bboxes to filter.
            filtered (np.ndarray): Pre-allocated array for filtered bboxes.
            forward (bool, optional): Whether to filter forward or backward. Defaults to True.
            average (bool, optional): Whether to average with existing filtered values. Defaults to False.

        Returns:
            np.ndarray: Filtered bboxes.

        """
        last_valid = None
        sequence = range(self.bbox_len) if forward else range(self.bbox_len - 1, -1, -1)

        for i in sequence:
            bbox = bboxes[i]
            if not np.all(bbox == 0):
                if last_valid is None:
                    kf.x[:4] = bbox
                    kf.x[4:] = 0
                    filtered[i] = bbox if not average else (filtered[i] + bbox) / 2
                else:
                    kf.predict()
                    kf.update(bbox)
                    filtered[i] = kf.x[:4] if not average else (filtered[i] + kf.x[:4]) / 2
                last_valid = bbox
            elif last_valid is not None:
                kf.predict()
                filtered[i] = kf.x[:4] if not average else (filtered[i] + kf.x[:4]) / 2

        return filtered

    def _preprocess_with_kalman(self, process_noise: float, measurement_noise: float) -> None:
        """Preprocess all bboxes with Kalman filtering.

        This method applies Kalman filtering to all bboxes in both forward and backward directions
        to achieve better smoothing and prediction of missing bboxes.

        Args:
            process_noise (float): Process noise parameter for Kalman filter.
            measurement_noise (float): Measurement noise parameter for Kalman filter.

        """
        # Initialize arrays for filtered bboxes
        filtered_left = np.zeros((self.bbox_len, 4), dtype=np.float32)
        filtered_right = np.zeros((self.bbox_len, 4), dtype=np.float32)

        # Forward pass
        kf_left = self._init_kalman_filter(process_noise, measurement_noise)
        kf_right = self._init_kalman_filter(process_noise, measurement_noise)
        filtered_left = self._filter_sequence(kf_left, self.bboxes_left, filtered_left)
        filtered_right = self._filter_sequence(kf_right, self.bboxes_right, filtered_right)

        # Backward pass to smooth predictions
        kf_left = self._init_kalman_filter(process_noise, measurement_noise)
        kf_right = self._init_kalman_filter(process_noise, measurement_noise)
        filtered_left = self._filter_sequence(kf_left, self.bboxes_left, filtered_left, forward=False, average=True)
        filtered_right = self._filter_sequence(kf_right, self.bboxes_right, filtered_right, forward=False, average=True)

        # Store filtered bboxes
        self.filtered_left = filtered_left
        self.filtered_right = filtered_right

    @staticmethod
    def _init_kalman_filter(process_noise: float, measurement_noise: float) -> KalmanFilter:
        """Initialize a Kalman filter for bbox tracking.

        Args:
            process_noise (float): Process noise parameter.
            measurement_noise (float): Measurement noise parameter.

        Returns:
            KalmanFilter: Initialized Kalman filter.

        """
        kf = KalmanFilter(dim_x=8, dim_z=4)  # State: [x, y, w, h, dx, dy, dw, dh]
        kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + dx
                [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + dy
                [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + dw
                [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + dh
                [0, 0, 0, 0, 1, 0, 0, 0],  # dx = dx
                [0, 0, 0, 0, 0, 1, 0, 0],  # dy = dy
                [0, 0, 0, 0, 0, 0, 1, 0],  # dw = dw
                [0, 0, 0, 0, 0, 0, 0, 1],  # dh = dh
            ],
        )
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
        )
        kf.Q *= process_noise
        kf.R *= measurement_noise
        return kf

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
            IndexError: If the frame index is out of range.

        """
        if frame_idx >= self.bbox_len:
            message = f"Frame index {frame_idx} out of range for bbox file with {self.bbox_len} frames"
            raise IndexError(message)

        if self.use_kalman:
            # Return preprocessed bboxes
            return self.filtered_left[frame_idx], self.filtered_right[frame_idx]
        return self.bboxes_left[frame_idx], self.bboxes_right[frame_idx]

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
            IndexError: If the frame index is out of range.

        """
        return self.get_bbox(idx)
