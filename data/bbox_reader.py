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
        bbox_path (Path): Path to directory containing bbox files or single bbox file.
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
        min_bbox_diagonal: float = 50,
    ) -> None:
        """Initialize the BboxReader."""
        self.bbox_path = Path(bbox_path)
        self.fmt_frame_fn = fmt_frame_fn
        self.single_file = single_file
        self.use_kalman = use_kalman
        self.min_bbox_diagonal = min_bbox_diagonal
        self.measurement_noise = measurement_noise
        self.filtered_left = None
        self.filtered_right = None
        self.bboxes_left = None
        self.bboxes_right = None
        self.bbox_len = 0

        self._load_bboxes()

        if use_kalman:
            self._preprocess_with_kalman(process_noise, measurement_noise)

    def _load_bboxes(self) -> None:
        """Load bounding boxes from file(s)."""
        if self.single_file:
            self._load_from_single_file()
        else:
            self._load_from_directory()

    def _load_from_single_file(self) -> None:
        """Load bounding boxes from a single file."""
        with self.bbox_path.open() as f:
            self.bbox_len = sum(1 for _ in f)

        bboxes = np.loadtxt(self.bbox_path, dtype=np.float32)
        if len(bboxes.shape) == 1:
            bboxes = bboxes.reshape(1, -1)

        self.bboxes_left = bboxes[:, :4]
        self.bboxes_right = bboxes[:, 4:]

    def _load_from_directory(self) -> None:
        """Load bounding boxes from a directory of files."""
        self.bbox_len = len(list(self.bbox_path.glob("*.txt")))
        self.bboxes_left = np.zeros((self.bbox_len, 4), dtype=np.float32)
        self.bboxes_right = np.zeros((self.bbox_len, 4), dtype=np.float32)

        try:
            for i in range(self.bbox_len):
                file_path = self.bbox_path / self.fmt_frame_fn(i)
                if file_path.exists():
                    bbox_data = np.loadtxt(file_path, dtype=np.float32)
                    if len(bbox_data.shape) == 1:
                        mid = len(bbox_data) // 2
                        self.bboxes_left[i] = bbox_data[:mid]
                        self.bboxes_right[i] = bbox_data[mid:]
                    else:
                        self.bboxes_left[i] = bbox_data[0]
                        self.bboxes_right[i] = bbox_data[1]
        except (FileNotFoundError, ValueError, IndexError):
            pass  # Keep zeros for missing or invalid bboxes

    def _find_valid_measurements(self, bboxes: np.ndarray) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
        """Find valid measurements in the sequence of bboxes."""
        valid_frames = []
        valid_centers = []
        valid_sizes = []

        for i in range(self.bbox_len):
            bbox = bboxes[i]
            if not np.all(bbox == 0):
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                diagonal = np.sqrt(width**2 + height**2)

                if diagonal >= self.min_bbox_diagonal:
                    valid_frames.append(i)
                    center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
                    size = np.array([width, height])
                    valid_centers.append(center)
                    valid_sizes.append(size)

        return valid_frames, valid_centers, valid_sizes

    def _interpolate_sequence(
        self,
        valid_frames: list[int],
        valid_centers: list[np.ndarray],
        valid_sizes: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Interpolate missing frames between valid measurements."""
        interpolated = np.zeros((self.bbox_len, 4), dtype=np.float32)
        is_interpolated = np.zeros(self.bbox_len, dtype=bool)

        if not valid_frames:
            return interpolated, is_interpolated

        # Process pairs of valid measurements
        for i in range(len(valid_frames) - 1):
            start_idx, end_idx = valid_frames[i], valid_frames[i + 1]
            start_center, end_center = valid_centers[i], valid_centers[i + 1]
            start_size, end_size = valid_sizes[i], valid_sizes[i + 1]

            # Set start frame
            interpolated[start_idx] = self._center_to_corners(start_center, start_size)

            # Interpolate intermediate frames
            for j in range(start_idx + 1, end_idx):
                t = (j - start_idx) / (end_idx - start_idx)
                center = start_center * (1 - t) + end_center * t
                size = start_size * (1 - t) + end_size * t
                interpolated[j] = self._center_to_corners(center, size)
                is_interpolated[j] = True

        # Set last valid frame
        if valid_frames:
            last_idx = valid_frames[-1]
            interpolated[last_idx] = self._center_to_corners(valid_centers[-1], valid_sizes[-1])

        return interpolated, is_interpolated

    @staticmethod
    def _center_to_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
        """Convert center and size to corner format."""
        half_width, half_height = size[0] / 2, size[1] / 2
        return np.array(
            [
                center[0] - half_width,  # x1
                center[1] - half_height,  # y1
                center[0] + half_width,  # x2
                center[1] + half_height,  # y2
            ],
        )

    def _apply_kalman_filter(
        self,
        kf: KalmanFilter,
        interpolated: np.ndarray,
        is_interpolated: np.ndarray,
        forward: bool,
    ) -> np.ndarray:
        """Apply Kalman filtering to the interpolated sequence."""
        filtered = np.zeros_like(interpolated)
        sequence = range(self.bbox_len) if forward else range(self.bbox_len - 1, -1, -1)
        last_measurement = None

        for i in sequence:
            bbox = interpolated[i]
            if np.all(bbox == 0):
                continue

            # Convert to center format
            cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            measurement = np.array([cx, cy, w, h])

            if last_measurement is None:
                kf.x[:4, 0] = measurement
                kf.x[4:, 0] = 0  # Zero initial velocity
                filtered[i] = bbox
            else:
                # Higher noise for interpolated values and size measurements
                noise = self.measurement_noise * (10.0 if is_interpolated[i] else 1.0)
                kf.R = np.eye(4) * noise
                kf.R[2:4, 2:4] *= 2.0  # Even higher noise for size measurements

                kf.predict()
                kf.update(measurement)
                filtered[i] = self._center_to_corners(kf.x[:2, 0], kf.x[2:4, 0])

            last_measurement = measurement

        return filtered

    def _filter_sequence(
        self,
        kf: KalmanFilter,
        bboxes: np.ndarray,
        filtered: np.ndarray,
        forward: bool = True,
    ) -> np.ndarray:
        """Apply Kalman filtering to a sequence of bboxes."""
        valid_frames, valid_centers, valid_sizes = self._find_valid_measurements(bboxes)
        if not valid_frames:
            return filtered

        interpolated, is_interpolated = self._interpolate_sequence(valid_frames, valid_centers, valid_sizes)
        return self._apply_kalman_filter(kf, interpolated, is_interpolated, forward)

    def _preprocess_with_kalman(self, process_noise: float, measurement_noise: float) -> None:
        """Preprocess all bboxes with Kalman filtering."""
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
        filtered_left = self._filter_sequence(kf_left, self.bboxes_left, filtered_left, forward=False)
        filtered_right = self._filter_sequence(kf_right, self.bboxes_right, filtered_right, forward=False)

        self.filtered_left = filtered_left
        self.filtered_right = filtered_right

    @staticmethod
    def _init_kalman_filter(process_noise: float, measurement_noise: float) -> KalmanFilter:
        """Initialize a Kalman filter for bbox tracking."""
        kf = KalmanFilter(dim_x=8, dim_z=4)  # State: [x, y, w, h, dx, dy, dw, dh]

        # State transition matrix
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

        # Measurement matrix
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
        )

        # Process noise matrix
        kf.Q = np.eye(8) * process_noise
        kf.Q[2:4, 2:4] *= 0.1  # Lower noise for w, h
        kf.Q[6:8, 6:8] *= 0.1  # Lower noise for dw, dh

        # Measurement noise matrix
        kf.R = np.eye(4) * measurement_noise
        kf.R[2:4, 2:4] *= 2.0  # Higher noise for width and height

        return kf

    def get_bbox(self, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get bounding box coordinates for a single frame."""
        if frame_idx >= self.bbox_len:
            msg = f"Frame index {frame_idx} out of range for bbox file with {self.bbox_len} frames"
            raise IndexError(msg)

        if self.use_kalman:
            return self.filtered_left[frame_idx], self.filtered_right[frame_idx]
        return self.bboxes_left[frame_idx], self.bboxes_right[frame_idx]

    def get_bbox_sequence(self, frame_idxs: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Get bounding box coordinates for a sequence of frames."""
        left_hands = []
        right_hands = []
        for frame_idx in frame_idxs:
            left, right = self.get_bbox(frame_idx)
            left_hands.append(left)
            right_hands.append(right)
        return left_hands, right_hands

    def __len__(self) -> int:
        """Get the total number of frames."""
        return self.bbox_len

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get bounding box coordinates for a frame by index."""
        return self.get_bbox(idx)
