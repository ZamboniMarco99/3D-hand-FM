"""VideoReader class for efficient reading and processing of video files and frame directories.

Based on https://gist.github.com/algvr/53781f020b3f7744fdacab2738cf21ba by Alexey Gavryushin

The VideoReader class offers functionality to:
- Read frames from video files or directories containing individual frame images
- Support caching for improved performance
- Resize frames to specified dimensions
- Handle virtual frame indexing for flexible video processing
- Convert between real and virtual frame indices based on assumed frame rates
"""

import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class VideoReader:
    """A class for reading video files and frame directories efficiently.

    This class provides methods to read frames from either a video file or a directory
    containing individual frame images. It supports caching, resizing, and virtual
    frame indexing for flexible video processing.

    Attributes:
        video_path (str): Path to the video file.
        frame_dir_path (str | None): Path to directory containing video frames.
        max_width (int | None): Maximum width for resizing frames.
        max_height (int | None): Maximum height for resizing frames.
        fmt_frame_fn (Callable[[int], str] | None): Function to transform frame indexes into filenames.
        fps (float): Actual frames per second of the video.
        video_len (int): Total number of frames in the video.
        video_width (int): Width of the video frames.
        video_height (int): Height of the video frames.

    """

    def __init__(
        self,
        video_path: str,
        frame_dir_path: str | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
        fmt_frame_fn: Callable[[int], str] | None = None,
        crop: bool = False,
    ) -> None:
        """Initialize the VideoReader.

        Args:
            video_path (str): Path to the video file.
            frame_dir_path (str | None, optional): Path to directory containing video frames. Defaults to None.
            max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
            max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
            fmt_frame_fn (Callable[[int], str] | None, optional): Function to transform frame indexes into filenames.
                If provided, it should take an integer frame index and return a string filename. Defaults to None.
            crop (bool, optional): If True, crop videos to exact max_width and max_height sizes. Defaults to False.

        Raises:
            ValueError: If video_path is None and frame_dir_path is not provided
             or if no frames are found in frame_dir_path.

        """
        self.video_path = video_path
        self.frame_dir_path = (
            frame_dir_path if frame_dir_path not in ["", None] and Path(frame_dir_path).is_dir() else None
        )
        self.max_width = max_width
        self.max_height = max_height
        self.fmt_frame_fn = fmt_frame_fn
        self.crop = crop

        if video_path is not None:
            video_cap = cv2.VideoCapture(video_path)
            self.video_len = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            if frame_dir_path in ["", None]:
                msg = "frame_dir_path must be provided if video_path is not provided"
                raise ValueError(msg)
            fns = [fn for fn in os.listdir(frame_dir_path) if fn.split(".")[-1].lower() in ["jpg", "jpeg", "png"]]
            if len(fns) == 0:
                msg = "No frames found in frame_dir_path"
                raise ValueError(msg)
            self.video_len = len(fns)
            path_0 = Path(frame_dir_path) / fns[0]
            with Image.open(path_0) as img:
                self.video_width = img.width
                self.video_height = img.height

    def __len__(self) -> int:
        """Get the total number of frames in the video.

        Returns:
            int: The total number of actual frames in the video.

        """
        return self.video_len

    def get_frames(self, frame_idxs: list[int]) -> list[np.ndarray]:
        """Retrieve multiple frames from the video.

        Args:
            frame_idxs (list[int]): List of frame indices to read.

        Returns:
            list[np.ndarray]: List of images as NumPy arrays.

        """
        frames = []
        for frame_idx in frame_idxs:
            frame_fn = self.fmt_frame_fn(frame_idx)
            img_path = Path(self.frame_dir_path) / frame_fn
            img = cv2.imread(str(img_path))
            img = img[:, :, ::-1]
            if self.max_width is not None or self.max_height is not None:
                img_pil = Image.fromarray(img)
                if self.crop:
                    # Calculate the aspect ratios
                    aspect_ratio_img = img_pil.width / img_pil.height
                    aspect_ratio_target = self.max_width / self.max_height

                    # Resize based on the target aspect ratio
                    if aspect_ratio_img > aspect_ratio_target:
                        # Image is wider than the target aspect ratio
                        new_height = self.max_height
                        new_width = int(new_height * aspect_ratio_img)
                    else:
                        # Image is taller than the target aspect ratio
                        new_width = self.max_width
                        new_height = int(new_width / aspect_ratio_img)

                    # Resize the image
                    img_pil = img_pil.resize((new_width, new_height))

                    # Crop the image to the target dimensions
                    left = (new_width - self.max_width) / 2
                    top = (new_height - self.max_height) / 2
                    right = (new_width + self.max_width) / 2
                    bottom = (new_height + self.max_height) / 2

                    img_pil = img_pil.crop((left, top, right, bottom))
                else:
                    img_pil.thumbnail((self.max_width or 1e8, self.max_height or 1e8))
                img = np.array(img_pil)
            frames.append(img)
        return frames

    def get_frame(self, frame: int) -> np.ndarray:
        """Retrieve a single frame from the video.

        Args:
            frame (int): The index of the frame to retrieve.

        Returns:
            np.ndarray: The frame as a NumPy array.

        """
        return self.get_frames([frame])[0]

    def __getitem__(self, idx: int) -> np.ndarray:
        """Retrieve a single frame from the video.

        This method allows the VideoReader to be used as an iterable,
        returning frames based on the given index.

        Args:
            idx (int): The index of the frame to retrieve.

        Returns:
            np.ndarray: The frame as a NumPy array with shape (height, width, 3).

        """
        return self.get_frame(idx)
