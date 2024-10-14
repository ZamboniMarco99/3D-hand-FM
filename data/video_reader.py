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
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_VIDEO_READER_CHUNK_SIZE = 1
DEFAULT_VIDEO_READER_MAX_CHUNK_COUNT = 1


class VideoReader:
    """A class for reading video files and frame directories efficiently.

    This class provides methods to read frames from either a video file or a directory
    containing individual frame images. It supports caching, resizing, and virtual
    frame indexing for flexible video processing.

    Attributes:
        video_path (str): Path to the video file.
        frame_dir_path (str | None): Path to directory containing video frames.
        chunk_cache (dict): Cache for storing chunks of video frames.
        chunk_size (int): Number of frames to load in each chunk.
        max_chunk_count (int): Maximum number of chunks to keep in memory.
        max_width (int | None): Maximum width for resizing frames.
        max_height (int | None): Maximum height for resizing frames.
        assumed_fps (float): Assumed frames per second for virtual frame conversion.
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
        assumed_fps: float = -1,
        chunk_size: int = DEFAULT_VIDEO_READER_CHUNK_SIZE,
        max_chunk_count: int = DEFAULT_VIDEO_READER_MAX_CHUNK_COUNT,
        fmt_frame_fn: Callable[[int], str] | None = None,
        crop: bool = False,
    ) -> None:
        """Initialize the VideoReader.

        Args:
            video_path (str): Path to the video file.
            frame_dir_path (str | None, optional): Path to directory containing video frames. Defaults to None.
            max_width (int | None, optional): Maximum width for resizing frames. Defaults to None.
            max_height (int | None, optional): Maximum height for resizing frames. Defaults to None.
            assumed_fps (float, optional): Assumed frames per second for virtual frame conversion.
                If different from video's actual FPS, it creates a mapping between real and virtual frames.
                Defaults to -1 (use actual video FPS).
            chunk_size (int, optional): Number of frames to load in each chunk.
                Defaults to DEFAULT_VIDEO_READER_CHUNK_SIZE.
            max_chunk_count (int, optional): Maximum number of chunks to keep in memory.
                Defaults to DEFAULT_VIDEO_READER_MAX_CHUNK_COUNT.
            fmt_frame_fn (Callable[[int], str] | None, optional): Function to transform frame indexes into filenames.
                If provided, it should take an integer frame index and return a string filename. Defaults to None.
            crop (bool, optional): If True, crop videos to exact max_width and max_height sizes. Defaults to False.

        Raises:
            ValueError: If video_path is None and frame_dir_path is not provided
             or if no frames are found in frame_dir_path.

        """
        self.video_path = video_path
        self.chunk_cache = {}
        self.frame_dir_path = (
            frame_dir_path if frame_dir_path not in ["", None] and Path(frame_dir_path).is_dir() else None
        )
        self.chunk_size = chunk_size
        self.max_chunk_count = max_chunk_count
        self.max_width = max_width
        self.max_height = max_height
        self.assumed_fps = assumed_fps
        self.fmt_frame_fn = fmt_frame_fn
        self.crop = crop

        if video_path is not None:
            video_cap = cv2.VideoCapture(video_path)
            self.fps = video_cap.get(cv2.CAP_PROP_FPS)
            self.video_len = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            if frame_dir_path in ["", None]:
                msg = "frame_dir_path must be provided if video_path is not provided"
                raise ValueError(msg)
            self.fps = assumed_fps
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

    def get_virtual_frame_count(self) -> int:
        """Calculate the number of virtual frames based on assumed FPS.

        Returns:
            int: The number of virtual frames, which is the frame count if the video
                 were played at the assumed FPS rate.

        """
        num_virtual_frames = int(len(self) / self.fps * self.assumed_fps)
        while int(num_virtual_frames / self.assumed_fps * self.fps) > len(self):
            num_virtual_frames -= 1
        return num_virtual_frames

    def get_real_frame_idx(self, frame_idx: int) -> int:
        """Convert a virtual frame index to a real frame index.

        Args:
            frame_idx (int): The virtual frame index based on assumed FPS.

        Returns:
            int: The corresponding real frame index in the actual video.

        """
        return min(self.video_len - 1, round(frame_idx / self.assumed_fps * self.fps))

    def get_virtual_frame_idx(self, frame_idx: int) -> int:
        """Convert a real frame index to a virtual frame index.

        Args:
            frame_idx (int): The real frame index from the actual video.

        Returns:
            int: The corresponding virtual frame index based on assumed FPS.

        """
        return min(self.get_virtual_frame_count() - 1, round(frame_idx / self.fps * self.assumed_fps))

    def get_frame_and_index(self, frame: int) -> tuple[np.ndarray, int]:
        """Retrieve a single frame from the video and its corresponding real frame index.

        Args:
            frame (int): The index of the frame to retrieve. If assumed_fps is set,
                         this is treated as a virtual frame index.

        Returns:
            tuple[np.ndarray, int]: A tuple containing:
                - The frame as a NumPy array (np.ndarray).
                - The real frame index in the actual video (int).

        Note:
            If assumed_fps is set, the input frame index is treated as a virtual frame index
            and will be converted to a real frame index internally.

        """
        orig_frame = frame
        if self.assumed_fps != -1:
            frame = self.get_real_frame_idx(frame)

        if frame < 0:
            print(f"WARNING: attempt to read frame {frame} < 0; returning frame 0; orig_frame={orig_frame}")
            frame = 0

        if frame >= self.video_len:
            print(
                f"WARNING: attempt to read frame {frame} >= self.video_len={self.video_len}; "
                f"returning frame {self.video_len-1}; orig_frame={orig_frame}",
            )
            frame = self.video_len - 1

        chunk = frame // self.chunk_size
        if chunk not in self.chunk_cache:
            if len(self.chunk_cache) > self.max_chunk_count:
                min_chunk_key = min(self.chunk_cache.items(), key=lambda el: el[1]["last_access"])[0]
                del self.chunk_cache[min_chunk_key]
            chunk_frames = list(range(chunk * self.chunk_size, min((chunk + 1) * self.chunk_size, len(self))))
            self.chunk_cache[chunk] = {"last_access": time.time(), "frames": self.get_frames(chunk_frames)}

        self.chunk_cache[chunk]["last_access"] = time.time()

        ret_frame = self.chunk_cache[chunk]["frames"][frame % self.chunk_size]

        return ret_frame, frame

    def get_frame(self, frame: int) -> tuple[np.ndarray, int]:
        """Retrieve a single frame from the video.

        Args:
            frame (int): The index of the frame to retrieve. If assumed_fps is set,
                         this is treated as a virtual frame index.

        Returns:
            np.ndarray: The frame as a NumPy array.

        """
        ret_frame, _ = self.get_frame_and_index(frame)
        return ret_frame

    def _get_frames_from_frame_dir(self, frame_idxs: list[int]) -> list[np.ndarray]:
        """Retrieve frames from the frame directory.

        Args:
            frame_idxs (list[int]): List of frame indices to read.

        Returns:
            list[np.ndarray]: List of images as NumPy arrays.

        Raises:
            ValueError: If a frame is not found in the directory.

        """
        imgs = []
        for frame_idx in frame_idxs:
            frame_fn = self.fmt_frame_fn(frame_idx)
            img_path = Path(self.frame_dir_path) / frame_fn
            if img_path.is_file():
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
                        img = img.resize((new_width, new_height), Image.Image.Resampling.LANCZOS)

                        # Crop the image to the target dimensions
                        left = (new_width - self.max_width) / 2
                        top = (new_height - self.max_height) / 2
                        right = (new_width + self.max_width) / 2
                        bottom = (new_height + self.max_height) / 2

                        img_pil = img_pil.crop((left, top, right, bottom))

                    else:
                        img_pil.thumbnail((self.max_width or 1e8, self.max_height or 1e8))
                    img = np.array(img_pil)
                imgs.append(img)
            else:
                msg = f"Frame not found: frame_idx={frame_idx} img_path={img_path}"
                raise ValueError(msg)
        return imgs

    def _get_frames_from_video_file(self, frame_idxs: list[int]) -> list[np.ndarray]:
        """Retrieve frames from the video file.

        Args:
            frame_idxs (list[int]): List of frame indices to read.

        Returns:
            list[np.ndarray]: List of images as NumPy arrays.

        """
        missing_img_idxs = {}
        imgs = []

        video_cap = cv2.VideoCapture(str(self.video_path))
        delta = frame_idxs[1] - frame_idxs[0] if len(frame_idxs) > 1 else 1
        video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idxs[0])
        video_cap.setExceptionMode(True)

        num_read = 0

        current_frame_idx = frame_idxs[0]
        while num_read <= delta * (len(frame_idxs) - 1):
            success, img = video_cap.read()
            if not success:
                print(f"Error reading from video reader: self.video_path={self.video_path}")
                continue
            if num_read % delta == 0:
                img = img[:, :, ::-1]
                if self.max_width is not None or self.max_height is not None:
                    img_pil = Image.fromarray(img)
                    img_pil.thumbnail((self.max_width or 1e8, self.max_height or 1e8))
                    img = np.array(img_pil)

                if current_frame_idx in missing_img_idxs:
                    imgs[missing_img_idxs[current_frame_idx]] = img
                    del missing_img_idxs[current_frame_idx]
                else:
                    imgs.append(img)
            current_frame_idx += 1
            num_read += 1

        video_cap.release()
        return imgs

    def _get_frames_from_frame_dir_with_video_fallback(self, frame_idxs: list[int]) -> list[np.ndarray]:
        """Retrieve frames from the frame directory with a fallback to the video file for missing frames.

        Args:
            frame_idxs (list[int]): List of frame indices to read.

        Returns:
            list[np.ndarray]: List of images as NumPy arrays.

        """
        imgs = []
        missing_idxs = []
        for frame_idx in frame_idxs:
            frame_fn = self.fmt_frame_fn(frame_idx)
            img_path = Path(self.frame_dir_path) / frame_fn
            if img_path.is_file():
                img = cv2.imread(str(img_path))
                img = img[:, :, ::-1]
                if self.max_width is not None or self.max_height is not None:
                    img_pil = Image.fromarray(img)
                    img_pil.thumbnail((self.max_width or 1e8, self.max_height or 1e8))
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
                        img = img.resize((new_width, new_height), Image.Image.Resampling.LANCZOS)

                        # Crop the image to the target dimensions
                        left = (new_width - self.max_width) / 2
                        top = (new_height - self.max_height) / 2
                        right = (new_width + self.max_width) / 2
                        bottom = (new_height + self.max_height) / 2

                        img_pil = img_pil.crop((left, top, right, bottom))

                    else:
                        img_pil.thumbnail((self.max_width or 1e8, self.max_height or 1e8))
                    img = np.array(img_pil)
                imgs.append(img)
            else:
                imgs.append(None)
                missing_idxs.append(frame_idx)
        fallback_frames = self.get_frames_from_video_file(missing_idxs)
        for idx, frame in zip(missing_idxs, fallback_frames, strict=False):
            imgs[idx] = frame
        return imgs

    def get_frames(self, frame_idxs: list[int]) -> list[np.ndarray]:
        """Retrieve multiple frames from the video.

        Args:
            frame_idxs (list[int]): List of frame indices to read. These are real frame indices,
                                    not virtual frame indices.

        Returns:
            list[np.ndarray]: List of images as NumPy arrays.

        """
        frames = []
        if self.frame_dir_path not in ["", None]:
            try:
                frames = self._get_frames_from_frame_dir(frame_idxs)
            except ValueError as e:
                print(e)
                frames = self._get_frames_from_frame_dir_with_video_fallback(frame_idxs)
        else:
            frames = self._get_frames_from_video_file(frame_idxs)
        return frames

    def __getitem__(self, idx: int) -> np.ndarray:
        """Retrieve a single frame from the video.

        This method allows the VideoReader to be used as an iterable,
        returning frames based on the given index.

        Args:
            idx (int): The index of the frame to retrieve. If assumed_fps is set,
                       this is treated as a virtual frame index.

        Returns:
            np.ndarray: The frame as a NumPy array with shape (height, width, 3).

        """
        return self.get_frame(idx)
