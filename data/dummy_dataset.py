"""Dummy dataset implementation for testing purposes.

The DummyDataset class serves as a starting point for creating custom datasets.
It simulates a video dataset with configurable time, height, width, and frame rate.
The dataset returns dummy tensors representing video frames and labels.

Key features:
- Configurable video dimensions and frame rate
- Fixed dataset size of 100 samples
- Returns dummy tensors for video frames and labels
- Labels indicate whether half of the video frames are black (0) or white (1)

Example usage:
    dataset = DummyDataset(time=10, height=480, width=640, frame_rate=30)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
"""

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """A dummy dataset class for demonstration purposes.

    This class serves as a template for creating custom PyTorch datasets.
    It implements the basic structure required for a PyTorch Dataset,
    including initialization, length determination, and item retrieval.

    Note: This implementation returns empty data and should be modified
    for actual use in projects.
    """

    def __init__(self, time: int = 1, height: int = 640, width: int = 480, frame_rate: int = 30) -> None:
        """Initialize the DummyDataset.

        This method should be used to set up any necessary data structures
        or load data required for the dataset.
        """
        self.time = time
        self.height = height
        self.width = width
        self.frame_rate = frame_rate

    def __len__(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        """
        # Return the total number of samples in the dataset
        return 100

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve a sample from the dataset at the given index.

        This method generates a dummy video tensor and a corresponding class label.
        The video tensor represents a sequence of frames, where each frame is either
        randomly generated or set to black/white based on the index.

        For even indices:
        - Approximately half of the frames are set to black (pixel values of 0)
        - The class label is set to 1

        For odd indices:
        - Approximately half of the frames are set to white (pixel values of 255)
        - The class label is set to 0

        The remaining frames in both cases are filled with random values between 0 and 1.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing two torch tensors.
                The first tensor represents a video with shape [T, C, H, W], where:
                T is the number of frames (time steps),
                C is the number of channels,
                H is the height of each frame,
                W is the width of each frame.
                The second tensor's shape is not specified.

        """
        dummy_video = torch.rand(self.time * self.frame_rate, 3, self.height, self.width, dtype=torch.float)

        if idx % 2 == 0:
            # Set half of the frames to black randomly
            black_frames = torch.randint(0, 2, (self.time * self.frame_rate,), dtype=torch.bool)
            dummy_video[black_frames] = 0.0
            dummy_class = torch.ones(1, dtype=torch.long)
        else:
            # Set half of the frames to white randomly
            white_frames = torch.randint(0, 2, (self.time * self.frame_rate,), dtype=torch.bool)
            dummy_video[white_frames] = 1.0
            dummy_class = torch.zeros(1, dtype=torch.long)

        return dummy_video, dummy_class
