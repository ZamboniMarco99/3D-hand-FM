"""Arctic dataset package.

This package provides classes for working with the Arctic dataset, which consists of
multiple video sequences and corresponding MANO hand pose parameters.

The main classes are:
- ArcticDataset: Base dataset class that returns video clips and corresponding parameters
- ArcticDataModule: PyTorch Lightning DataModule for handling data loading and preparation
"""

from data.arctic.datamodule import ArcticDataModule
from data.arctic.dataset import ArcticDataset

__all__ = ["ArcticDataset", "ArcticDataModule"]
