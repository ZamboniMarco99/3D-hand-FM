"""H2O dataset package.

This package provides classes for working with the H2O dataset, which consists of
multiple video sequences and corresponding MANO hand pose parameters.

The main classes are:
- H2ODataset: Base dataset class that returns video clips and corresponding parameters
- H2OLastFrameDataset: Dataset that returns video clips and parameters of their last frames
- H2ODataModule: PyTorch Lightning DataModule for handling data loading and preparation
"""

from data.h2o.datamodule import H2ODataModule
from data.h2o.dataset import H2ODataset
from data.h2o.last_frame_dataset import H2OLastFrameDataset

__all__ = ["H2ODataset", "H2OLastFrameDataset", "H2ODataModule"]
