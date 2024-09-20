"""Dummy model implementation for video classification.

This module contains a simple video classifier using a 3D CNN architecture.
The DummyVideoClassifier class is designed to classify video sequences into
a specified number of classes using PyTorch Lightning.

Example usage:
    model = DummyVideoClassifier(num_classes=2, learning_rate=1e-3)
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader, val_dataloader)
"""

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam
from torchmetrics import Accuracy


class DummyVideoClassifier(pl.LightningModule):
    """A simple video classifier using a 3D CNN architecture.

    This model is designed to classify video sequences into a specified number of classes.
    It uses a series of 3D convolutional layers followed by a linear classifier.

    Attributes:
        features (nn.Sequential): The feature extraction layers of the CNN.
        classifier (nn.Linear): The final classification layer.
        train_accuracy (Accuracy): Metric to track training accuracy over the epoch.
        val_accuracy (Accuracy): Metric to track validation accuracy over the epoch.

    Args:
        num_classes (int): The number of classes to classify. Defaults to 2.
        learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.

    """

    def __init__(
        self,
        num_classes: int = 2,
        learning_rate: float = 1e-3,  # noqa: ARG002
        num_frames: int = 30,
        height: int = 480,
        width: int = 640,
    ) -> None:
        """Initialize the DummyVideoClassifier.

        This method sets up the model architecture, including the feature extraction
        layers and the final classification layer. It also saves the hyperparameters
        for later use and initializes accuracy metrics.

        Args:
            num_classes (int): The number of classes to classify. Defaults to 2.
            learning_rate (float): The learning rate for the optimizer. Defaults to 1e-3.
            num_frames (int): The number of frames in each video input. Defaults to 30.
            height (int): The height of each frame. Defaults to 480.
            width (int): The width of each frame. Defaults to 640.

        """
        super().__init__()
        self.save_hyperparameters()

        self.num_frames = num_frames
        self.height = height
        self.width = width

        # Simple CNN for video classification
        self.features = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Calculate the size of the flattened features directly
        c_out = 64  # number of channels in the last Conv3d layer
        t_out = num_frames // 2  # time dimension after 1 MaxPool3d operations
        h_out = height // 2  # height after 1 MaxPool3d operations
        w_out = width // 2  # width after 1 MaxPool3d operations
        flattened_size = c_out * t_out * h_out * w_out

        self.classifier = nn.Linear(flattened_size, num_classes)

        # Initialize accuracy metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).

        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Perform a single training step.

        Args:
            batch (tuple): A tuple containing the input tensors and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for this step.

        """
        x, y = batch
        # Permute the input tensor to match the expected shape for 3D convolutions
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.squeeze())
        self.train_accuracy(y_hat, y.squeeze())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        """Perform a single validation step.

        Args:
            batch (tuple): A tuple containing the input tensors and labels.
            batch_idx (int): The index of the current batch.

        """
        x, y = batch
        # Permute the input tensor to match the expected shape for 3D convolutions
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y.squeeze())
        self.val_accuracy(y_hat, y.squeeze())
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_accuracy, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: The configured optimizer.

        """
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
