"""Video Encoder module for Variational Autoencoder.

This module defines the Encoder class, which is responsible for encoding input video sequences
into a latent space representation. It utilizes the MViT v2 Small model as the backbone
and adapts it for use in a Variational Autoencoder (VAE) architecture.

Example usage:
    encoder = VideoEncoder(num_frames=16, height=224, width=224, latent_dim=256)
    video_input = torch.randn(1, 3, 16, 224, 224)  # (batch, channels, frames, height, width)
    mu, log_var = encoder(video_input)
"""

import torch
from torch import nn
from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s


class VideoEncoder(nn.Module):
    """Encoder module for the Video VAE using MViT v2 Small architecture.

    This encoder uses a pre-trained MViT v2 Small model as the backbone and
    adapts it for encoding video sequences into a latent space representation.

    Attributes:
        backbone (nn.Module): The MViT v2 Small backbone.
        fc_mu (nn.Linear): Fully connected layer for mean of latent space.
        fc_var (nn.Linear): Fully connected layer for log variance of latent space.

    Args:
        latent_dim (int): Dimensionality of the latent space.

    """

    def __init__(self, num_frames: int, height: int, width: int, latent_dim: int = 256) -> None:
        """Initialize the Encoder module.

        Args:
            num_frames (int): Number of frames in each video sequence.
            height (int): Height of each video frame.
            width (int): Width of each video frame.
            latent_dim (int, optional): Dimensionality of the latent space. Defaults to 256.

        Note:
            The input dimensions (input_channels, num_frames, height, width) should match
            the expected input shape of the MViT v2 Small model.

        """
        super().__init__()

        # Load pre-trained MViT v2 Small model
        self.backbone = mvit_v2_s(
            weights=MViT_V2_S_Weights.DEFAULT,
            spatial_size=(height, width),
            temporal_size=num_frames,
        )

        backbone_out_features = self.backbone.head.in_features

        # Remove the classification head
        self.backbone.head = nn.Identity()

        # Add fully connected layers for mean and log variance
        self.fc_mu = nn.Linear(backbone_out_features, latent_dim)
        self.fc_var = nn.Linear(backbone_out_features, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - mu: Mean of the latent Gaussian.
                - log_var: Log variance of the latent Gaussian.

        """
        # Ensure input is in the correct format (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W) -> (B, C, T, H, W)

        # Pass through the backbone
        features = self.backbone(x)

        # Compute mean and log variance
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)

        return mu, log_var
