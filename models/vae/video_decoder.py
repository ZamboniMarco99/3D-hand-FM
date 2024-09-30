"""Video Decoder module for Variational Autoencoder.

This module defines the VideoDecoder class, which is responsible for decoding
latent space representations back into video sequences. It utilizes transposed
3D convolutions to upsample the latent vector into a full video sequence.

Example usage:
    decoder = VideoDecoder(num_frames=16, height=224, width=224, latent_dim=256)
    latent_vector = torch.randn(1, 256)  # (batch, latent_dim)
    reconstructed_video = decoder(latent_vector)
"""

import torch
from torch import nn


class VideoDecoder(nn.Module):
    """Decoder module for the Video VAE.

    This decoder uses transposed 3D convolutions to upsample the latent vector
    into a full video sequence.

    Attributes:
        fc (nn.Linear): Fully connected layer to expand the latent vector.
        conv_transpose_layers (nn.ModuleList): List of transposed 3D convolutional layers.
        final_conv (nn.Conv3d): Final convolutional layer to produce the output video.

    Args:
        latent_dim (int): Dimensionality of the latent space.
        output_channels (int): Number of channels in the output video frames.
        num_frames (int): Number of frames in the output video sequence.
        height (int): Height of each output video frame.
        width (int): Width of each output video frame.

    """

    def __init__(self, num_frames: int, height: int, width: int, latent_dim: int = 256) -> None:
        """Initialize the VideoDecoder module.

        Args:
            num_frames (int): Number of frames in the output video sequence.
            height (int): Height of each output video frame.
            width (int): Width of each output video frame.
            latent_dim (int, optional): Dimensionality of the latent space. Defaults to 256.

        Note:
            The decoder will upsample the latent vector to produce a video sequence
            with dimensions (output_channels, num_frames, height, width).

        """
        super().__init__()

        # Calculate the initial spatial dimensions
        self.init_frame = num_frames // 16
        self.init_height = height // 16
        self.init_width = width // 16

        # Fully connected layer to expand the latent vector
        self.fc = nn.Linear(latent_dim, 512 * self.init_frame * self.init_height * self.init_width)

        # Transposed 3D convolutional layers
        self.conv_transpose_layers = nn.ModuleList(
            [
                nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            ],
        )

        # Final convolutional layer to produce the output video
        self.final_conv = nn.Conv3d(32, 3, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, latent_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, time, height, width).

        """
        # Expand and reshape the input
        x = self.fc(x)
        x = x.view(-1, 512, self.init_frame, self.init_height, self.init_width)

        # Apply transposed convolutions
        for conv_transpose in self.conv_transpose_layers:
            x = torch.relu(conv_transpose(x))

        # Final convolution to get the output video
        return torch.sigmoid(self.final_conv(x))
