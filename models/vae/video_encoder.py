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
from torchvision.models.video.mvit import MSBlockConfig, MViT, _mvit


def get_mvit_v2_s_block_setting() -> list[MSBlockConfig]:
    """Get the block setting for MViT v2 Small architecture.

    Returns:
        list[MSBlockConfig]: A list of MSBlockConfig objects, each representing a block's configuration.

    """
    config: dict[str, list] = {
        "num_heads": [1, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8],
        "input_channels": [96, 96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768],
        "output_channels": [96, 192, 192, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 384, 768, 768],
        "kernel_q": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "kernel_kv": [
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        "stride_q": [
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 2, 2],
            [1, 1, 1],
        ],
        "stride_kv": [
            [1, 8, 8],
            [1, 4, 4],
            [1, 4, 4],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 2, 2],
            [1, 1, 1],
            [1, 1, 1],
        ],
    }

    return [
        MSBlockConfig(
            num_heads=config["num_heads"][i],
            input_channels=config["input_channels"][i],
            output_channels=config["output_channels"][i],
            kernel_q=config["kernel_q"][i],
            kernel_kv=config["kernel_kv"][i],
            stride_q=config["stride_q"][i],
            stride_kv=config["stride_kv"][i],
        )
        for i in range(len(config["num_heads"]))
    ]


def mvit_v2_s(
    *,
    weights: None = None,
    progress: bool = True,
    spatial_size: tuple[int, int] = (224, 224),
    temporal_size: int = 16,
    **kwargs: dict,
) -> MViT:
    """Constructs a small MViTV2 architecture.

    Architecture based on `Multiscale Vision Transformers <https://arxiv.org/abs/2104.11227>`__ and
    `MViTv2: Improved Multiscale Vision Transformers for Classification
    and Detection <https://arxiv.org/abs/2112.01526>`__.

    .. betastatus:: video module

    Args:
        weights (:class:`~torchvision.models.video.MViT_V2_S_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.MViT_V2_S_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        spatial_size (tuple, optional): A tuple of two integers representing the
            height and width of the input frames. Default is (224, 224).
        temporal_size (int, optional): An integer representing the number of frames
            in the input video sequence. Default is 16.
        **kwargs: Additional parameters passed to the ``torchvision.models.video.MViT``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/mvit.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.MViT_V2_S_Weights
            :members:

    """
    block_setting = get_mvit_v2_s_block_setting()
    return _mvit(
        spatial_size=spatial_size,
        temporal_size=temporal_size,
        block_setting=block_setting,
        residual_pool=True,
        residual_with_cls_embed=False,
        rel_pos_embed=True,
        proj_after_attn=True,
        stochastic_depth_prob=kwargs.pop("stochastic_depth_prob", 0.2),
        weights=weights,
        progress=progress,
        **kwargs,
    )


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
            spatial_size=(height, width),
            temporal_size=num_frames,
        )

        backbone_out_features = get_mvit_v2_s_block_setting()[-1].output_channels

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
