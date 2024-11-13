from functools import partial

import torch
from torch import nn
from torchvision.models.video.mvit import *
from typing import Tuple, Sequence, Optional, Callable

from torchvision.models.video.mvit import MultiscaleBlock, PositionalEncoding
from torchvision.utils import _log_api_usage_once


class Dinov2Backbone(nn.Module):
    """
    DINOv2 Backbone for Feature Extraction.

    This module leverages the DINOv2 model to extract high-dimensional features from each frame of a video.
    Each frame is processed independently, allowing for efficient parallel feature extraction across temporal frames.

    Preprocessing:
        The input video frames should be resized to 224x224 pixels and normalized using the following mean and standard deviation:
            - mean = [0.485, 0.456, 0.406]
            - std = [0.229, 0.224, 0.225]

    Attributes:
        name (str): Name of the DINOv2 model variant to load.
        encoder (nn.Module): The DINOv2 model used for feature extraction.
        patch_size (int): The size of the patches used by the DINOv2 model.
        embed_dim (int): The dimensionality of the embeddings produced by the DINOv2 model.
    """

    def __init__(self, name: str = "dinov2_vitb14", pretrained: bool = False, *args, **kwargs):
        """
        Initializes the Dinov2Backbone module.

        Args:
            name (str, optional): The specific DINOv2 model variant to load. Defaults to "dinov2_vitb14".
            pretrained (bool, optional): If True, loads pretrained weights. Defaults to False.
            *args: Variable length argument list for additional parameters.
            **kwargs: Arbitrary keyword arguments for additional parameters.
        """
        super().__init__()
        self.name = name
        self.encoder = torch.hub.load("facebookresearch/dinov2", self.name, pretrained=pretrained)
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass to extract DINOv2 features from input video frames.

        Args:
            x (torch.Tensor): Input video frames of shape [B, T, C, H, W].
                - B (int): Batch size.
                - T (int): Temporal dimension (number of frames).
                - C (int): Number of channels.
                - H (int): Height of the frames.
                - W (int): Width of the frames.

        Returns:
            torch.Tensor: Extracted features of shape [B, T, N, D].
                - B (int): Batch size.
                - T (int): Temporal dimension (number of frames).
                - N (int): Number of spatial tokens per frame.
                - D (int): Embedding dimension.
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        y = self.encoder.get_intermediate_layers(x)[0]  # Output shape: [B*T, N, D]
        y = y.reshape(B, T, -1, self.embed_dim)
        return y


class DinoMVit(nn.Module):
    """
    DINO-MViT Model for Video Classification.

    This model integrates a DINOv2 backbone for feature extraction with a Multiscale Vision Transformer (MViT)
    to process spatio-temporal features extracted from video frames. It is designed for video classification tasks,
    leveraging both spatial and temporal information.

    Args:
        spatial_size (Tuple[int, int]): Spatial dimensions of the input frames as (Height, Width).
        temporal_size (int): Number of frames in the input video sequence.
        block_setting (Sequence[MSBlockConfig]): List of MSBlockConfig objects defining the MViT architecture.
        residual_pool (bool): Whether to use residual pooling in the MViT blocks.
        residual_with_cls_embed (bool): Whether to include the class token embedding in residual connections.
        rel_pos_embed (bool): Whether to use relative positional embeddings.
        proj_after_attn (bool): Whether to apply projection after attention.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
        attention_dropout (float, optional): Attention dropout probability. Defaults to 0.0.
        stochastic_depth_prob (float, optional): Stochastic depth probability. Defaults to 0.0.
        num_classes (int, optional): Number of output classes for classification. Defaults to 400.
        block (Optional[Callable[..., nn.Module]], optional): Block module to use in MViT (e.g., MultiscaleBlock). Defaults to None.
        norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use (e.g., nn.LayerNorm). Defaults to None.

    Attributes:
        dino_backbone (Dinov2Backbone): The DINOv2 model used for feature extraction.
        feature_proj (nn.Module): Projection layer to map DINOv2 features to the MViT embedding dimension.
        pos_encoding (PositionalEncoding): Positional encoding module for spatio-temporal features.
        blocks (nn.ModuleList): List of MViT blocks forming the transformer encoder.
        norm (nn.Module): Normalization layer applied after the transformer encoder.
        head (nn.Sequential): Classification head producing logits for each class.
    """

    def __init__(
        self,
        spatial_size: Tuple[int, int],  # Spatial size (H, W)
        temporal_size: int,  # Number of frames T
        block_setting: Sequence["MSBlockConfig"],  # Define MSBlockConfig appropriately
        residual_pool: bool,
        residual_with_cls_embed: bool,
        rel_pos_embed: bool,
        proj_after_attn: bool,
        dropout: float = 0.5,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        num_classes: int = 400,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Initializes the DinoMVit model.

        Args:
            spatial_size (Tuple[int, int]): Spatial dimensions of the input frames as (Height, Width).
            temporal_size (int): Number of frames in the input video sequence.
            block_setting (Sequence["MSBlockConfig"]): List of MSBlockConfig objects defining the MViT architecture.
            residual_pool (bool): Whether to use residual pooling in the MViT blocks.
            residual_with_cls_embed (bool): Whether to include the class token embedding in residual connections.
            rel_pos_embed (bool): Whether to use relative positional embeddings.
            proj_after_attn (bool): Whether to apply projection after attention.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
            attention_dropout (float, optional): Attention dropout probability. Defaults to 0.0.
            stochastic_depth_prob (float, optional): Stochastic depth probability. Defaults to 0.0.
            num_classes (int, optional): Number of output classes for classification. Defaults to 400.
            block (Optional[Callable[..., nn.Module]], optional): Block module to use in MViT (e.g., MultiscaleBlock). Defaults to None.
            norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use (e.g., nn.LayerNorm). Defaults to None.

        Raises:
            ValueError: If the block_setting sequence is empty.
        """
        super().__init__()
        _log_api_usage_once(self)
        # The first 4 block settings are being skipped, because they have only a 96 dimensional embedding.
        block_setting = block_setting[4:]
        embed_dim = block_setting[0].input_channels
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Replace the MVit Patch Embedding module with DINOv2 Backbone
        self.dino_backbone = Dinov2Backbone()
        feature_dim = self.dino_backbone.embed_dim

        # Feature Projection Layer
        if feature_dim != embed_dim:
            self.feature_proj = nn.Linear(feature_dim, embed_dim)
        else:
            self.feature_proj = nn.Identity()

        # Calculate number of patches per frame
        h_patches = spatial_size[0] // self.dino_backbone.patch_size
        w_patches = spatial_size[1] // self.dino_backbone.patch_size

        # Spatio-Temporal Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=embed_dim,
            spatial_size=(h_patches, w_patches),
            temporal_size=temporal_size,
            rel_pos_embed=rel_pos_embed,
        )

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList()
        input_size = [temporal_size, h_patches * w_patches]  # [T, N]

        for stage_block_id, cnf in enumerate(block_setting):
            # Adjust stochastic depth probability based on the depth of the stage block
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)

            self.blocks.append(
                block(
                    input_size=input_size,
                    cnf=cnf,
                    residual_pool=residual_pool,
                    residual_with_cls_embed=residual_with_cls_embed,
                    rel_pos_embed=rel_pos_embed,
                    proj_after_attn=proj_after_attn,
                    dropout=attention_dropout,
                    stochastic_depth_prob=sd_prob,
                    norm_layer=norm_layer,
                )
            )

            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
        self.norm = norm_layer(block_setting[-1].output_channels)

        # Classification Head
        self.head = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(block_setting[-1].output_channels, num_classes),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the DinoMVit model.

        Args:
            x (torch.Tensor): Input video frames of shape [B, T, C, H, W].
                - B (int): Batch size.
                - T (int): Temporal dimension (number of frames).
                - C (int): Number of channels.
                - H (int): Height of the frames.
                - W (int): Width of the frames.

        Returns:
            torch.Tensor: Classification logits of shape [B, num_classes].
                - B (int): Batch size.
                - num_classes (int): Number of output classes.
        """
        # Extract features using DINOv2 backbone
        x = self.dino_backbone(x)  # [B, T, N, D]

        # Project features to match embedding dimension of MViT
        x = self.feature_proj(x)  # [B, T, N, D_model]

        # Flatten temporal and spatial dimensions for transformer processing
        x = x.flatten(1, 2)  # [B, T*N, D_model]

        # Add spatio-temporal positional encodings
        x = self.pos_encoding(x)  # [B, T*N, D_model]

        # Pass features through the transformer encoder blocks
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)  # [B, T*N, D_model]

        # classifier "token" as used by standard language architectures
        x = x[:, 0]  # [B, D_model]

        # Apply the classification head to obtain logits
        x = self.head(x)  # [B, num_classes]

        return x
