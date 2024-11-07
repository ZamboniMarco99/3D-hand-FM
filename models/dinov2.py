# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license
from functools import partial

import torch
from torch import nn
from torchvision.models.video.mvit import *
from typing import Tuple, Sequence, Optional, Callable

from torchvision.models.video.mvit import MultiscaleBlock, PositionalEncoding
from torchvision.utils import _log_api_usage_once


class Dinov2Backbone(nn.Module):
    """
    DinoV2 Backbone for feature extraction.

    Preprocessing:
        The input video frames are resized to 224x224 and normalized using the following mean and std:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    """

    def __init__(self, name="dinov2_vitb14", pretrained=False, *args, **kwargs):
        super().__init__()
        self.name = name
        self.encoder = torch.hub.load("facebookresearch/dinov2", self.name, pretrained=pretrained)
        self.patch_size = self.encoder.patch_size
        self.embed_dim = self.encoder.embed_dim

    def forward(self, x):
        """
        Forward pass to extract Dinov2 features.

        Args:
            x (torch.Tensor): Input video frames of shape [B, T, C, H, W].
                B: Batch size
                C: Number of channels
                T: Temporal dimension (number of frames)
                H: Height of the frames
                W: Width of the frames

        Returns:
            torch.Tensor: Extracted features of shape [B, T, N, D].
                B: Batch size
                T: Temporal dimension (number of frames)
                N: Number of spatial tokens per frame
                D: Embedding dimension
        """
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        y = self.encoder.get_intermediate_layers(x)[0]  # ViT-L+896x896: [bs,4096,1024] - [bs,nb_patches,emb]
        y = y.reshape(B, T, -1, self.embed_dim)
        return y


class DinoMVit(nn.Module):
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
        Modified MViT to accept video features with temporal and spatial positional encodings.

        Args:
            spatial_size (Tuple[int, int]): Spatial size of the input frames (H, W).
            temporal_size (int): Number of frames in the video sequence.
            block_setting (Sequence['MSBlockConfig']): List of MSBlockConfig objects defining the architecture.
            residual_pool (bool): Whether to use residual pooling.
            residual_with_cls_embed (bool): Whether to use residual connections with class token embedding.
            rel_pos_embed (bool): Whether to use relative positional embeddings.
            proj_after_attn (bool): Whether to project after attention.
            dropout (float, optional): Dropout probability. Defaults to 0.5.
            attention_dropout (float, optional): Attention dropout probability. Defaults to 0.0.
            stochastic_depth_prob (float, optional): Stochastic depth probability. Defaults to 0.0.
            num_classes (int, optional): Number of output classes. Defaults to 400.
            block (Optional[Callable[..., nn.Module]], optional): Block module to use. Defaults to None.
            norm_layer (Optional[Callable[..., nn.Module]], optional): Normalization layer to use. Defaults to None.
        """
        super().__init__()
        _log_api_usage_once(self)
        block_setting = block_setting[4:]
        embed_dim = block_setting[0].input_channels
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")

        if block is None:
            block = MultiscaleBlock

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # Replace Patch Embedding module with Dino
        self.dino_backbone = Dinov2Backbone()
        feature_dim = self.dino_backbone.embed_dim

        # Feature Projection Layer
        if feature_dim != embed_dim:
            self.feature_proj = nn.Linear(feature_dim, embed_dim)
        else:
            self.feature_proj = nn.Identity()

        h_patches = spatial_size[0] // self.dino_backbone.patch_size
        w_patches = spatial_size[1] // self.dino_backbone.patch_size

        # Spatio-Temporal Class Positional Encoding
        self.pos_encoding = PositionalEncoding(
            embed_size=embed_dim,
            spatial_size=(h_patches, w_patches),
            temporal_size=temporal_size,
            rel_pos_embed=rel_pos_embed,
        )

        # Encoder module
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

        # Classifier module
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
        Forward pass for dino featurized video.
        Args:
            x (torch.Tensor): Input video frames [B, T, C, H, W]
        Returns:
            torch.Tensor: Logits [B, num_classes].
        """
        # Extract features using Dinov2 backbone
        x = self.dino_backbone(x)  # [B, T, N, D]

        # Project features if necessary
        x = self.feature_proj(x)  # [B, T, N, D_model]

        x = x.flatten(1, 2)  # [B, T*N, D_model]

        # add positional encoding
        x = self.pos_encoding(x)  # [B, 1 + T*N, D_model]

        # pass patches through the encoder
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)  # [B, 1 + T*N, D_model]

        # classifier "token" as used by standard language architectures
        x = x[:, 0]  # [B, D_model]
        x = self.head(x)  # [B, num_classes]

        return x
