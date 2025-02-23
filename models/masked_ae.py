"""Masked Autoencoder implementation for video transformers.

This module implements a masked autoencoder approach for self-supervised pretraining
of video transformers, specifically adapted for the MViT architecture.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchvision.models.video.mvit import MViT_V2_S_Weights

from models.video_mano_regressor import mvit_v2_s


class MaskedAutoencoderViT(pl.LightningModule):
    """Masked Autoencoder for Video Vision Transformers.

    This implements a self-supervised pretraining approach where random patches
    in the input video are masked and the model learns to reconstruct them.

    Args:
        num_frames (int): Number of frames in input video.
        height (int): Height of input frames.
        width (int): Width of input frames.
        patch_size (tuple): Size of patches to extract (t, h, w).
        mask_ratio (float): Ratio of patches to mask.
        decoder_dim (int): Dimension of decoder features.
        decoder_depth (int): Number of transformer layers in decoder.
        decoder_heads (int): Number of attention heads in decoder.
        learning_rate (float): Learning rate for optimization.

    """

    encoder: nn.Module
    mask_token: nn.Parameter
    decoder_embed: nn.Linear
    decoder_pos_embed: nn.Parameter
    decoder: nn.TransformerDecoder
    decoder_pred: nn.Linear

    def __init__(
        self,
        num_frames: int = 16,
        height: int = 224,
        width: int = 224,
        patch_size: tuple[int, int, int] = (2, 16, 16),
        mask_ratio: float = 0.75,  # noqa: ARG002
        decoder_dim: int = 512,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        learning_rate: float = 1.5e-4,  # noqa: ARG002
    ) -> None:
        """Initialize the Masked Autoencoder.

        Args:
            num_frames (int, optional): Number of frames in input video. Defaults to 16.
            height (int, optional): Height of input frames. Defaults to 224.
            width (int, optional): Width of input frames. Defaults to 224.
            patch_size (tuple[int, int, int], optional): Size of patches to extract (t, h, w).
                Defaults to (2, 16, 16).
            mask_ratio (float, optional): Ratio of patches to mask during training.
                Defaults to 0.75.
            decoder_dim (int, optional): Dimension of decoder features. Defaults to 512.
            decoder_depth (int, optional): Number of transformer layers in decoder.
                Defaults to 4.
            decoder_heads (int, optional): Number of attention heads in decoder.
                Defaults to 8.
            learning_rate (float, optional): Learning rate for optimization.
                Defaults to 1.5e-4.

        """
        super().__init__()

        self.save_hyperparameters()

        # Calculate number of patches
        self.num_patches = (num_frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])

        # Encoder (MViT backbone)
        self.encoder = mvit_v2_s(
            spatial_size=(height, width),
            temporal_size=num_frames,
            weights=MViT_V2_S_Weights.DEFAULT,  # Use pretrained weights
        )
        encoder_dim = self.encoder.head.in_features
        self.encoder.head = nn.Identity()  # Remove classification head

        # Masking token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # Decoder components
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, decoder_dim),  # +1 for cls token
            requires_grad=True,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            activation=F.gelu,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=decoder_depth,
        )

        # Reconstruction head - predict in latent space
        self.decoder_pred = nn.Linear(
            decoder_dim,
            96,  # Match MViT's initial embedding dimension
        )

        # Initialize weights
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                torch.nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

    def random_masking(self, x: Tensor, mask_ratio: float) -> tuple[Tensor, Tensor, Tensor]:
        """Randomly mask input patches.

        Args:
            x (Tensor): Input tensor of shape (B, L, D).
            mask_ratio (float): Ratio of patches to mask.

        Returns:
            tuple:
                - Tensor of unmasked patches (B, (1-ratio)*L, D)
                - Tensor of masking indices (B, L)
                - Tensor of restore indices (B, L)

        """
        B, L, D = x.shape  # noqa: N806
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first len_keep indices
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x,
            dim=1,
            index=ids_keep.unsqueeze(-1).repeat(1, 1, D),
        )

        # Generate mask
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x: Tensor, mask_ratio: float) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass through encoder with masking.

        Args:
            x (Tensor): Input tensor of shape (B, C, T, H, W).
            mask_ratio (float): Ratio of patches to mask.

        Returns:
            tuple:
                - Tensor of encoded features
                - Tensor of original patches in latent space
                - Tensor of masking indices
                - Tensor of restore indices

        """
        # Convert video to patches using MViT's conv_proj
        patches = self.encoder.conv_proj(x)  # B, D, T', H', W'
        patches = patches.flatten(2).transpose(1, 2)  # B, L, D

        # Add masking before encoder
        x_masked, mask, ids_restore = self.random_masking(patches, mask_ratio)

        # Add cls token and positional encoding using MViT's pos_encoding
        x_masked = self.encoder.pos_encoding(x_masked)

        # Pass through encoder blocks using MViT's spatial and temporal sizes
        thw = (self.encoder.pos_encoding.temporal_size, *self.encoder.pos_encoding.spatial_size)

        for block in self.encoder.blocks:
            x_masked, thw = block(x_masked, thw)
        x_masked = self.encoder.norm(x_masked)

        return x_masked, patches, mask, ids_restore

    def forward_decoder(self, x: Tensor, ids_restore: Tensor) -> Tensor:
        """Forward pass through decoder.

        Args:
            x (Tensor): Encoded features.
            ids_restore (Tensor): Indices to restore original sequence.

        Returns:
            Tensor: Reconstructed patches.

        """
        B = x.shape[0]  # noqa: N806

        # Remove cls token
        x = x[:, 1:]

        # Embed tokens
        x = self.decoder_embed(x)

        # Append mask tokens
        mask_tokens = self.mask_token.repeat(B, ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[-1]),
        )

        # Add positional embedding
        x = x + self.decoder_pos_embed[:, 1:]  # Skip cls token position

        # Decoder blocks
        x = self.decoder(x, x)

        # Predictor
        return self.decoder_pred(x)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, T, H, W).

        Returns:
            tuple:
                - Tensor of reconstructed patches in latent space
                - Tensor of original patches in latent space
                - Tensor of mask

        """
        # Encode with masking
        latent, patches, mask, ids_restore = self.forward_encoder(x, self.hparams.mask_ratio)

        # Decode
        pred = self.forward_decoder(latent, ids_restore)

        return pred, patches, mask

    def training_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,  # noqa: ARG002
    ) -> Tensor:
        """Training step.

        Args:
            batch (tuple): Tuple of (video, _) where video has shape (B, T, C, H, W).
            batch_idx (int): Index of batch.

        Returns:
            Tensor: Loss value.

        """
        x = batch[0]
        x = x.permute(0, 2, 1, 3, 4)  # B, T, C, H, W -> B, C, T, H, W

        # Forward pass
        pred, patches, mask = self(x)

        # Calculate loss on masked patches only
        loss = F.mse_loss(pred, patches, reduction="none")
        loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * patches.shape[-1])

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Validation step.

        Args:
            batch (tuple): Tuple of (video, _) where video has shape (B, T, C, H, W).
            batch_idx (int): Index of batch.

        """
        x = batch[0]
        x = x.permute(0, 2, 1, 3, 4)  # B, T, C, H, W -> B, C, T, H, W

        # Forward pass
        pred, patches, mask = self(x)

        # Calculate loss on masked patches only
        loss = F.mse_loss(pred, patches, reduction="none")
        loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * patches.shape[-1])

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Optimizer:
        """Configure optimizers.

        Returns:
            Optimizer: Optimizer instance.

        """
        return AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.05,
        )
