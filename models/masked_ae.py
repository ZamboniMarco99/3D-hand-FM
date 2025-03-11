"""Masked Autoencoder implementation for video transformers.

This module implements a masked autoencoder approach for self-supervised pretraining
of video transformers, specifically adapted for the MViT architecture.
"""

import math

import pytorch_lightning as pl
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.optimizer import Optimizer
from torchvision.models.video.mvit import MViT_V2_S_Weights

from models.video_mano_regressor import get_mvit_v2_s_block_setting, mvit_v2_s


class MaskedAutoencoderViT(pl.LightningModule):
    """Masked Autoencoder for Video Vision Transformers.

    This implements a self-supervised pretraining approach where random patches
    in the input video are masked and the model learns to reconstruct them.

    Args:
        num_frames (int): Number of frames in input video.
        height (int): Height of input frames.
        width (int): Width of input frames.
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
    ids_restore: Tensor

    def __init__(
        self,
        num_frames: int = 16,
        height: int = 224,
        width: int = 224,
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

        # Encoder (MViT backbone)
        self.encoder = mvit_v2_s(
            spatial_size=(height, width),
            temporal_size=num_frames,
            weights=MViT_V2_S_Weights.DEFAULT,  # Use pretrained weights
        )
        encoder_dim = get_mvit_v2_s_block_setting()[-1].output_channels
        self.encoder.head = nn.Identity()  # Remove classification head

        # Get patch size from MViT's conv_proj layer
        patch_size = (
            self.encoder.conv_proj.stride[0],  # temporal stride
            self.encoder.conv_proj.stride[1],  # height stride
            self.encoder.conv_proj.stride[2],  # width stride
        )

        # Calculate number of patches based on input dimensions and patch size
        self.num_patches = (num_frames // patch_size[0]) * (height // patch_size[1]) * (width // patch_size[2])

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
        num_patches = L - 1  # Exclude the class token
        num_masked = int(num_patches * mask_ratio)

        random_indices = torch.rand(B, num_patches, device=x.device).argsort(dim=1)
        mask = torch.ones(B, num_patches, device=x.device)

        masked_indices = random_indices[:, :num_masked]
        mask.scatter_(1, masked_indices, 0)  # Set the masked indices to 0

        # Keep the class token and remove the masked patches
        mask = torch.cat(
            [torch.ones(B, 1, device=x.device), mask],
            dim=1,
        )  # Add the class token back into the mask (always kept)

        # Apply the mask to the patches using advanced indexing
        x_masked = x * mask.unsqueeze(-1)  # Element-wise multiplication to mask out the patches

        # Remove the masked patches: we select the unmasked patches for each batch
        # We need to reshape the tensor to get rid of masked patches
        x_masked = x_masked[mask.bool()].view(B, -1, D)  # Flatten to get rid of masked patches

        # Restore indices (for reordering or unmasking purposes)
        ids_restore = random_indices

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

        # Add positional encoding first using MViT's pos_encoding
        patches = self.encoder.pos_encoding(patches)

        # Apply masking after positional encoding
        x_masked, mask, ids_restore = self.random_masking(patches, mask_ratio)

        total_patches = x_masked.shape[1] - 1
        orig_thw = (self.encoder.pos_encoding.temporal_size, *self.encoder.pos_encoding.spatial_size)
        total_hw = orig_thw[1] * orig_thw[2]

        # Maintain aspect ratios while adjusting for fewer patches
        ratio_t = orig_thw[0] / total_patches
        ratio_hw = total_hw / total_patches
        new_t = max(1, round(total_patches * ratio_t))
        new_h = max(1, round(math.sqrt(total_patches * ratio_hw * orig_thw[1] / orig_thw[2])))
        new_w = max(1, round(total_patches / (new_t * new_h)))
        thw = (new_t, new_h, new_w)

        for block in self.encoder.blocks:
            x_masked, thw = block(x_masked, thw)
        x_masked = self.encoder.norm(x_masked)

        return x_masked, patches, mask, ids_restore

    def forward_decoder(self, x: Tensor, ids_restore: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through decoder.

        Args:
            x (Tensor): Encoded features.
            ids_restore (Tensor): Indices to restore original sequence.
            mask (Tensor): Mask of shape (B, L).

        Returns:
            tuple:
                - Tensor: Reconstructed patches for masked tokens only
                - Tensor: Indices of masked tokens in the original sequence

        """
        B = x.shape[0]  # noqa: N806

        # Remove cls token
        x = x[:, 1:]

        # Embed tokens
        x = self.decoder_embed(x)

        # Get number of masked tokens
        n_masked = ids_restore.shape[1] - x.shape[1]

        # Generate mask tokens
        mask_tokens = self.mask_token.repeat(B, n_masked, 1)

        # Find which indices were masked (where mask == 0)
        masked_positions = mask == 0  # Shape (B, n_masked)
        # Select the positional embeddings corresponding to the masked positions
        decoder_pos_embed_expanded = self.decoder_pos_embed.expand(B, -1, -1)  # Shape (B, num_patches, decoder_dim)
        print(f"{decoder_pos_embed_expanded.shape=}")
        print(f"{masked_positions.shape=}")
        print(f"{decoder_pos_embed_expanded[masked_positions].shape=}")

        masked_pos_embed = decoder_pos_embed_expanded[masked_positions].view(
            B, n_masked, -1
        )  # Shape (B, n_masked, decoder_dim)

        # Add positional embeddings to mask_tokens
        mask_tokens = mask_tokens + masked_pos_embed

        # Decoder blocks - only process masked tokens
        decoded = self.decoder(mask_tokens, x)

        # Predictor
        pred = self.decoder_pred(decoded)

        # Return predictions and the positions they correspond to
        return pred, masked_positions

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x (Tensor): Input tensor of shape (B, C, T, H, W).

        Returns:
            tuple:
                - Tensor of reconstructed patches for masked positions
                - Tensor of original patches in latent space
                - Tensor of mask
                - Tensor of masked positions
                - Tensor of restore indices for mapping masked positions back

        """
        # Encode with masking
        latent, patches, mask, ids_restore = self.forward_encoder(x, self.hparams.mask_ratio)

        # Decode - only get predictions for masked tokens
        pred, masked_positions = self.forward_decoder(latent, ids_restore, mask)

        return pred, patches, mask, masked_positions, ids_restore

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

        # Forward pass - get predictions only for masked tokens
        pred, patches, mask, masked_positions, ids_restore = self(x)

        target = patches.gather(
            1,
            ids_restore.gather(1, masked_positions.unsqueeze(-1).expand(-1, -1, 1)).expand(-1, -1, patches.shape[-1]),
        )

        # Calculate loss only on masked patches
        loss = F.mse_loss(pred, target)

        self.log("train/loss", loss, on_step=True, sync_dist=True, on_epoch=True, prog_bar=True)
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

        # Forward pass - get predictions only for masked tokens
        pred, patches, mask, masked_positions, ids_restore = self(x)

        target = patches.gather(
            1,
            ids_restore.gather(1, masked_positions.unsqueeze(-1).expand(-1, -1, 1)).expand(-1, -1, patches.shape[-1]),
        )

        # Calculate loss only on masked patches
        loss = F.mse_loss(pred, target)

        self.log("val/loss", loss, on_step=False, sync_dist=True, on_epoch=True, prog_bar=True)

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
