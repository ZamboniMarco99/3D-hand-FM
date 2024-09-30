"""Video Variational Autoencoder (VAE) implementation.

This module contains the VideoVAE class, which implements a Variational Autoencoder
for video data using PyTorch Lightning. The VAE consists of an encoder and a decoder,
designed to learn a compressed latent representation of video sequences.

Example usage:
    model = VideoVAE(input_channels=3, num_frames=30, height=64, width=64)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, train_dataloader, val_dataloader)
"""

import pytorch_lightning as pl
import torch
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam

from models.vae.video_decoder import VideoDecoder
from models.vae.video_encoder import VideoEncoder


class VideoVAE(pl.LightningModule):
    """Video Variational Autoencoder (VAE) model.

    This class implements a VAE for video data using PyTorch Lightning.

    Args:
        input_channels (int): Number of input channels in the video frames.
        num_frames (int): Number of frames in each video sequence.
        height (int): Height of each video frame.
        width (int): Width of each video frame.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.

    Attributes:
        encoder (Encoder): The encoder network of the VAE.
        decoder (Decoder): The decoder network of the VAE.

    """

    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        learning_rate: float = 1e-3,  # noqa: ARG002
        kld_weight: float = 1e-4,
        mse_weight: float = 1e-8,
    ) -> None:
        """Initialize the VideoVAE model.

        Args:
            input_channels (int): Number of input channels in the video frames.
            num_frames (int): Number of frames in each video sequence.
            height (int): Height of each video frame.
            width (int): Width of each video frame.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.

        """
        super().__init__()
        self.save_hyperparameters()

        self.encoder = VideoEncoder(num_frames=num_frames, height=height, width=width)
        self.decoder = VideoDecoder(num_frames=num_frames, height=height, width=width)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The reconstructed input tensor.
                - The mean of the latent Gaussian.
                - The log variance of the latent Gaussian.

        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        return self.decoder(z), mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterize the latent space using the reparameterization trick.

        Args:
            mu (torch.Tensor): Mean of the latent Gaussian.
            log_var (torch.Tensor): Log variance of the latent Gaussian.

        Returns:
            torch.Tensor: Sampled latent vector.

        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Training step of the VAE model.

        Args:
            batch (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.

        """
        x, _ = batch
        # Permute the input tensor to match the expected shape for 3D convolutions
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        recon_x, mu, log_var = self(x)
        losses = self.loss_function(recon_x, x, mu, log_var)
        self.log("train/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/mse_loss", losses["mse"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/kld_loss", losses["kld"], on_step=False, on_epoch=True, prog_bar=False)
        return losses["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> None:  # noqa: ARG002
        """Validation step of the VAE model.

        Args:
            batch (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).
            batch_idx (int): Index of the batch.

        """
        x, _ = batch
        # Permute the input tensor to match the expected shape for 3D convolutions
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        recon_x, mu, log_var = self(x)
        losses = self.loss_function(recon_x, x, mu, log_var)
        self.log("val/loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mse_loss", losses["mse"], on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/kld_loss", losses["kld"], on_step=False, on_epoch=True, prog_bar=False)

    def loss_function(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the VAE loss function.

        Args:
            recon_x (torch.Tensor): Reconstructed input tensor.
            x (torch.Tensor): Original input tensor.
            mu (torch.Tensor): Mean of the latent Gaussian.
            log_var (torch.Tensor): Log variance of the latent Gaussian.

        Returns:
            dict: A dictionary containing:
                - 'loss': Total loss (reconstruction loss + KL divergence)
                - 'mse': Mean Squared Error (reconstruction loss)
                - 'kld': Kullback-Leibler Divergence

        """
        # MSE loss for video reconstruction
        mse_weight = self.hparams.mse_weight
        kld_weight = self.hparams.kld_weight
        mse = F.mse_loss(recon_x, x, reduction="sum")
        kld = torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        loss = mse * mse_weight + kld * kld_weight
        return {"loss": loss, "mse": mse, "kld": kld}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the VAE model.

        Returns:
            torch.optim.Optimizer: The Adam optimizer instance.

        """
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
