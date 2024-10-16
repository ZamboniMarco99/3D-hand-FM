"""Video MANO Regressor implementation using MViT.

This module contains the VideoMANORegressor class, which implements a model
to regress MANO parameters from video sequences using PyTorch Lightning.
The model uses a Multiscale Vision Transformer (MViT) as the encoder to extract features
from video frames and a regressor to predict MANO parameters.

Example usage:
    model = VideoMANORegressor(num_frames=32, height=256, width=256)
    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model, train_dataloader, val_dataloader)
"""

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam
from torchvision.models.video.mvit import MSBlockConfig, MViT, MViT_V2_S_Weights, _mvit
from torchvision.models.video.mvit import mvit_v2_s as _mvit_v2_s_pretrained

from models.utils import get_mano_joints


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


class VideoMANORegressor(pl.LightningModule):
    """Video MANO Regressor model using MViT.

    This class implements a model to regress MANO parameters from video sequences
    using PyTorch Lightning with a Multiscale Vision Transformer (MViT) as the encoder.

    Args:
        num_frames (int): Number of frames in each video sequence.
        height (int): Height of each video frame.
        width (int): Width of each video frame.
        mano_params (int): Number of MANO parameters to predict.
        learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.

    Attributes:
        encoder (nn.Module): The MViT encoder to extract features from video frames.
        regressor (nn.Sequential): The regressor network to predict MANO parameters.

    """

    def __init__(
        self,
        num_frames: int,
        height: int,
        width: int,
        mano_root: str,  # noqa: ARG002
        mano_params: int = 122,  # Two hands, 61 parameters per hand
        learning_rate: float = 1e-3,  # noqa: ARG002
        pretrained: bool = False,
    ) -> None:
        """Initialize the VideoMANORegressor model.

        Args:
            num_frames (int): Number of frames in each video sequence.
            height (int): Height of each video frame.
            width (int): Width of each video frame.
            mano_root (str): Root path of the MANO model files.
            mano_params (int, optional): Number of MANO parameters to predict. Defaults to 122 (61 per hand).
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.

        Note:
            The model uses an MViT v2 Small backbone as the encoder, followed by a regressor
            network to predict MANO parameters for both hands.

        """
        super().__init__()
        self.save_hyperparameters()

        # MViT encoder
        if pretrained:
            self.backbone = _mvit_v2_s_pretrained(
                weights=MViT_V2_S_Weights.DEFAULT,
            )
        else:
            self.backbone = mvit_v2_s(
                spatial_size=(height, width),
                temporal_size=num_frames,
            )

        backbone_out_features = get_mvit_v2_s_block_setting()[-1].output_channels

        # Remove the classification head
        self.backbone.head = nn.Identity()

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(backbone_out_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, mano_params * num_frames),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the VideoMANORegressor model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            torch.Tensor: Predicted MANO parameters.

        """
        features = self.backbone(x)
        mano_params = self.regressor(features)
        return mano_params.view(x.shape[0], -1, self.hparams.mano_params)

    def loss_function(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate the loss for the model.

        Args:
            y_pred (torch.Tensor): Predicted MANO parameters.
            y_true (torch.Tensor): Ground truth MANO parameters.

        Returns:
            torch.Tensor: The computed loss.

        """
        return F.mse_loss(y_pred, y_true)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Training step of the VideoMANORegressor model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing input video frames and target MANO parameters.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.

        """
        x, y = batch

        # Ensure input is in the correct format for MViT (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Additional metrics
        mse = F.mse_loss(y_pred, y, reduction="mean")
        mae = F.l1_loss(y_pred, y, reduction="mean")
        self.log("train/mean_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_mae", mae, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        """Validation step of the VideoMANORegressor model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing input video frames and target MANO parameters.
            batch_idx (int): Index of the batch.

        """
        x, y = batch

        # Ensure input is in the correct format for MViT (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]

        y_pred = self(x)
        loss = self.loss_function(y_pred, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Additional metrics
        mse = F.mse_loss(y_pred, y, reduction="mean")
        mae = F.l1_loss(y_pred, y, reduction="mean")

        pred_left_hand_joints, pred_right_hand_joints = get_mano_joints(y_pred, mano_root=self.hparams.mano_root)
        target_left_hand_joints, target_right_hand_joints = get_mano_joints(y, mano_root=self.hparams.mano_root)
        left_mje = torch.linalg.vector_norm(pred_left_hand_joints - target_left_hand_joints, dim=-1).mean(dim=-1).mean()
        right_mje = (
            torch.linalg.vector_norm(pred_right_hand_joints - target_right_hand_joints, dim=-1).mean(dim=-1).mean()
        )

        self.log("val/mean_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_left_mje", left_mje, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_right_mje", right_mje, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_mje", left_mje + right_mje, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the VideoMANORegressor model.

        Returns:
            torch.optim.Optimizer: The Adam optimizer instance.

        """
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
