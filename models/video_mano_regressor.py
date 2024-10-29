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
from manopth.manolayer import ManoLayer
from torch import nn
from torch.nn import functional as F  # noqa: N812
from torch.optim import Adam
from torchvision.models.video.mvit import MSBlockConfig, MViT, MViT_V2_S_Weights, _mvit
from torchvision.models.video.mvit import mvit_v2_s as _mvit_v2_s_pretrained

from models.utils import get_mano_joints, project_joints_to_2d


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
        mano_root: str,
        ncomps: int = 45,
        use_pca: bool = False,
        flat_hand_mean: bool = True,
        mano_params: int = 61,  # Single hand, 61 parameters
        learning_rate: float = 1e-3,  # noqa: ARG002
        pretrained: bool = False,
    ) -> None:
        """Initialize the VideoMANORegressor model.

        Args:
            num_frames (int): Number of frames in each video sequence.
            height (int): Height of each video frame.
            width (int): Width of each video frame.
            mano_root (str): Root path of the MANO model files.
            ncomps (int, optional): Number of PCA components. Defaults to 45.
            use_pca (bool, optional): Whether to use PCA for pose parameters. Defaults to False.
            flat_hand_mean (bool, optional): Whether to use flat hand mean. Defaults to True.
            mano_params (int, optional): Number of MANO parameters to predict. Defaults to 61.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.

        Note:
            The model uses an MViT v2 Small backbone as the encoder, followed by a regressor
            network to predict MANO parameters for a single hand.

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

        # Regressors for left and right hands
        self.regressor_left = nn.Sequential(
            nn.Linear(backbone_out_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, mano_params * num_frames),
        )

        self.regressor_right = nn.Sequential(
            nn.Linear(backbone_out_features, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, mano_params * num_frames),
        )
        self.mano_left = ManoLayer(
            mano_root=mano_root,
            ncomps=ncomps,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            side="left",
        )
        self.mano_left.requires_grad_(requires_grad=False)

        self.mano_right = ManoLayer(
            mano_root=mano_root,
            ncomps=ncomps,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean,
            side="right",
        )
        self.mano_right.requires_grad_(requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the VideoMANORegressor model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            torch.Tensor: Predicted MANO parameters.

        """
        features = self.backbone(x)
        left_hand_params = self.regressor_left(features)
        right_hand_params = self.regressor_right(features)

        # Reshape the outputs
        left_hand_params = left_hand_params.view(x.shape[0], -1, self.hparams.mano_params)
        right_hand_params = right_hand_params.view(x.shape[0], -1, self.hparams.mano_params)

        left_hand_joints, right_hand_joints = get_mano_joints(
            left_hand_params,
            right_hand_params,
            self.mano_left,
            self.mano_right,
        )

        return left_hand_params, right_hand_params, left_hand_joints, right_hand_joints

    def loss_function(
        self,
        y_pred_left: torch.Tensor,
        y_pred_right: torch.Tensor,
        y_true_left: torch.Tensor,
        y_true_right: torch.Tensor,
        pred_left_hand_joints: torch.Tensor,
        pred_right_hand_joints: torch.Tensor,
        true_left_hand_joints: torch.Tensor,
        true_right_hand_joints: torch.Tensor,
        intrinsic_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the loss for the model.

        The loss function consists of three components for each hand:
        1. Pose loss: MSE between predicted and ground truth pose parameters (first 45 values).
        2. Shape loss: MSE between predicted and ground truth shape parameters (last 10 values).
        3. Keypoints loss: L1 loss between predicted and ground truth 3D hand joints.

        The total loss for each hand is the sum of these three components.

        Args:
            y_pred_left (torch.Tensor): Predicted MANO parameters for left hand.
            y_pred_right (torch.Tensor): Predicted MANO parameters for right hand.
            y_true_left (torch.Tensor): Ground truth MANO parameters for left hand.
            y_true_right (torch.Tensor): Ground truth MANO parameters for right hand.
            pred_left_hand_joints (torch.Tensor): Predicted 3D keypoints for left hand.
            pred_right_hand_joints (torch.Tensor): Predicted 3D keypoints for right hand.
            true_left_hand_joints (torch.Tensor): Ground truth 3D keypoints for left hand.
            true_right_hand_joints (torch.Tensor): Ground truth 3D keypoints for right hand.
            intrinsic_matrix (torch.Tensor): Camera intrinsic matrix.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The computed losses for left and right hands.

        """
        true_keypoints_2d_left = project_joints_to_2d(true_left_hand_joints, intrinsic_matrix)
        true_keypoints_2d_right = project_joints_to_2d(true_right_hand_joints, intrinsic_matrix)
        pred_keypoints_2d_left = project_joints_to_2d(pred_left_hand_joints, intrinsic_matrix)
        pred_keypoints_2d_right = project_joints_to_2d(pred_right_hand_joints, intrinsic_matrix)
        # Left hand loss
        pose_loss_left = F.mse_loss(y_pred_left[..., :45], y_true_left[..., :45])
        shape_loss_left = F.mse_loss(y_pred_left[..., -10:], y_true_left[..., -10:])
        keypoints_loss_left = F.l1_loss(pred_left_hand_joints, true_left_hand_joints)
        keypoints_2d_loss_left = F.l1_loss(pred_keypoints_2d_left, true_keypoints_2d_left)
        left_hand_loss = pose_loss_left + shape_loss_left + keypoints_loss_left + keypoints_2d_loss_left

        # Right hand loss
        pose_loss_right = F.mse_loss(y_pred_right[..., :45], y_true_right[..., :45])
        shape_loss_right = F.mse_loss(y_pred_right[..., -10:], y_true_right[..., -10:])
        keypoints_loss_right = F.l1_loss(pred_right_hand_joints, true_right_hand_joints)
        keypoints_2d_loss_right = F.l1_loss(pred_keypoints_2d_right, true_keypoints_2d_right)
        right_hand_loss = pose_loss_right + shape_loss_right + keypoints_loss_right + keypoints_2d_loss_right

        return left_hand_loss, right_hand_loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Training step of the VideoMANORegressor model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing input video frames and target MANO parameters.
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.

        """
        x, y_left, y_right, _ = batch

        # Ensure input is in the correct format for MViT (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]

        y_pred_left, y_pred_right, pred_left_hand_joints, pred_right_hand_joints = self(x)
        true_left_hand_joints, true_right_hand_joints = get_mano_joints(
            y_left,
            y_right,
            self.mano_left,
            self.mano_right,
        )
        left_loss, right_loss = self.loss_function(
            y_pred_left,
            y_pred_right,
            y_left,
            y_right,
            pred_left_hand_joints,
            pred_right_hand_joints,
            true_left_hand_joints,
            true_right_hand_joints,
        )
        loss = left_loss + right_loss

        # Additional metrics
        left_mje = torch.linalg.vector_norm(pred_left_hand_joints - true_left_hand_joints, dim=-1).mean(dim=-1).mean()
        right_mje = (
            torch.linalg.vector_norm(pred_right_hand_joints - true_right_hand_joints, dim=-1).mean(dim=-1).mean()
        )
        left_mse = F.mse_loss(y_pred_left, y_left)
        right_mse = F.mse_loss(y_pred_right, y_right)
        left_mae = F.l1_loss(y_pred_left, y_left)
        right_mae = F.l1_loss(y_pred_right, y_right)

        self.log("train/left_loss", left_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/right_loss", right_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mean_left_mse", left_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_right_mse", right_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_left_mae", left_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_right_mae", right_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_left_mje", left_mje, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_right_mje", right_mje, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_mje", left_mje + right_mje, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:  # noqa: ARG002
        """Validation step of the VideoMANORegressor model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple containing input video frames and target MANO parameters.
            batch_idx (int): Index of the batch.

        """
        x, y_left, y_right, _ = batch

        # Ensure input is in the correct format for MViT (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]

        y_pred_left, y_pred_right, pred_left_hand_joints, pred_right_hand_joints = self(x)
        true_left_hand_joints, true_right_hand_joints = get_mano_joints(
            y_left,
            y_right,
            self.mano_left,
            self.mano_right,
        )
        left_loss, right_loss = self.loss_function(
            y_pred_left,
            y_pred_right,
            y_left,
            y_right,
            pred_left_hand_joints,
            pred_right_hand_joints,
            true_left_hand_joints,
            true_right_hand_joints,
        )
        loss = left_loss + right_loss

        # Additional metrics
        left_mje = torch.linalg.vector_norm(pred_left_hand_joints - true_left_hand_joints, dim=-1).mean(dim=-1).mean()
        right_mje = (
            torch.linalg.vector_norm(pred_right_hand_joints - true_right_hand_joints, dim=-1).mean(dim=-1).mean()
        )
        left_mse = F.mse_loss(y_pred_left, y_left)
        right_mse = F.mse_loss(y_pred_right, y_right)
        left_mae = F.l1_loss(y_pred_left, y_left)
        right_mae = F.l1_loss(y_pred_right, y_right)

        self.log("val/left_loss", left_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/right_loss", right_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mean_left_mse", left_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_right_mse", right_mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_left_mae", left_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_right_mae", right_mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_left_mje", left_mje, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_right_mje", right_mje, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_mje", left_mje + right_mje, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the VideoMANORegressor model.

        Returns:
            torch.optim.Optimizer: The Adam optimizer instance.

        """
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
