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

from models.utils import get_mano_joints, mano_to_sixd, project_joints_to_2d, reconstruction_error


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
        loss_weights: dict[str, float] | None = None,  # noqa: ARG002
        pretrained: bool = False,
        focal_length: float | None = None,
        sixd: bool = False,
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
            loss_weights (dict[str, float], optional): Loss weights for the different components.
            pretrained (bool, optional): Whether to use pretrained weights for the backbone. Defaults to False.
            focal_length (float | None, optional): Focal length for camera intrinsics. Defaults to width/2.
            sixd (bool, optional): Whether to use 6D pose representation. Defaults to False.

        Note:
            The model uses an MViT v2 Small backbone as the encoder, followed by a regressor
            network to predict MANO parameters for a single hand.

        """
        super().__init__()

        # Set default focal length if not provided
        if focal_length is None:
            focal_length = width / 2

        self.save_hyperparameters()

        # Create default camera intrinsic matrix
        self.register_buffer(
            "intrinsic_matrix",
            torch.tensor(
                [
                    [focal_length, 0, width / 2],
                    [0, focal_length, height / 2],
                    [0, 0, 1],
                ],
                dtype=torch.float32,
            ),
        )

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

        # Regressors for left (only) hand
        self.regressor = nn.Sequential(
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
        self.sixd = sixd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the VideoMANORegressor model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tuple containing:
                - Predicted MANO parameters
                - Predicted 3D hand joints
                - Predicted 2D hand joints

        """
        features = self.backbone(x)
        hand_params = self.regressor(features)

        # Reshape the outputs
        hand_params = hand_params.view(x.shape[0], -1, self.hparams.mano_params)

        hand_joints = get_mano_joints(
            hand_params,
            self.mano_left,
            self.sixd,
        )

        # Project 3D joints to 2D using predicted translation
        mano_trans = hand_params[..., :3].unsqueeze(2)

        # Predict reverse depth
        mano_trans[..., 2] = self.hparams.focal_length / (mano_trans[..., 2] + 1e-9)

        # Scale to milimeters
        mano_trans = mano_trans * 1000
        hand_joints_2d = project_joints_to_2d(
            hand_joints + mano_trans,
            self.intrinsic_matrix,
        )

        return hand_params, hand_joints, hand_joints_2d

    def loss_function(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        pred_hand_joints: torch.Tensor,
        true_hand_joints: torch.Tensor,
        pred_keypoints_2d: torch.Tensor,
        true_keypoints_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the loss for the model.

        The loss function consists of three components:
        1. Pose loss: MSE between predicted and ground truth pose parameters (first 45 values).
        2. Shape loss: MSE between predicted and ground truth shape parameters (last 10 values).
        3. Keypoints loss: L1 loss between predicted and ground truth 3D hand joints.

        The total loss is the sum of these three components.

        Args:
            y_pred (torch.Tensor): Predicted MANO parameters.
            y_true (torch.Tensor): Ground truth MANO parameters.
            pred_hand_joints (torch.Tensor): Predicted 3D keypoints.
            true_hand_joints (torch.Tensor): Ground truth 3D keypoints.
            pred_keypoints_2d (torch.Tensor): Predicted 2D keypoints.
            true_keypoints_2d (torch.Tensor): Ground truth 2D keypoints.

        Returns:
            torch.Tensor: The computed loss.

        """
        end_global_orientation = 9 if self.sixd else 6
        global_orientation_loss = F.mse_loss(
            y_pred[..., 3:end_global_orientation],
            y_true[..., 3:end_global_orientation],
        )
        pose_loss = F.mse_loss(
            y_pred[..., end_global_orientation:-10],
            y_true[..., end_global_orientation:-10],
        )
        shape_loss = F.mse_loss(y_pred[..., -10:], y_true[..., -10:])
        keypoints_loss = F.l1_loss(pred_hand_joints, true_hand_joints)
        keypoints_2d_loss = F.l1_loss(pred_keypoints_2d, true_keypoints_2d)

        if self.hparams.loss_weights is None:
            losses = {
                "global_orientation": global_orientation_loss,
                "pose": pose_loss,
                "shape": shape_loss,
                "keypoints_3d": keypoints_loss,
                "keypoints_2d": keypoints_2d_loss,
            }
        else:
            losses = {
                "global_orientation": self.hparams.loss_weights["global_orientation"] * global_orientation_loss,
                "pose": self.hparams.loss_weights["pose"] * pose_loss,
                "shape": self.hparams.loss_weights["shape"] * shape_loss,
                "keypoints_3d": self.hparams.loss_weights["keypoints_3d"] * keypoints_loss,
                "keypoints_2d": self.hparams.loss_weights["keypoints_2d"] * keypoints_2d_loss,
            }

        losses["loss"] = sum(losses.values())
        return losses

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> torch.Tensor:
        """Training step of the VideoMANORegressor model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing:
                - Input video frames (B, T, C, H, W)
                - Target MANO parameters (B, T, 61)
                - Target 3D joints (B, T, J, 3)
                - Target 2D joints (B, T, J, 2)
            batch_idx (int): Index of the batch.

        Returns:
            torch.Tensor: Loss value.

        """
        x, y, true_hand_joints, true_keypoints_2d = batch

        if self.sixd:
            # convert to 6D pose
            y = mano_to_sixd(y)

        # Ensure input is in the correct format for MViT (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]

        y_pred, pred_hand_joints, pred_keypoints_2d = self(x)

        losses = self.loss_function(
            y_pred,
            y,
            pred_hand_joints,
            true_hand_joints,
            pred_keypoints_2d,
            true_keypoints_2d,
        )
        loss = losses["loss"]

        # Additional metrics
        mje = torch.linalg.vector_norm(pred_hand_joints - true_hand_joints, dim=-1).mean(dim=-1).mean()
        mje_2d = torch.linalg.vector_norm(pred_keypoints_2d - true_keypoints_2d, dim=-1).mean(dim=-1).mean()
        mse = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)
        pamje = reconstruction_error(pred_hand_joints, true_hand_joints)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/mean_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_mje", mje, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_mje_2d", mje_2d, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/mean_pamje", pamje, on_step=False, on_epoch=True, sync_dist=True)
        for key, value in losses.items():
            self.log(f"train/losses/{key}", value, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,  # noqa: ARG002
    ) -> None:
        """Validation step of the VideoMANORegressor model.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing:
                - Input video frames (B, T, C, H, W)
                - Target MANO parameters (B, T, 61)
                - Target 3D joints (B, T, J, 3)
                - Target 2D joints (B, T, J, 2)
            batch_idx (int): Index of the batch.

        """
        x, y, true_hand_joints, true_keypoints_2d = batch

        if self.sixd:
            # convert to 6D pose
            y = mano_to_sixd(y)

        # Ensure input is in the correct format for MViT (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]

        y_pred, pred_hand_joints, pred_keypoints_2d = self(x)

        losses = self.loss_function(
            y_pred,
            y,
            pred_hand_joints,
            true_hand_joints,
            pred_keypoints_2d,
            true_keypoints_2d,
        )
        loss = losses["loss"]
        # Additional metrics
        mje = torch.linalg.vector_norm(pred_hand_joints - true_hand_joints, dim=-1).mean(dim=-1).mean()
        mje_2d = torch.linalg.vector_norm(pred_keypoints_2d - true_keypoints_2d, dim=-1).mean(dim=-1).mean()
        mse = F.mse_loss(y_pred, y)
        mae = F.l1_loss(y_pred, y)
        pamje = reconstruction_error(pred_hand_joints, true_hand_joints)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val/mean_mse", mse, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_mae", mae, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_mje", mje, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_mje_2d", mje_2d, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val/mean_pamje", pamje, on_step=False, on_epoch=True, sync_dist=True)
        for key, value in losses.items():
            self.log(f"val/losses/{key}", value, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the VideoMANORegressor model.

        Returns:
            torch.optim.Optimizer: The Adam optimizer instance.

        """
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
