"""Video data augmentation transforms.

This module provides a collection of video-specific data augmentation transforms
that can be applied consistently across all frames in a video sequence. Each transform
preserves the corresponding MANO hand pose parameters and camera intrinsics.

The transforms are designed to work with video tensors of shape (T, C, H, W) or
(N, T, C, H, W), where:
- N is the batch size (optional)
- T is the number of frames
- C is the number of channels
- H is the height
- W is the width

Each transform takes as input:
- A video tensor
- Left hand MANO parameters tensor of shape (T, 61)
- Right hand MANO parameters tensor of shape (T, 61)
- Optional camera intrinsic matrix tensor of shape (3, 3)

And returns the transformed versions of these tensors while maintaining consistency
between the video frames and hand pose parameters.
"""

import math
import random

import torch
import torchvision.transforms.functional as F  # noqa: N812
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode


class VideoColorJitter(nn.Module):
    """Apply color jitter to video frames consistently.

    Args:
        brightness (float, optional): How much to jitter brightness. Defaults to 0.4.
        contrast (float, optional): How much to jitter contrast. Defaults to 0.4.
        saturation (float, optional): How much to jitter saturation. Defaults to 0.4.
        hue (float, optional): How much to jitter hue. Defaults to 0.1.

    """

    def __init__(
        self,
        brightness: float = 0.4,
        contrast: float = 0.4,
        saturation: float = 0.4,
        hue: float = 0.1,
    ) -> None:
        """Initialize the VideoColorJitter transform.

        Args:
            brightness (float, optional): Maximum brightness adjustment factor.
                Values > 1 increase brightness, values < 1 decrease it. Defaults to 0.4.
            contrast (float, optional): Maximum contrast adjustment factor.
                Values > 1 increase contrast, values < 1 decrease it. Defaults to 0.4.
            saturation (float, optional): Maximum saturation adjustment factor.
                Values > 1 increase saturation, values < 1 decrease it. Defaults to 0.4.
            hue (float, optional): Maximum hue adjustment factor in the range [-0.5, 0.5].
                Positive values rotate hue clockwise, negative counterclockwise. Defaults to 0.1.

        """
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def forward(
        self,
        video: Tensor,
        mano_left: Tensor,
        mano_right: Tensor,
        intrinsic_matrix: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Forward pass of the VideoColorJitter transform.

        Args:
            video (Tensor): Video tensor of shape (T, C, H, W) or (N, T, C, H, W)
            mano_left (Tensor): Left hand MANO parameters with shape (T, 61)
                containing translation, pose and shape parameters
            mano_right (Tensor): Right hand MANO parameters with shape (T, 61)
                containing translation, pose and shape parameters
            intrinsic_matrix (Tensor, optional): Camera intrinsic matrix with shape (3, 3)

        Returns:
            tuple: A tuple containing:
                - video (Tensor): Color jittered video tensor
                - mano_left (Tensor): Unchanged left hand MANO parameters
                - mano_right (Tensor): Unchanged right hand MANO parameters
                - intrinsic_matrix (Tensor, optional): Unchanged camera intrinsic matrix

        """
        need_squeeze = False
        if video.ndim == 4:  # noqa: PLR2004
            video = video.unsqueeze(0)
            need_squeeze = True

        # Apply same transform across all frames
        b = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
        c = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
        s = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        h = random.uniform(-self.hue, self.hue)

        N, T, C, H, W = video.shape  # noqa: N806
        video = video.reshape(-1, C, H, W)  # (N*T,C,H,W)

        # Apply color transforms
        video = F.adjust_brightness(video, b)
        video = F.adjust_contrast(video, c)
        video = F.adjust_saturation(video, s)
        video = F.adjust_hue(video, h)

        video = video.reshape(N, T, C, H, W)

        if need_squeeze:
            video = video.squeeze(0)

        return video, mano_left, mano_right, intrinsic_matrix


class VideoRandomRotation(nn.Module):
    """Randomly rotate video and MANO parameters.

    Args:
        degrees (float, optional): Maximum rotation angle in degrees. Defaults to 30.

    """

    def __init__(self, degrees: float = 30) -> None:
        """Initialize the RandomRotation transform.

        Args:
            degrees (float, optional): Maximum rotation angle in degrees. Defaults to 30.

        """
        super().__init__()
        self.degrees = degrees

    def forward(
        self,
        video: Tensor,
        mano_left: Tensor,
        mano_right: Tensor,
        intrinsic_matrix: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Forward pass of the RandomRotation transform.

        Args:
            video (Tensor): Video tensor of shape (T,C,H,W) or (N,T,C,H,W)
            mano_left (Tensor): Left hand MANO parameters with shape (T, 61)
                containing translation, pose and shape parameters
            mano_right (Tensor): Right hand MANO parameters with shape (T, 61)
                containing translation, pose and shape parameters
            intrinsic_matrix (Tensor, optional): Camera intrinsic matrix with shape (3, 3)

        Returns:
            tuple: A tuple containing:
                - video (Tensor): Rotated video tensor
                - mano_left (Tensor): Rotated left hand MANO parameters
                - mano_right (Tensor): Rotated right hand MANO parameters
                - intrinsic_matrix (Tensor, optional): Transformed camera intrinsic matrix

        """
        angle = random.uniform(-self.degrees, self.degrees)
        rad = math.radians(angle)

        # Rotate video in opposite direction (-angle) to match MANO rotation
        need_squeeze = False
        if video.ndim == 4:  # noqa: PLR2004
            video = video.unsqueeze(0)
            need_squeeze = True

        N, T, C, H, W = video.shape  # noqa: N806
        video = video.reshape(-1, C, H, W)  # (N*T,C,H,W)

        video = F.rotate(video, -angle, interpolation=InterpolationMode.BILINEAR)

        video = video.reshape(N, T, C, H, W)

        if need_squeeze:
            video = video.squeeze(0)

        # Rotate MANO pose parameters
        rot_mat = torch.tensor(
            [
                [math.cos(rad), -math.sin(rad), 0],
                [math.sin(rad), math.cos(rad), 0],
                [0, 0, 1],
            ],
            device=video.device,
        )

        # Transform intrinsic matrix if provided
        if intrinsic_matrix is not None:
            # Create transformation matrix for image center translation and rotation
            h, w = H, W
            center_x, center_y = w / 2, h / 2

            # Matrix to move image center to origin
            translate_to_origin = torch.tensor(
                [
                    [1, 0, -center_x],
                    [0, 1, -center_y],
                    [0, 0, 1],
                ],
                device=video.device,
                dtype=torch.float32,
            )

            # Matrix to move back from origin
            translate_from_origin = torch.tensor(
                [
                    [1, 0, center_x],
                    [0, 1, center_y],
                    [0, 0, 1],
                ],
                device=video.device,
                dtype=torch.float32,
            )
            intrinsic_matrix = intrinsic_matrix.to(torch.float32)

            # Apply transformation: translate_from_origin @ rot_mat @ translate_to_origin @ intrinsic_matrix
            intrinsic_matrix = torch.matmul(
                translate_from_origin,
                torch.matmul(rot_mat, torch.matmul(translate_to_origin, intrinsic_matrix)),
            )

        return video, mano_left, mano_right, intrinsic_matrix
