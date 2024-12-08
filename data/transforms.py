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
        contrast: float = 0.05,
        saturation: float = 0.05,
        hue: float = 0.05,
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
        mano: Tensor,
        intrinsic_matrix: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Forward pass of the VideoColorJitter transform.

        Args:
            video (Tensor): Video tensor of shape (T, C, H, W) or (N, T, C, H, W)
            mano (Tensor): MANO parameters with shape (T, 61)
                containing translation, pose and shape parameters
            intrinsic_matrix (Tensor, optional): Camera intrinsic matrix with shape (3, 3)

        Returns:
            tuple: A tuple containing:
                - video (Tensor): Color jittered video tensor
                - mano (Tensor): Unchanged MANO parameters
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

        return video, mano, intrinsic_matrix


class VideoRandomRotation(nn.Module):
    """Randomly rotate video and MANO parameters.

    Args:
        degrees (float, optional): Maximum rotation angle in degrees. Defaults to 30.

    """

    def __init__(self, degrees: float = 10) -> None:
        """Initialize the RandomRotation transform.

        Args:
            degrees (float, optional): Maximum rotation angle in degrees. Defaults to 30.

        """
        super().__init__()
        self.degrees = degrees

    def forward(
        self,
        video: Tensor,
        mano: Tensor,
        intrinsic_matrix: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Forward pass of the RandomRotation transform.

        Args:
            video (Tensor): Video tensor of shape (T,C,H,W) or (N,T,C,H,W)
            mano (Tensor): MANO parameters with shape (T, 61)
                containing translation, pose and shape parameters
            intrinsic_matrix (Tensor, optional): Camera intrinsic matrix with shape (3, 3)

        Returns:
            tuple: A tuple containing:
                - video (Tensor): Rotated video tensor
                - mano (Tensor): Rotated MANO parameters
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

        return video, mano, intrinsic_matrix


class VideoMirror(nn.Module):
    """Mirror video frames and adjust MANO parameters accordingly.

    This transform horizontally flips the video frames and updates the MANO parameters
    to maintain consistency with the mirrored view.
    """

    def __init__(self, p: float = 0.5) -> None:
        """Initialize the VideoMirror transform.

        Args:
            p (float, optional): Probability of applying the transform. Defaults to 0.5.

        """
        super().__init__()
        self.p = p

    def forward(
        self,
        video: Tensor,
        mano: Tensor,
        joints_2d: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Forward pass of the VideoMirror transform.

        Args:
            video (Tensor): Video tensor of shape (T, C, H, W) or (N, T, C, H, W)
            mano (Tensor): MANO parameters with shape (T, 61)
                containing translation, pose and shape parameters
            joints_2d (Tensor | None): 2D joint coordinates with shape (T, J, 2)

        Returns:
            tuple[Tensor, Tensor, Tensor | None]: Tuple containing:
                - Mirrored video tensor
                - Updated MANO parameters
                - Updated 2D joint coordinates (if provided)

        """
        if random.random() > self.p:
            return video, mano, joints_2d

        # Handle both batched and unbatched videos
        need_squeeze = False
        if video.dim() == 4:  # noqa: PLR2004
            video = video.unsqueeze(0)
            need_squeeze = True

        # Mirror the video frames
        video = torch.flip(video, dims=[-1])

        # Mirror the translation parameters (x coordinate)
        mano[..., 0] *= -1

        # Mirror relevant pose parameters
        #  # Negate x and z components of rotations
        mano[..., 3:51:3] *= -1
        mano[..., 5:51:3] *= -1

        # Mirror 2D joint coordinates if provided
        if joints_2d is not None:
            # Get width of video for mirroring x coordinates
            width = video.shape[-1]
            # Mirror x coordinates (first dimension of joints)
            joints_2d[..., 0] = width - joints_2d[..., 0]

        if need_squeeze:
            video = video.squeeze(0)

        return video, mano, joints_2d


class CropHand:
    """Transform to crop hand from video frames."""

    def __init__(self, output_size: int = 224, padding_factor: float = 1.2) -> None:
        """Initialize the CropHand transform.

        Args:
            output_size (int): Size of the output square crop in pixels
            padding_factor (float): Factor to increase the crop size by. For example,
                1.2 means the crop will be 20% larger than the tight bounding box.

        """
        self.output_size = output_size
        self.padding_factor = padding_factor

    def process_bbox(self, bbox: Tensor, h: int, w: int) -> tuple[int, int, int, int]:
        """Process a bounding box to generate square crop coordinates.

        This function takes a bounding box and:
        1. Calculates the center point of the box
        2. Determines the largest dimension (width or height) to make a square crop
        3. Applies padding factor to increase crop size
        4. Adjusts crop coordinates if they exceed image boundaries while maintaining square shape

        Args:
            bbox (Tensor): Bounding box coordinates [x_min, y_min, x_max, y_max]
            h (int): Height of the image
            w (int): Width of the image

        Returns:
            tuple[int, int, int, int]: Adjusted crop coordinates (x_min, y_min, x_max, y_max)
            that define a square region within image bounds

        """
        x_min, y_min, x_max, y_max = bbox

        # Calculate center and size
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        # Use larger dimension for square crop and apply padding
        size = max(width, height) * self.padding_factor
        half_size = size / 2

        # Initial crop coordinates centered on the hand
        crop_x_min = int(center_x - half_size)
        crop_y_min = int(center_y - half_size)
        crop_x_max = int(center_x + half_size)
        crop_y_max = int(center_y + half_size)

        # Adjust coordinates if they exceed image bounds
        # If crop goes beyond left edge
        if crop_x_min < 0:
            crop_x_max -= crop_x_min  # Shift right while maintaining size
            crop_x_min = 0
        # If crop goes beyond top edge
        if crop_y_min < 0:
            crop_y_max -= crop_y_min  # Shift down while maintaining size
            crop_y_min = 0
        # If crop goes beyond right edge
        if crop_x_max > w:
            crop_x_min -= crop_x_max - w  # Shift left while maintaining size
            crop_x_max = w
        # If crop goes beyond bottom edge
        if crop_y_max > h:
            crop_y_min -= crop_y_max - h  # Shift up while maintaining size
            crop_y_max = h

        return crop_x_min, crop_y_min, crop_x_max, crop_y_max

    def __call__(
        self,
        video: Tensor,
        bbox: Tensor,
        keypoints_2d: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Apply hand cropping transform.

        Args:
            video (Tensor): Video tensor of shape (T, C, H, W) or (B, T, C, H, W)
            bbox (Tensor): Hand bounding boxes for each frame
            keypoints_2d (Tensor | None): Optional 2D keypoints of shape (T, N, 2) or (B, T, N, 2)

        Returns:
            tuple[Tensor, Tensor | None]: Cropped and resized video and transformed 2D keypoints if provided

        """
        # Handle both batched and unbatched inputs
        need_squeeze = False
        if video.dim() == 4:  # noqa: PLR2004
            video = video.unsqueeze(0)
            bbox = bbox.unsqueeze(0)
            if keypoints_2d is not None:
                keypoints_2d = keypoints_2d.unsqueeze(0)
            need_squeeze = True

        # Get dimensions
        b, t, c, h, w = video.shape

        # Initialize output tensors
        video_crops = []
        keypoints_2d_updated = [] if keypoints_2d is not None else None

        # Process each batch independently
        for batch_idx in range(b):
            batch_crops = []
            batch_keypoints = [] if keypoints_2d is not None else None

            # Process each frame in the batch
            for time_idx in range(t):
                # Get crop coordinates
                crop = self.process_bbox(bbox[batch_idx, time_idx], h, w)

                # Crop and resize frame
                frame_crop = video[batch_idx, time_idx, :, crop[1] : crop[3], crop[0] : crop[2]]
                frame_resized = torch.nn.functional.interpolate(
                    frame_crop.unsqueeze(0),
                    size=(self.output_size, self.output_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                batch_crops.append(frame_resized)

                # Transform 2D keypoints if provided
                if keypoints_2d is not None:
                    keypoints_t = keypoints_2d[batch_idx, time_idx].clone()
                    # Adjust for crop offset
                    keypoints_t[..., 0] -= crop[0]
                    keypoints_t[..., 1] -= crop[1]
                    # Scale to new size
                    scale = self.output_size / (crop[2] - crop[0])
                    keypoints_t *= scale
                    batch_keypoints.append(keypoints_t)

            # Stack frames and parameters for current batch
            video_crops.append(torch.stack(batch_crops))
            if keypoints_2d is not None:
                keypoints_2d_updated.append(torch.stack(batch_keypoints))

        # Stack all batches
        video_cropped = torch.stack(video_crops)
        keypoints_2d_cropped = torch.stack(keypoints_2d_updated) if keypoints_2d is not None else None

        if need_squeeze:
            video_cropped = video_cropped.squeeze(0)
            if keypoints_2d_cropped is not None:
                keypoints_2d_cropped = keypoints_2d_cropped.squeeze(0)

        return video_cropped, keypoints_2d_cropped
