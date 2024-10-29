"""Utility functions for the models.

This module contains utility functions that are used across different models
in the project. These functions provide common operations and transformations
that are helpful in processing data, executing MANO models, and other related tasks.
"""

import torch
from manopth.manolayer import ManoLayer


def get_mano_joints(
    mano_params_left: torch.Tensor,
    mano_params_right: torch.Tensor,
    mano_left: ManoLayer,
    mano_right: ManoLayer,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the MANO model to generate hand joints.

    Args:
        mano_params_left (torch.Tensor): Tensor containing MANO parameters for the left hand.
            Expected shape: (batch_size, 61) where 61 = 3 (translation) + 45 (pose) + 10 (shape).
        mano_params_right (torch.Tensor): Tensor containing MANO parameters for the right hand.
            Expected shape: (batch_size, 61) where 61 = 3 (translation) + 45 (pose) + 10 (shape).
        mano_left (ManoLayer): MANO model for the left hand.
        mano_right (ManoLayer): MANO model for the right hand.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Left hand joints: Tensor of shape (batch_size, num_joints, 3)
            - Right hand joints: Tensor of shape (batch_size, num_joints, 3)

    """
    # Get the device of mano_params
    batch_size = mano_params_left.shape[0]
    num_frames = mano_params_left.shape[1]

    # Push the time dimension in the batch dimension
    left_params = mano_params_left.view(-1, mano_params_left.shape[2])
    right_params = mano_params_right.view(-1, mano_params_right.shape[2])

    # Process left hand
    _, left_hand_joints = mano_left(
        left_params[:, 3:51],  # pose
        left_params[:, 51:],  # shape
        left_params[:, :3],  # translation
    )

    # Process right hand
    _, right_hand_joints = mano_right(
        right_params[:, 3:51],  # pose
        right_params[:, 51:],  # shape
        right_params[:, :3],  # translation
    )

    # Reshape the joints to match the original batch size and time dimension
    left_hand_joints = left_hand_joints.view(batch_size, num_frames, -1, 3)
    right_hand_joints = right_hand_joints.view(batch_size, num_frames, -1, 3)

    return left_hand_joints, right_hand_joints


def project_joints_to_2d(
    joints_3d: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
) -> torch.Tensor:
    """Project 3D hand joints to 2D image coordinates.

    Args:
        joints_3d (torch.Tensor): 3D joint coordinates with shape (batch_size, num_frames, num_joints, 3)
        intrinsic_matrix (torch.Tensor): Camera intrinsic matrix with shape (batch_size, 3, 3)

    Returns:
        torch.Tensor: 2D joint coordinates with shape (batch_size, num_frames, num_joints, 2)

    """
    batch_size, num_frames, num_joints, _ = joints_3d.shape

    # Project 3D keypoints to 2D for each batch
    keypoints_2d = []
    for b in range(batch_size):
        batch_keypoints = []
        # Get intrinsic matrix for this batch
        batch_intrinsics = intrinsic_matrix[b]

        # Process each frame in the batch
        for f in range(num_frames):
            # Get joints for this frame
            joints_f = joints_3d[b, f]  # Shape: (num_joints, 3)

            # Transpose joints to match matrix multiplication dimensions
            joints_f = joints_f.transpose(0, 1)  # Shape: (3, num_joints)

            # Matrix multiply with intrinsic matrix
            keypoints_2d_temp = torch.matmul(batch_intrinsics, joints_f).transpose(0, 1)

            # Divide by z coordinates for perspective projection
            batch_keypoints.append(keypoints_2d_temp[..., :2] / keypoints_2d_temp[..., 2:])

        # Stack frames for this batch
        keypoints_2d.append(torch.stack(batch_keypoints))

    # Stack all batches
    return torch.stack(keypoints_2d)  # Shape: (batch_size, num_frames, num_joints, 2)
