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
