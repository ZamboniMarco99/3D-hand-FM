"""Utility functions for the models.

This module contains utility functions that are used across different models
in the project. These functions provide common operations and transformations
that are helpful in processing data, executing MANO models, and other related tasks.
"""

import torch
from manopth.manolayer import ManoLayer


def get_mano_joints(
    mano_params: torch.Tensor,
    mano_root: str,
    use_pca: bool = True,
    flat_hand_mean: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the MANO model to generate hand joints.

    Args:
        mano_params (torch.Tensor): Tensor containing MANO parameters for both hands concatenated.
            Expected shape: (batch_size, 122) where 122 = 61 (left hand) + 61 (right hand)
            Each 61 = 3 (translation) + 45 (pose) + 10 (shape) for each hand.
        mano_root (str): Path to the directory containing MANO model files.
        use_pca (bool, optional): Whether to use PCA for pose parameters. Defaults to True.
        flat_hand_mean (bool, optional): Whether to use flat hand mean. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Left hand joints: Tensor of shape (batch_size, num_joints, 3)
            - Right hand joints: Tensor of shape (batch_size, num_joints, 3)

    """
    # Initialize MANO layers for left and right hands
    mano_left = ManoLayer(
        mano_root=mano_root,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        side="left",
    )
    mano_right = ManoLayer(
        mano_root=mano_root,
        use_pca=use_pca,
        flat_hand_mean=flat_hand_mean,
        side="right",
    )

    # Split the parameters for left and right hands
    left_params = mano_params[:, :61]
    right_params = mano_params[:, 61:]

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

    return left_hand_joints, right_hand_joints
