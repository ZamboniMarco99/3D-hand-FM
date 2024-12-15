"""Utility functions for the models.

This module contains utility functions that are used across different models
in the project. These functions provide common operations and transformations
that are helpful in processing data, executing MANO models, and other related tasks.
"""

import torch
from manopth.manolayer import ManoLayer


def get_mano_joints(
    mano_params: torch.Tensor,
    mano: ManoLayer,
) -> torch.Tensor:
    """Execute the MANO model to generate hand joints.

    Args:
        mano_params (torch.Tensor): Tensor containing MANO parameters for the hand.
            Expected shape: (batch_size, 61) where 61 = 3 (translation) + 45 (pose) + 10 (shape).
        mano (ManoLayer): MANO model for the hand.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, num_joints, 3) of hand joints without translation

    """
    # Get the device of mano_params
    batch_size = mano_params.shape[0]
    num_frames = mano_params.shape[1]

    # Push the time dimension in the batch dimension
    params = mano_params.view(-1, mano_params.shape[2])

    # Process hand without translation
    _, hand_joints = mano(
        params[:, 3:51],  # pose
        params[:, 51:],  # shape
    )

    # Reshape the joints to match the original batch size and time dimension
    return hand_joints.view(batch_size, num_frames, -1, 3)


def get_mano_joints_both_hands(
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
        intrinsic_matrix (torch.Tensor): Camera intrinsic matrix with shape (3, 3) or (batch_size, 3, 3)

    Returns:
        torch.Tensor: 2D joint coordinates with shape (batch_size, num_frames, num_joints, 2)

    """
    batch_size, num_frames, num_joints, _ = joints_3d.shape

    # Reshape joints to combine batch and frames dimensions
    joints_3d = joints_3d.view(batch_size * num_frames, num_joints, 3)

    # Handle both single and batched intrinsic matrices
    if intrinsic_matrix.dim() == 2:  # noqa: PLR2004
        # Single intrinsic matrix - expand to match batch size
        intrinsic_matrix = intrinsic_matrix.unsqueeze(0).expand(batch_size * num_frames, -1, -1)
    else:
        # Batched intrinsic matrix - repeat for each frame
        intrinsic_matrix = intrinsic_matrix.unsqueeze(1).repeat(1, num_frames, 1, 1)
        intrinsic_matrix = intrinsic_matrix.view(batch_size * num_frames, 3, 3)

    # Transpose joints for matrix multiplication
    joints_3d = joints_3d.transpose(1, 2)  # Shape: (batch_size * num_frames, 3, num_joints)

    # Project all points at once
    projected = torch.bmm(intrinsic_matrix, joints_3d)  # Shape: (batch_size * num_frames, 3, num_joints)
    projected = projected.transpose(1, 2)  # Shape: (batch_size * num_frames, num_joints, 3)

    # Perspective division
    keypoints_2d = projected[..., :2] / projected[..., 2:3]

    # Restore batch and frames dimensions
    return keypoints_2d.view(batch_size, num_frames, num_joints, 2)


def compute_similarity_transform(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Computes a similarity transform (sR, t) in a batched way.

    It that takes a set of 3D points s1 (B, N, 3) closest to a set of 3D points s2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.

    Credit: Hamer

    Args:
        s1 (torch.Tensor): First set of points of shape (B, N, 3).
        s2 (torch.Tensor): Second set of points of shape (B, N, 3).

    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.

    """
    batch_size = s1.shape[0]
    s1 = s1.permute(0, 2, 1)
    s2 = s2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = s1.mean(dim=2, keepdim=True)
    mu2 = s2.mean(dim=2, keepdim=True)
    x1 = s1 - mu1
    x2 = s2 - mu2

    # 2. Compute variance of x1 used for scale.
    var1 = (x1**2).sum(dim=(1, 2))

    # 3. The outer product of x1 and x2.
    k = torch.matmul(x1, x2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'k) is R=U*V', where U, V are singular vectors of k.
    u, s, v = torch.svd(k)
    vh = v.permute(0, 2, 1)

    # Construct z that fixes the orientation of R to get det(R)=1.
    z = torch.eye(u.shape[1], device=u.device).unsqueeze(0).repeat(batch_size, 1, 1)
    z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(u, vh)))

    # Construct R.
    r = torch.matmul(torch.matmul(v, z), u.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(r, k).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale * torch.matmul(r, mu1)

    # 7. Error:
    s1_hat = scale * torch.matmul(r, s1) + t

    return s1_hat.permute(0, 2, 1)


def reconstruction_error(s1: torch.Tensor, s2: torch.Tensor) -> torch.Tensor:
    """Computes the mean Euclidean distance of 2 set of points s1, s2 after performing Procrustes alignment.

    Credit:

    Args:
        s1 (torch.Tensor): First set of points of shape (B, T, N, 3).
        s2 (torch.Tensor): Second set of points of shape (B, T, N, 3).

    Returns:
        (torch.Tensor): Reconstruction error.

    """
    # Reshape to (B*T, N, 3) for compute_similarity_transform
    batch_size, num_frames = s1.shape[:2]
    s1_reshaped = s1.reshape(-1, *s1.shape[2:])
    s2_reshaped = s2.reshape(-1, *s2.shape[2:])

    s1_hat = compute_similarity_transform(s1_reshaped, s2_reshaped)

    # Reshape back to (B, T, N, 3)
    s1_hat = s1_hat.reshape(batch_size, num_frames, *s1_hat.shape[1:])
    # First calculate mean per clip independently
    clip_means = torch.sqrt(((s1_hat - s2) ** 2).sum(dim=-1)).mean(dim=-1)
    # Then take mean across all clips
    return clip_means.mean()
