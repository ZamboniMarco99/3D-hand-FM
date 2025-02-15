"""Loss functions for training hand pose estimation models."""

import torch
from torch.nn import functional as F  # noqa: N812


def keypoint_diversity_loss(
    predictions_2d: torch.Tensor,
    hand_available: torch.Tensor,
    epsilon: float = 10.0,
) -> torch.Tensor:
    """Vectorized implementation of Keypoint Diversity Loss for 2D keypoints.

    Penalizes keypoints that are closer than epsilon distance within the same frame.

    Args:
        predictions_2d: Tensor of shape (B, T, J, 2) where B is batch size,
            T is number of frames, J is number of joints, and 2 is for x,y coordinates.
        hand_available: Tensor of shape (B, T) of floats indicating which frames have valid hands
            (1.0 for valid, 0.0 for invalid).
        epsilon: Float, minimum distance threshold for diversity.

    Returns:
        loss: Scalar tensor representing the diversity loss.

    """
    B, T, J, _ = predictions_2d.shape  # noqa: N806

    # Reshape to handle each frame independently and apply hand availability mask
    points = predictions_2d.reshape(B * T, J, 2)  # Shape: (B*T, J, 2)
    hand_available = hand_available.reshape(-1).bool()  # Shape: (B*T)

    # Compute pairwise distances between keypoints for each frame
    dists = torch.cdist(points, points, p=2)  # Shape: (B*T, J, J)

    # Create mask to ignore self-distances
    mask = ~torch.eye(J, dtype=torch.bool, device=dists.device).unsqueeze(0)  # Shape: (1, J, J)

    # Apply hand availability mask
    mask = mask & hand_available.unsqueeze(-1).unsqueeze(-1)  # Shape: (B*T, J, J)

    # Convert mask to float for multiplication
    mask = mask.float()

    # Compute diversity loss only for valid pairs (where mask is True)
    diversity_loss = F.relu(epsilon - dists) * mask  # Shape: (B*T, J, J)

    # Sum violations and normalize by number of valid pairs
    num_valid_pairs = mask.sum()
    if num_valid_pairs == 0:
        return torch.tensor(0.0, device=predictions_2d.device)

    return diversity_loss.sum() / num_valid_pairs


def temporal_diversity_loss(
    mano_params: torch.Tensor,
    hand_available: torch.Tensor,
    epsilon: float = 0.2,
) -> torch.Tensor:
    """Compute temporal diversity loss for MANO parameters.

    Penalizes when MANO parameters are too similar across consecutive frames by ensuring
    a minimum difference between frames. This encourages temporal variation and prevents
    the model from predicting static/repeated poses.

    Args:
        mano_params: Tensor of shape (B, T, P) where B is batch size,
            T is number of frames, and P is number of MANO parameters.
        hand_available: Tensor of shape (B, T) of floats indicating which frames have valid hands
            (1.0 for valid, 0.0 for invalid).
        epsilon: Float, minimum difference threshold between frames.

    Returns:
        loss: Scalar tensor representing the temporal diversity loss.

    """
    # Convert hand_available to boolean
    hand_available = hand_available.bool()

    # Compute differences between consecutive frames
    frame_diffs = mano_params[:, 1:] - mano_params[:, :-1]  # Shape: (B, T-1, P)

    # Create mask for consecutive valid frames
    valid_pairs = hand_available[:, 1:] & hand_available[:, :-1]  # Shape: (B, T-1)

    # Convert valid_pairs to float for multiplication
    valid_pairs = valid_pairs.float()

    # Compute L2 norm of differences along parameter dimension
    frame_diff_norms = torch.norm(frame_diffs, dim=-1)  # Shape: (B, T-1)

    # Penalize frames that are too similar (diff < epsilon), only for valid pairs
    diversity_loss = F.relu(epsilon - frame_diff_norms) * valid_pairs  # Shape: (B, T-1)

    # Average over valid pairs only
    num_valid_pairs = valid_pairs.sum()
    if num_valid_pairs == 0:
        return torch.tensor(0.0, device=mano_params.device)

    return diversity_loss.sum() / num_valid_pairs
