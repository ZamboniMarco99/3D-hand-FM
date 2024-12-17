import torch
from torch.nn import functional as F  # noqa: N812


def keypoint_diversity_loss(predictions_2d: torch.Tensor, epsilon: float = 10.0) -> torch.Tensor:
    """Vectorized implementation of Keypoint Diversity Loss for 2D keypoints.

    Penalizes keypoints that are closer than epsilon distance within the same frame.

    Args:
        predictions_2d: Tensor of shape (B, T, J, 2) where B is batch size,
            T is number of frames, J is number of joints, and 2 is for x,y coordinates.
        epsilon: Float, minimum distance threshold for diversity.

    Returns:
        loss: Scalar tensor representing the diversity loss.

    """
    B, T, J, _ = predictions_2d.shape  # noqa: N806

    # Reshape to handle each frame independently
    points = predictions_2d.reshape(B * T, J, 2)  # Shape: (B*T, J, 2)

    # Compute pairwise distances between keypoints for each frame
    dists = torch.cdist(points, points, p=2)  # Shape: (B*T, J, J)

    # Create mask to ignore self-distances
    mask = ~torch.eye(J, dtype=torch.bool, device=dists.device).unsqueeze(0)  # Shape: (1, J, J)

    # Compute diversity loss only for valid pairs (where mask is True)
    diversity_loss = F.relu(epsilon - dists) * mask  # Shape: (B*T, J, J)

    # Sum violations and normalize by number of valid pairs
    num_valid_pairs = mask.sum()
    return diversity_loss.sum() / (num_valid_pairs * (B * T))


def temporal_diversity_loss(mano_params: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    """Compute temporal diversity loss for MANO parameters.

    Penalizes when MANO parameters are too similar across consecutive frames by ensuring
    a minimum difference between frames. This encourages temporal variation and prevents
    the model from predicting static/repeated poses.

    Args:
        mano_params: Tensor of shape (B, T, P) where B is batch size,
            T is number of frames, and P is number of MANO parameters.
        epsilon: Float, minimum difference threshold between frames.

    Returns:
        loss: Scalar tensor representing the temporal diversity loss.

    """
    # Compute differences between consecutive frames
    frame_diffs = mano_params[:, 1:] - mano_params[:, :-1]  # Shape: (B, T-1, P)

    # Compute L2 norm of differences along parameter dimension
    frame_diff_norms = torch.norm(frame_diffs, dim=-1)  # Shape: (B, T-1)

    # Penalize frames that are too similar (diff < epsilon)
    diversity_loss = F.relu(epsilon - frame_diff_norms)  # Shape: (B, T-1)

    # Average over batch and time dimensions
    return diversity_loss.mean()
