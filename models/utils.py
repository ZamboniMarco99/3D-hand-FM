"""Utility functions for the models.

This module contains utility functions that are used across different models
in the project. These functions provide common operations and transformations
that are helpful in processing data, executing MANO models, and other related tasks.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from manopth.manolayer import ManoLayer


def get_mano_joints(
    mano_params: torch.Tensor,
    mano: ManoLayer,
    from_sixd: bool = False,
) -> torch.Tensor:
    """Execute the MANO model to generate hand joints.

    Args:
        mano_params (torch.Tensor): Tensor containing MANO parameters for the hand.
            Expected shape: (batch_size, 61) where 61 = 3 (translation) + 45 (pose) + 10 (shape).
        mano (ManoLayer): MANO model for the hand.
        from_sixd (bool): If True, the input parameters are in 6D representation. Default: False.

    Returns:
        torch.Tensor: Tensor of shape (batch_size, num_joints, 3) of hand joints without translation

    """
    # Get the device of mano_params
    batch_size = mano_params.shape[0]
    num_frames = mano_params.shape[1]

    # Push the time dimension in the batch dimension
    params = mano_params.view(-1, mano_params.shape[2])

    pose = params[..., 3:-10]
    shape = params[..., -10:]
    if from_sixd:
        pose = sixd_to_rotmat(pose)

    # Process hand without translation
    _, hand_joints = mano(
        pose,  # pose
        shape,  # shape

    )

    # Reshape the joints to match the original batch size and time dimension
    return hand_joints.view(batch_size, num_frames, -1, 3)


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

    Credit: Hamer

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


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.

    Credit: Hamer

    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:,0], norm_quat[:,1], norm_quat[:,2], norm_quat[:,3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Works by first converting it to a quaternion.

    Credit: Hamer

    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p = 2, dim = 1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim = 1)
    return quat_to_rotmat(quat)

def aa_to_sixd(x: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle representation to 6D rotation representation.
    
    Args:
        x (torch.Tensor): Axis-angle tensor of shape (B, J*3) or (B, T, J*3)
            where B=batch size, T=sequence length, J=number of joints
            
    Returns:
        torch.Tensor: 6D rotation representation with shape (B, J*6) or (B, T, J*6)
    """
    orig_shape = x.shape
    if len(orig_shape) == 3:  # (B, T, J*3)
        x = x.view(-1, orig_shape[-1])  # Combine B and T dimensions
        
    # Reshape to (B*J, 3) or (B*T*J, 3)
    x = x.reshape(-1, 3)
        
    # Convert to rotation matrices (B*J, 3, 3) or (B*T*J, 3, 3)
    rot_mats = aa_to_rotmat(x)
    
    # Extract first two columns to get 6D representation
    sixd = rot_mats[..., :, :2].reshape(-1, 6)
    
    if len(orig_shape) == 3:
        # Restore B, T dimensions
        sixd = sixd.reshape(orig_shape[0], orig_shape[1], -1)
    else:
        # Restore B dimension
        sixd = sixd.reshape(orig_shape[0], -1)
        
    return sixd




def sixd_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """
    Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019

    Credit: Hamer
    Args:
        x (torch.Tensor): 6D rotation representation with shape (B, J*6) or (B, T, J*6)
            where B=batch size, T=sequence length, J=number of joints
    Returns:
        torch.Tensor: Batch of corresponding rotation matrices with shape (B, J, 3, 3) or (B, T, J, 3, 3)
    """
    orig_shape = x.shape
    
    # Handle both (B, J*6) and (B, T, J*6) inputs
    if len(orig_shape) == 3:  # (B, T, J*6)
        x = x.reshape(-1, orig_shape[-1])  # Combine B and T dimensions
        
    # Reshape to (B*J, 6) or (B*T*J, 6)
    x = x.reshape(-1, 6)
    
    # Convert to rotation matrices
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    rot_mats = torch.stack((b1, b2, b3), dim=-1)
    
    # Reshape back to original dimensions
    if len(orig_shape) == 3:
        # Restore B, T, J dimensions for (B, T, J*6) input
        rot_mats = rot_mats.reshape(orig_shape[0], orig_shape[1], -1, 3, 3)
    else:
        # Restore B, J dimensions for (B, J*6) input
        rot_mats = rot_mats.reshape(orig_shape[0], -1, 3, 3)
        
    return rot_mats

