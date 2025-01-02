"""Utility functions for the models.

This module contains utility functions that are used across different models
in the project. These functions provide common operations and transformations
that are helpful in processing data, executing MANO models, and other related tasks.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from manopth.manolayer import ManoLayer
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle


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

    if from_sixd:
        mano_params = sixd_to_mano(mano_params)

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
    from_sixd: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the MANO model to generate hand joints.

    Args:
        mano_params_left (torch.Tensor): Tensor containing MANO parameters for the left hand.
            Expected shape: (batch_size, 61) where 61 = 3 (translation) + 45 (pose) + 10 (shape).
        mano_params_right (torch.Tensor): Tensor containing MANO parameters for the right hand.
            Expected shape: (batch_size, 61) where 61 = 3 (translation) + 45 (pose) + 10 (shape).
        mano_left (ManoLayer): MANO model for the left hand.
        mano_right (ManoLayer): MANO model for the right hand.
        from_sixd (bool): If True, the input parameters are in 6D representation. Default: False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Left hand joints: Tensor of shape (batch_size, num_joints, 3)
            - Right hand joints: Tensor of shape (batch_size, num_joints, 3)

    """
    # Get the device of mano_params
    batch_size = mano_params_left.shape[0]
    num_frames = mano_params_left.shape[1]

    if from_sixd:
        mano_params_left = sixd_to_mano(mano_params_left)
        mano_params_right = sixd_to_mano(mano_params_right)

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


def mano_to_sixd(x: torch.Tensor) -> torch.Tensor:
    """Convert a 61=3+10+48 element MANO pose parameter to a 109=3+10+96  6D representation.

    Args:
        x (torch.Tensor): Input tensor with shape (..., 61)

    Returns:
        torch.Tensor: Output tensor with shape (..., 109)

    """
    translation = x[..., :3]
    pose = x[..., 3:-10]
    shape = x[..., -10:]

    # Convert pose to 6D representation
    pose_6d = axisang_to_sixd(pose)

    # Concatenate translation, shape, and pose_6d
    return torch.cat([translation, pose_6d, shape], dim=-1)


def sixd_to_mano(x: torch.Tensor) -> torch.Tensor:
    """Convert a 109=3+10+96 element 6D representation to a 61=3+10+48 element MANO pose parameter.

    Args:
        x (torch.Tensor): Input tensor with shape (..., 109)

    Returns:
        torch.Tensor: Output tensor with shape (..., 61)

    """
    # Split the input tensor into translation, shape, and pose_6d parameters
    translation = x[..., :3]
    pose_6d = x[..., 3:-10]
    shape = x[..., -10:]

    # Convert pose_6d to axis-angle representation
    pose = sixd_to_axisang(pose_6d)

    # Concatenate translation, pose, and shape
    return torch.cat([translation, pose, shape], dim=-1)


def sixd_to_axisang(x: torch.Tensor) -> torch.Tensor:
    """Convert a 6D representation to an axis-angle representation.

    We use a Gram-Schmidt-like process.

    Input: (..., n*6) tensor
    Output: (..., n*3) tensor.
    """
    dims = x.shape
    if x.shape[-1] % 6 != 0:
        msg = f"Last dimension must be a multiple of 6. Got {x.shape[-1]}."
        raise ValueError(msg)

    # Reshape (..., n*6) to (-1, 6)
    x = x.reshape(-1, 6)

    # Convert 6D to rotation matrix using Gram-Schmidt-like process
    b1 = x[..., :3]
    b2 = x[..., 3:]
    b1 = F.normalize(b1, dim=-1)
    dot_b1_b2 = torch.sum(b1 * b2, dim=-1, keepdim=True)
    b2 = b2 - dot_b1_b2 * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    # Form the rotation matrix by stacking b1, b2, b3 as rows
    rotmat = torch.stack([b1, b2, b3], dim=-2)  # Shape (-1, 3, 3)

    # Convert rotation matrix to axis-angle
    axisang = matrix_to_axis_angle(rotmat)  # Shape (-1, 3)

    # Reshape back to (..., n*3)
    return axisang.reshape(*dims[:-1], dims[-1] // 2)


def axisang_to_sixd(x: torch.Tensor) -> torch.Tensor:
    """Convert an axis-angle representation to a 6D representation.

    Input: (..., n*3) tensor
    Output: (..., n*6) tensor.
    """
    dims = x.shape
    if x.shape[-1] % 3 != 0:
        msg = f"Last dimension must be a multiple of 3. Got {x.shape[-1]}."
        raise ValueError(msg)

    # Reshape (..., n*3) to (-1, 3)
    x = x.reshape(-1, 3)

    # Convert axis-angle to rotation matrix
    rotmat = axis_angle_to_matrix(x)  # Shape (-1, 3, 3)

    # take first two rows of rotation matrix
    sixd = rotmat[..., :2, :]  # Shape (-1, 2, 3)

    # Reshape back to original dimensions (..., n*6)
    return sixd.reshape(*dims[:-1], dims[-1] * 2)


def test_sixd_conversion() -> None:
    """Test the conversion between 6D and axis-angle representations.

    Also test if gradients are propagated correctly.
    """
    tests = [
        torch.randn(6),
        torch.randn(10, 100, 100, 96),
        torch.randn(3, 4, 5, 2, 6),
    ]
    for test in tests:
        print(f"{test.shape}: {loop_consistency_test(test)}")

    # test if the mano conversion is consistent
    x = torch.randn(1, 61)
    y = mano_to_sixd(x)
    x_ = sixd_to_mano(y)
    print("Mano Conversion works: ", torch.allclose(x, x_, atol=1e-5))

    # test to see if gradients are propagated correctly
    # the reversed process gets stuck in a local minimum
    x = torch.randn(1, 3, requires_grad=True)
    print("Input tensor:", x)
    x_ = torch.randn(1, 3, requires_grad=True)
    y = axisang_to_sixd(x).detach()

    optimizer = torch.optim.SGD([x_], lr=0.1)  # SGD optimizer with learning rate 0.01

    num_iterations = 500
    for i in range(num_iterations):
        optimizer.zero_grad()
        x6 = axisang_to_sixd(x_)  # dummy operation
        loss = torch.mean((y - x6) ** 2)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.item()}")

    print("Final optimized input_tensor:", x_)


def loop_consistency_test(x6: torch.Tensor) -> tuple[bool, torch.Tensor]:
    """Test the consistency of the conversion functions.

    Args:
        x6 (torch.Tensor): Input tensor with shape (..., n*6)

    Returns:
        tuple[bool, torch.Tensor]: A tuple containing:
            - A boolean indicating if the conversion is consistent.
    - The maximum error in the conversion.

    """
    x3 = sixd_to_axisang(x6)
    x6_ = axisang_to_sixd(x3)
    x3_ = sixd_to_axisang(x6_)
    x6__ = axisang_to_sixd(x3_)
    error = torch.reshape(x6_ - x6__, (1, -1))
    return torch.allclose(x6_, x6__, atol=1e-5), torch.max(torch.abs(error))
