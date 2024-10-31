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
