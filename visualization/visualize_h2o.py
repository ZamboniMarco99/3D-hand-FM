from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from data.h2o_datamodule import H2ODataModule
from models.video_mano_regressor import VideoMANORegressor
from visualization.mano_renderer import ManoRenderer


def get_mano_dict(mano_params_left: torch.Tensor, mano_params_right: torch.Tensor) -> dict[str, torch.Tensor]:
    """Convert MANO parameters for both hands to a dictionary format.

    Args:
        mano_params_left (torch.Tensor): Tensor containing MANO parameters for the left hand.
            Expected shape: (batch_size, 61) where 61 = 3 (translation) + 45 (pose) + 10 (shape).
        mano_params_right (torch.Tensor): Tensor containing MANO parameters for the right hand.
            Expected shape: (batch_size, 61) where 61 = 3 (translation) + 45 (pose) + 10 (shape).

    Returns:
        dict: A dictionary containing MANO parameters for both hands with keys:
            - "left_pose": Left hand pose parameters (45-dimensional)
            - "left_shape": Left hand shape parameters (10-dimensional)
            - "left_tran": Left hand translation parameters (3-dimensional)
            - "right_pose": Right hand pose parameters (45-dimensional)
            - "right_shape": Right hand shape parameters (10-dimensional)
            - "right_tran": Right hand translation parameters (3-dimensional)

    """
    return {
        "left_tran": mano_params_left[:, :3],
        "left_pose": mano_params_left[:, 3:51],
        "left_shape": mano_params_left[:, 51:],
        "right_tran": mano_params_right[:, :3],
        "right_pose": mano_params_right[:, 3:51],
        "right_shape": mano_params_right[:, 51:],
    }


@hydra.main(config_path="configs", config_name="visualize_h20.yaml", version_base="1.1")
def main(cfg: DictConfig) -> None:
    # Initialize wandb
    # Load checkpoint from the ckpts directory
    checkpoint_path = Path("ckpts") / cfg.checkpoint
    if not checkpoint_path.exists():
        msg = f"Checkpoint file not found: {checkpoint_path}"
        raise FileNotFoundError(msg)

    model = VideoMANORegressor.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Initialize the H2O datamodule
    datamodule = H2ODataModule(
        dataset_prefix=cfg.data.dataset_prefix,
        cameras=cfg.data.cameras,
        max_width=cfg.data.max_width,
        max_height=cfg.data.max_height,
        num_frames=cfg.data.num_frames,
        batch_size=1,  # We'll process one sample at a time
        num_workers=cfg.data.loader.num_workers,
        crop=cfg.data.pretrained,
    )
    datamodule.setup(stage="validate")
    val_dataloader = datamodule.val_dataloader()

    # Initialize the ManoRenderer
    renderer = ManoRenderer(image_size=(cfg.data.max_width, cfg.data.max_height))

    # Create a directory to save the images
    save_dir = Path("visualization_results")
    save_dir.mkdir(exist_ok=True)

    # Process validation data
    for batch_idx, batch in enumerate(tqdm(val_dataloader)):
        clip, mano_left, mano_right, intrinsics = batch
        clip = clip.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W] -> [B, C, T, H, W]
        intrinsics = intrinsics[0]

        # Create directories for this clip
        pred_clip_dir = save_dir / "predictions" / f"clip_{batch_idx}"
        gt_clip_dir = save_dir / "ground_truth" / f"clip_{batch_idx}"
        pred_clip_dir.mkdir(parents=True, exist_ok=True)
        gt_clip_dir.mkdir(parents=True, exist_ok=True)

        # Get model predictions
        with torch.no_grad():
            y_pred_left, y_pred_right = model(clip)

        y_pred_left = y_pred_left.to("cpu")
        y_pred_right = y_pred_right.to("cpu")

        # Get the first (and only) item in the batch
        sample_clip = clip[0]

        width, height = cfg.data.max_width, cfg.data.max_height

        # Set camera intrinsics for the renderer
        renderer.set_camera_intrinsics(intrinsics, width, height)

        # Visualize the results for each frame in the clip
        for frame_idx in range(sample_clip.shape[1]):  # Iterate over frames
            # Get the predicted and ground truth MANO joints for the current frame
            pred_mano_dict = get_mano_dict(y_pred_left[:, frame_idx], y_pred_right[:, frame_idx])
            gt_mano_dict = get_mano_dict(mano_left[:, frame_idx], mano_right[:, frame_idx])

            # Project 3D keypoints to 2D for predictions and ground truth
            frame_pred_mano_joints = renderer.project_points(pred_mano_dict, intrinsics)
            frame_gt_mano_joints = renderer.project_points(gt_mano_dict, intrinsics)

            frame = sample_clip[frame_idx].permute(1, 2, 0).cpu().numpy()
            frame = (frame - frame.min()) / (frame.max() - frame.min())
            frame = (frame * 255).astype(np.uint8)

            # Visualize the results for predictions and ground truth
            for vis_type, mano_joints, save_dir in [
                ("Predicted", frame_pred_mano_joints, pred_clip_dir),
                ("Ground Truth", frame_gt_mano_joints, gt_clip_dir),
            ]:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

                # Original image
                ax1.imshow(frame)
                ax1.set_title(f"Original Image (Frame {frame_idx + 1})")

                # Rendered image with keypoints
                ax2.imshow(np.ones((height, width, 3)))  # White background
                ax2.set_title(f"{vis_type} Projected Hand Joints (Frame {frame_idx + 1})")

                for side in ["left", "right"]:
                    if f"{side}_keypoints_2d" in mano_joints:
                        keypoints_2d = mano_joints[f"{side}_keypoints_2d"]
                        color = "r" if side == "left" else "b"
                        ax2.scatter(
                            keypoints_2d[:, 0],
                            keypoints_2d[:, 1],
                            c=color,
                            s=5,
                            label=f"{side.capitalize()} Hand",
                        )

                ax2.legend()
                ax2.set_xlim(0, width)
                ax2.set_ylim(height, 0)  # Invert y-axis to match image coordinates

                plt.tight_layout()

                # Save the figure in the appropriate directory
                save_path = save_dir / f"frame_{frame_idx + 1}.png"
                fig.savefig(save_path)

                plt.close(fig)


if __name__ == "__main__":
    main()
