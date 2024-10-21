import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from data.h2o_datamodule import H2ODataModule
from models.utils import get_mano_joints
from models.video_mano_regressor import VideoMANORegressor
from visualization.mano_renderer import ManoRenderer


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

        # Create directories for this clip
        pred_clip_dir = save_dir / "predictions" / f"clip_{batch_idx}"
        gt_clip_dir = save_dir / "ground_truth" / f"clip_{batch_idx}"
        pred_clip_dir.mkdir(parents=True, exist_ok=True)
        gt_clip_dir.mkdir(parents=True, exist_ok=True)

        # Get model predictions
        with torch.no_grad():
            y_pred_left, y_pred_right = model(clip)

        # Get MANO joints for predictions and ground truth
        pred_left_hand_joints, pred_right_hand_joints = get_mano_joints(
            y_pred_left,
            y_pred_right,
            mano_root=os.environ.get("MANO_ROOT"),
        )
        gt_left_hand_joints, gt_right_hand_joints = get_mano_joints(
            mano_left,
            mano_right,
            mano_root=os.environ.get("MANO_ROOT"),
        )

        # Get the first (and only) item in the batch
        sample_clip = clip[0]
        sample_pred_left_joints = pred_left_joints[0].cpu().numpy()
        sample_pred_right_joints = pred_right_joints[0].cpu().numpy()
        sample_gt_left_joints = gt_left_joints[0].cpu().numpy()
        sample_gt_right_joints = gt_right_joints[0].cpu().numpy()

        # Load camera intrinsics (assuming a default intrinsic matrix for visualization)
        # In a real scenario, you would need to get this information from your dataset
        intrinsic_matrix = np.array([[1000, 0, cfg.data.max_width / 2], [0, 1000, cfg.data.max_height / 2], [0, 0, 1]])
        width, height = cfg.data.max_width, cfg.data.max_height

        # Set camera intrinsics for the renderer
        renderer.set_camera_intrinsics(intrinsic_matrix, width, height)

        # Visualize the results for each frame in the clip
        for frame_idx in range(sample_clip.shape[1]):  # Iterate over frames
            # Get the predicted and ground truth MANO joints for the current frame
            frame_pred_left_joints = sample_pred_left_joints[frame_idx]
            frame_pred_right_joints = sample_pred_right_joints[frame_idx]
            frame_gt_left_joints = sample_gt_left_joints[frame_idx]
            frame_gt_right_joints = sample_gt_right_joints[frame_idx]

            # Combine left and right MANO joints for predictions and ground truth
            frame_pred_mano_joints = {
                "left_keypoints_3d": frame_pred_left_joints,
                "right_keypoints_3d": frame_pred_right_joints,
            }
            frame_gt_mano_joints = {
                "left_keypoints_3d": frame_gt_left_joints,
                "right_keypoints_3d": frame_gt_right_joints,
            }

            # Project 3D keypoints to 2D for predictions and ground truth
            frame_pred_mano_joints = renderer.project_points(frame_pred_mano_joints, intrinsic_matrix)
            frame_gt_mano_joints = renderer.project_points(frame_gt_mano_joints, intrinsic_matrix)

            # Visualize the results for predictions and ground truth
            for vis_type, mano_joints, save_dir in [
                ("Predicted", frame_pred_mano_joints, pred_clip_dir),
                ("Ground Truth", frame_gt_mano_joints, gt_clip_dir),
            ]:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

                # Original image
                ax1.imshow(sample_clip[frame_idx].permute(1, 2, 0).cpu().numpy())
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
