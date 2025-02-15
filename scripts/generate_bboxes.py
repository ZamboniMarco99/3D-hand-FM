import argparse
import contextlib
import os
from pathlib import Path

import hamer
import imageio.v3 as iio
import numpy as np
from detectron2.config import LazyConfig
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from vitpose_model import ViTPoseModel

# Type aliases
BBox = list[float]  # [x_min, y_min, x_max, y_max]


@contextlib.contextmanager
def temporary_cwd_context(x: Path):
    """Temporarily change our working directory."""
    d = os.getcwd()
    os.chdir(x)
    try:
        yield
    finally:
        os.chdir(d)


def generate_bboxes(image: np.ndarray) -> tuple[BBox, BBox]:
    """Generate bounding boxes for left and right hands in an image.

    Args:
        image (np.ndarray): Input image in RGB format

    Returns:
        Tuple[BBox, BBox]: Left and right hand bounding boxes. Each bbox is [x_min, y_min, x_max, y_max].
                          If a hand is not detected, returns [0.0, 0.0, 0.0, 0.0].

    """
    det_out = detector(image[:, :, ::-1])
    det_instances = det_out["instances"]
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        image,
        [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
    )

    # Default bbox values if no detection
    default_bbox = [0.0, 0.0, 0.0, 0.0]
    lbbox = rbbox = None

    if not vitposes_out:
        return default_bbox, default_bbox

    # Get the pose with highest average confidence
    pose_avg_confidence = [np.mean(pose["keypoints"][:, 2]) for pose in vitposes_out]
    most_probable_pose = vitposes_out[pose_avg_confidence.index(max(pose_avg_confidence))]

    # Extract hand keypoints
    left_hand_keyp = most_probable_pose["keypoints"][-42:-21]
    right_hand_keyp = most_probable_pose["keypoints"][-21:]

    # Process left and right hand detections
    ldetect = rdetect = False

    # Left hand
    valid = left_hand_keyp[:, 2] > 0.5
    if sum(valid) > 3:
        lbbox = [
            left_hand_keyp[valid, 0].min(),
            left_hand_keyp[valid, 1].min(),
            left_hand_keyp[valid, 0].max(),
            left_hand_keyp[valid, 1].max(),
        ]
        ldetect = True

    # Right hand
    valid = right_hand_keyp[:, 2] > 0.5
    if sum(valid) > 3:
        rbbox = [
            right_hand_keyp[valid, 0].min(),
            right_hand_keyp[valid, 1].min(),
            right_hand_keyp[valid, 0].max(),
            right_hand_keyp[valid, 1].max(),
        ]
        rdetect = True

    # Handle overlapping detections
    if ldetect and rdetect:
        bboxes_dims = [
            left_hand_keyp[:, 0].max() - left_hand_keyp[:, 0].min(),
            left_hand_keyp[:, 1].max() - left_hand_keyp[:, 1].min(),
            right_hand_keyp[:, 0].max() - right_hand_keyp[:, 0].min(),
            right_hand_keyp[:, 1].max() - right_hand_keyp[:, 1].min(),
        ]
        norm_side = max(bboxes_dims)
        keyp_dist = np.sqrt(np.sum((right_hand_keyp[:, :2] - left_hand_keyp[:, :2]) ** 2, axis=1)) / norm_side

        if np.mean(keyp_dist) < 0.5:
            # Keep only the most confident detection
            if left_hand_keyp[0, 2] > right_hand_keyp[0, 2]:
                rbbox = default_bbox
            else:
                lbbox = default_bbox

    # Set default values for undetected hands
    if not ldetect:
        lbbox = default_bbox
    if not rdetect:
        rbbox = default_bbox

    return lbbox, rbbox


def process_sequence(
    h2o_path: Path,
    subject: str,
    obj: str,
) -> None:
    """Process sequences of images to generate hand bounding boxes for all clips and cameras.
    Saves results to 'predicted_bboxes.txt' in each camera directory.

    Args:
        h2o_path (Path): Base path to H2O dataset
        subject (str): Subject ID
        obj (str): Object ID

    """
    base_path = h2o_path / subject / obj
    if not base_path.exists():
        raise ValueError(f"Path does not exist: {base_path}")

    total_frames = 0
    # Process each clip in the directory
    for clip_dir in sorted(base_path.iterdir()):
        if not clip_dir.is_dir():
            continue

        clip = clip_dir.name

        # Process each camera in the clip directory
        for cam_dir in sorted(clip_dir.iterdir()):
            if not cam_dir.is_dir() or not cam_dir.name.startswith("cam"):
                continue

            cam = cam_dir.name
            sequence_path = cam_dir / "rgb"
            bbox_file = cam_dir / "predicted_bboxes.txt"

            if not sequence_path.exists():
                print(f"Skipping non-existent path: {sequence_path}")
                continue

            print(f"Processing {subject=}, {obj=}, {clip=}, {cam=}")
            frames_processed = 0

            # Create/overwrite file with bounding boxes
            with open(bbox_file, "w") as f:
                for file in sorted(sequence_path.glob("*.png")):
                    image = iio.imread(file)

                    # Convert grayscale to RGB
                    if len(image.shape) == 2:
                        image = np.stack([image] * 3, axis=-1)

                    # Handle RGBA images
                    if image.shape[-1] == 4:
                        image = image / 255.0
                        image = image[:, :, :3] * image[:, :, 3:4] + 1.0 * (1.0 - image[:, :, 3:4])
                        image = (image * 255).astype(np.uint8)

                    lbbox, rbbox = generate_bboxes(image)
                    frames_processed += 1

                    # Save bboxes to file (8 values per line: 4 for left, 4 for right)
                    bbox_line = np.concatenate([lbbox, rbbox])
                    np.savetxt(f, bbox_line[None], fmt="%.3f")

            if frames_processed > 0:
                print(f"Saved {frames_processed} frames to {bbox_file}")
                total_frames += frames_processed

    print(f"\nTotal processed frames: {total_frames}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate hand bounding boxes for H2O dataset sequences")
    parser.add_argument("--subject", required=True, help="Subject ID (e.g., 'subject2')")
    parser.add_argument("--obj", required=True, help="Object ID (e.g., 'o1')")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/cluster/project/cvg/students/mzamboni/data/H2O"),
        help="Path to H2O dataset",
    )
    return parser.parse_args()


# Initialize models
hamer_directory = Path(hamer.__file__).parent.parent

with temporary_cwd_context(hamer_directory):
    # Initialize ViTPose model
    cpm = ViTPoseModel("cuda")

    # Initialize Detectron2 model
    cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)

if __name__ == "__main__":
    h2o_path = Path("/cluster/project/cvg/students/mzamboni/data/H2O")
    args = parse_args()
    process_sequence(h2o_path, args.subject, args.obj)
