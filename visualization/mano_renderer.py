"""Minimal code for visualizing MANO poses in the H2O dataset."""

import os

import cv2
import numpy as np
import open3d as o3d
import torch
from manopth.manolayer import ManoLayer
from open3d.geometry import Image as ImageO3d

MANO_MODEL_PATH = os.environ.get("MANO_ROOT")


class ManoRenderer:
    def __init__(self, image_size=(224, 224), mano_root=MANO_MODEL_PATH):
        self.image_size = image_size
        # create offscreen renderer (not thread safe)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(*self.image_size)
        # create mano layers for reuse to avoid disk read
        # access with self.mano["right"] or self.mano["left"]
        self.mano = {
            side: ManoLayer(
                mano_root=mano_root,
                ncomps=45,
                use_pca=False,
                flat_hand_mean=True,
                side=side,
            )
            for side in ["right", "left"]
        }
        # create hand material for reuse
        self.material = o3d.visualization.rendering.MaterialRecord()  # Create a material
        self.material.base_color = [209 / 255.0, 163 / 255.0, 164 / 255.0, 1.0]
        self.material.shader = "defaultLit"
        # set background color to black
        self.renderer.scene.set_background([1, 1, 1, 1])

        # setup default camera extrinsic
        # camera looking at the origin from a distance of 1, up vector is in the negative y direction
        self.renderer.scene.camera.look_at([0, 0, 1], [0, 0, 0], [0, -1, 0])

    def set_camera_intrinsics(self, intrinsic_matrix, width, height):
        """Set camera intrinsic matrix
        :param intrinsic_matrix: 3x3 numpy array
        :param width: image width
        :param height: image height
        """
        self.renderer.scene.camera.set_projection(
            intrinsic_matrix,
            0.1,
            1000,
            width,
            height,
        )

    def project_points(self, hand_pose, intrinsic_matrix):
        """Takes hand pose and returns 2D keypoints"""
        for side in ["right", "left"]:
            # skip if a side is missing
            if f"{side}_pose" not in hand_pose:
                continue

            hand_verts, mano_keypoints_3d = self.mano[side](
                torch.tensor(hand_pose[f"{side}_pose"], dtype=torch.float32),
                torch.tensor(hand_pose[f"{side}_shape"], dtype=torch.float32),
                torch.tensor(hand_pose[f"{side}_tran"], dtype=torch.float32),
            )
            # project 3D keypoints to 2D
            keypoints_3d = mano_keypoints_3d[0].detach().cpu().numpy()
            keypoints_2d = np.dot(intrinsic_matrix, keypoints_3d.T).T
            keypoints_2d = keypoints_2d[:, :2] / keypoints_2d[:, 2:]
            hand_pose[f"{side}_keypoints_2d"] = keypoints_2d
            hand_pose[f"{side}_keypoints_3d"] = keypoints_3d
        return hand_pose

    def renderSingleHand(self, item):
        """Render a single hand pose using open3d. This method returns o3d image format.
        To get numpy array use np.asarray(image)[0]
        item must contain the following keys:
        - side: "left" or "right"
        - optionally intrinsic_matrix: 3x3 numpy array
        - optionally crop: cropped image of the hand
        - pose: 45-dim numpy array representing hand pose
        - shape: 10-dim numpy array representing hand shape
        - tran: 3-dim numpy array representing hand translation


        :param item: dictionary containing hand pose, shape and translation for a single hand
        :return: o3d.cpu.pybind.geometry.Image
        """
        side = item["side"]
        if "intrinsic_matrix" in item:
            intrinsic_matrix = item["intrinsic_matrix"]
            self.set_camera_intrinsics(intrinsic_matrix, *self.image_size)

        if "image" in item:
            image = o3d.cpu.pybind.geometry.Image(cv2.cvtColor(item["image"], cv2.COLOR_BGR2RGB))
            self.renderer.scene.set_background([1, 1, 1, 1], image=image)

        mano = self.mano[side]
        hand_verts, mano_keypoints_3d = mano(
            torch.tensor(item["pose"], dtype=torch.float32),
            torch.tensor(item["shape"], dtype=torch.float32),
            torch.tensor(item["tran"], dtype=torch.float32),
        )
        # prepare hand mesh
        hand_verts = hand_verts[0].detach().cpu().numpy()
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
        mesh.triangles = o3d.utility.Vector3iVector(mano.th_faces)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.reshape(
                [209 / 255.0, 163 / 255.0, 164 / 255.0] * np.shape(mesh.vertices)[0],
                (-1, 3),
            ),
        )
        mesh.compute_vertex_normals()
        # add hand mesh to the scene
        self.renderer.scene.add_geometry(f"{side} hand", mesh, self.material)

        # Render and remove geometry from the scene after rendering
        rendered_image = self.renderer.render_to_image()
        self.renderer.scene.clear_geometry()

        return rendered_image

    def render(self, hand_pose, frame=None, automatic_camera=False):
        """Render the hand pose using open3d. This method returns o3d image format.
        To get numpy array use np.asarray(image)[0]

        :param hand_pose: dictionary containing hand pose, shape and translation for left and right hand
        :param automatic_camera: if True, camera is set to look at the hand, if False, camera is set to look at the origin
        :param frame: background image for rendering
        :return: o3d.cpu.pybind.geometry.Image
        """
        if frame is not None:
            # if image is np array-> need to convert to o3d.cpu.pybind.geometry.Image(frame)
            if isinstance(frame, np.ndarray):
                frame = ImageO3d(frame)
            self.renderer.scene.set_background([0, 0, 0, 1], image=frame)
        else:
            self.renderer.scene.set_background([1, 1, 1, 1])

        for side in ["right", "left"]:
            mano = self.mano[side]
            # skip if a side is missing
            if f"{side}_pose" not in hand_pose:
                continue

            hand_verts, mano_keypoints_3d = mano(
                torch.tensor(hand_pose[f"{side}_pose"], dtype=torch.float32),
                torch.tensor(hand_pose[f"{side}_shape"], dtype=torch.float32),
                torch.tensor(hand_pose[f"{side}_tran"], dtype=torch.float32),
            )
            # prepare hand mesh
            hand_verts = hand_verts[0].detach().cpu().numpy()
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
            mesh.triangles = o3d.utility.Vector3iVector(mano.th_faces)
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                np.reshape(
                    [209 / 255.0, 163 / 255.0, 164 / 255.0] * np.shape(mesh.vertices)[0],
                    (-1, 3),
                ),
            )
            mesh.compute_vertex_normals()
            # add hand mesh to the scene
            self.renderer.scene.add_geometry(f"{side} hand", mesh, self.material)

        if automatic_camera:
            # get object bounding box of one hand
            bbox = mesh.get_axis_aligned_bounding_box()
            look_at_point = bbox.get_center()
            camera_position = look_at_point + [
                0,
                0,
                -max(bbox.get_extent()) * 3,
            ]  # Place the camera far enough to see the entire object
            up_vector = [0, -1, 0]  # Up direction is along the y-axis
            self.renderer.scene.camera.look_at(
                look_at_point,
                camera_position,
                up_vector,
            )

        # Render and remove geometry from the scene after rendering
        rendered_image = self.renderer.render_to_image()
        self.renderer.scene.clear_geometry()

        return rendered_image


def load_cam_intrinsics(file):
    """Load camera intrinsics from file"""
    [fx, fy, cx, cy, w, h] = np.loadtxt(file)
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), int(w), int(h)


def load_cam_extrinsics(file):
    """Load camera extrinsics from file"""
    return np.loadtxt(file).reshape((4, 4))


def load_hand_pose(file):
    """Load hand pose from file"""
    load_hand = np.loadtxt(file)
    hand_pose = {
        "left_tran": np.expand_dims(load_hand[1:4], 0),
        "left_pose": np.expand_dims(load_hand[4:52], 0),
        "left_shape": np.expand_dims(load_hand[52:62], 0),
        "right_tran": np.expand_dims(load_hand[63:66], 0),
        "right_pose": np.expand_dims(load_hand[66:114], 0),
        "right_shape": np.expand_dims(load_hand[114:124], 0),
        "success": load_hand[0] == 1,
    }

    return hand_pose
