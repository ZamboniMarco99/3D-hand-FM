"""Visualization utilities for rendering 3D scenes.

Credit: Hamer
"""

import os

if "PYOPENGL_PLATFORM" not in os.environ:
    os.environ["PYOPENGL_PLATFORM"] = "egl"


import cv2
import numpy as np
import pyrender
import trimesh


def create_raymond_lights() -> list[pyrender.Node]:
    """Create a set of Raymond lights for scene illumination.

    Raymond lighting is a three-point lighting setup commonly used in computer graphics
    to provide balanced illumination of objects. It consists of three directional lights
    positioned at specific angles around the subject.

    Returns:
        List[pyrender.Node]: A list of directional light nodes positioned according to
            the Raymond lighting setup.

    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas, strict=False):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(
            pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
                matrix=matrix,
            ),
        )

    return nodes


class Renderer:
    """A renderer class for visualizing MANO hand meshes using pyrender.

    This class provides functionality to render 3D hand meshes on images using pyrender,
    with support for different viewing angles, mesh colors, and rendering configurations.
    """

    def __init__(self, focal_length: float, img_size: int, faces: np.ndarray) -> None:
        """Initialize the renderer with camera and mesh parameters.

        Args:
            focal_length: The focal length of the camera in pixels.
            img_size: The size of the output image (assumed square) in pixels.
            faces: Array of shape (F, 3) containing the mesh faces indices.

        """
        self.focal_length = focal_length
        self.img_res = img_size

        # add faces that make the hand mesh watertight
        faces_new = np.array(
            [
                [92, 38, 234],
                [234, 38, 239],
                [38, 122, 239],
                [239, 122, 279],
                [122, 118, 279],
                [279, 118, 215],
                [118, 117, 215],
                [215, 117, 214],
                [117, 119, 214],
                [214, 119, 121],
                [119, 120, 121],
                [121, 120, 78],
                [120, 108, 78],
                [78, 108, 79],
            ],
        )
        faces = np.concatenate([faces, faces_new], axis=0)

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces
        self.faces_left = self.faces[:, [0, 2, 1]]

    def __call__(
        self,
        vertices: np.ndarray,
        camera_translation: np.ndarray,
        image: np.ndarray,
        full_frame: bool = False,
        imgname: str | None = None,
        mesh_base_color: tuple[float, float, float] = (1.0, 1.0, 0.9),
        scene_bg_color: tuple[float, float, float] = (0, 0, 0),
    ) -> np.ndarray:
        """Render the hand mesh on an input image.

        Args:
            vertices: Array of shape (V, 3) containing the mesh vertices coordinates.
            camera_translation: Array of shape (3,) with the camera translation.
            image: Array of shape (H, W, 3) containing the background image.
            full_frame: If True, render on the full image specified by imgname.
            imgname: Path to the original image file. Required if full_frame is True.
            side_view: If True, render the mesh from a side view.
            rot_angle: Rotation angle in degrees for side view rendering.
            mesh_base_color: Base color for the mesh as RGB values in [0, 1].
            scene_bg_color: Background color as RGB values in [0, 1].
            return_rgba: If True, return RGBA image instead of RGB.

        Returns:
            np.ndarray: The rendered image with the mesh overlaid. Shape is (H, W, 3)
                for RGB or (H, W, 4) for RGBA if return_rgba is True.

        """
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.0

        renderer = pyrender.OffscreenRenderer(
            viewport_width=image.shape[1],
            viewport_height=image.shape[0],
            point_size=1.0,
        )
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode="OPAQUE",
            baseColorFactor=(*mesh_base_color, 1.0),
        )

        camera_translation[0] *= -1.0

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, "mesh")

        camera_pose: np.ndarray = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center: list[float] = [image.shape[1] / 2.0, image.shape[0] / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=self.focal_length,
            fy=self.focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12,
        )
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        output_img = color[:, :, :3] * valid_mask + (1 - valid_mask) * image

        return output_img.astype(np.float32)
