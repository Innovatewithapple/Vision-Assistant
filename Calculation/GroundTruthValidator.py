import mujoco
import numpy as np


class GroundTruthValidator:

    def __init__(self,model,data,camera_name,bottle_height,image_width,image_height,focal_length):
        self.model = model
        self.data = data

        self.bottle_height = bottle_height

        self.image_width = image_width
        self.image_height = image_height

        self.focal_length = focal_length

        # Image center / principal point
        self.cx = image_width / 2.0
        self.cy = image_height / 2.0

        # Get camera ID
        self.camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name
        )

        # Make sure MuJoCo runtime positions are calculated
        mujoco.mj_forward(model, data)

    # ========================================================
    # GET CAMERA POSE
    # ========================================================

    def get_camera_pose(self):

        camera_position = self.data.cam_xpos[
            self.camera_id
        ].copy()

        camera_rotation = self.data.cam_xmat[
            self.camera_id
        ].reshape(3, 3).copy()

        return camera_position, camera_rotation

    # ========================================================
    # GET ALL BOTTLE GROUND TRUTH
    # ========================================================

    def get_bottle_ground_truth(self):

        ground_truth_bottles = {}

        for body_id in range(self.model.nbody):

            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id
            )

            if (
                body_name is not None
                and body_name.startswith("bottle_")
            ):

                body_position = self.data.xpos[
                    body_id
                ].copy()

                # Bottle center
                ground_truth_center = body_position.copy()

                ground_truth_center[2] += (
                    self.bottle_height / 2.0
                )

                ground_truth_bottles[
                    body_name
                ] = ground_truth_center

        return ground_truth_bottles

    # ========================================================
    # WORLD XYZ -> IMAGE PIXEL
    # ========================================================

    def world_to_image(self, world_point):

        camera_position, camera_rotation = (
            self.get_camera_pose()
        )

        # World -> camera coordinates
        camera_point = (camera_rotation.T @ (world_point - camera_position))

        x_cam = camera_point[0]
        y_cam = camera_point[1]
        z_cam = camera_point[2]

        # MuJoCo camera looks along -Z
        depth = -z_cam

        if depth <= 0:
            return None

        # Camera -> image coordinates
        u = self.cx + (
            self.focal_length
            * x_cam
            / depth
        )

        v = self.cy - (
            self.focal_length
            * y_cam
            / depth
        )

        return np.array([u, v])

    # ========================================================
    # PROJECT ALL GROUND TRUTH BOTTLES
    # ========================================================

    def get_ground_truth_pixels(self,ground_truth_bottles):

        ground_truth_pixels = {}

        for bottle_name, world_position in (
            ground_truth_bottles.items()
        ):

            pixel = self.world_to_image(
                world_position
            )

            if pixel is not None:

                ground_truth_pixels[
                    bottle_name
                ] = pixel

        return ground_truth_pixels

    # ========================================================
    # MATCH YOLO DETECTION TO GT USING 2D
    # ========================================================

    def match_detection(self,detection_center,ground_truth_pixels,used_bottles):

        nearest_bottle = None
        nearest_pixel_distance = float("inf")

        for bottle_name, gt_pixel in (
            ground_truth_pixels.items()
        ):

            # Don't match the same GT bottle twice
            if bottle_name in used_bottles:
                continue

            pixel_error = np.linalg.norm(
                detection_center - gt_pixel
            )

            if pixel_error < nearest_pixel_distance:

                nearest_pixel_distance = pixel_error
                nearest_bottle = bottle_name

        return (nearest_bottle,nearest_pixel_distance)