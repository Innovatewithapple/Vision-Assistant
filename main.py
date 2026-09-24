from ast import mod
import cv2
import mujoco.renderer
from polars import col
from YOLO.draw_detection import draw_detection
from YOLO.detection import Detector
from YOLO.segmentation import Segmentation,colors
from YOLO.pose import PoseEstimator
import numpy as np
from Calculation.DistanceEstiamtor import DistanceEstimator
import math
import mujoco
# from Scenes.single_person_scene import xml
# from Scenes.two_person_scene import xml
# from Scenes.threemug_on_table_scene import xml
from Scenes.mug_on_table_scene import xml
from Calculation.camera_utils import get_camera_parameters
from Calculation.GroundTruthValidator import GroundTruthValidator


# ============================================================
# Yolo
# ============================================================
# pose_estimator = PoseEstimator()
segmentor = Segmentation()

# ============================================================
# IMAGE & SCENE SETUP
# ============================================================
image_width = 1280
image_height = 720

center_y = image_height / 2.0
center_x = image_width / 2.0

# Compile your MuJoCo model structure locally
model = mujoco.MjModel.from_xml_string(xml)

# ============================================================
# DYNAMIC AUTOMATED INITIALIZATION
# ============================================================
# Call your helper module function directly using the local model memory structure
cam_params = get_camera_parameters(model, camera_name="main_camera", target_body_name="camera_target")

# Calculate pixel focal scale dynamically using the extracted FOV parameter
vertical_fov_degrees = cam_params["fovy_degrees"]
focal_length = center_y / math.tan(math.radians(vertical_fov_degrees / 2.0))

# Initialize the estimator class dynamically using the auto-extracted values
distance_estimator = DistanceEstimator(
    cam_height_meters=cam_params["camera_height"], 
    tilt_angle_degrees=cam_params["tilt_angle_degrees"], 
    focal_length_pixels=focal_length, 
    optical_center_y=center_y
)

# Print validation checks to the console logs
print("\n--- Pipeline Parameter Initialized Completely Automatically ---")
print(f"Height Applied: {cam_params['camera_height']} m")
print(f"FOV Applied:    {cam_params['fovy_degrees']}°")
print(f"Tilt Applied:   {cam_params['tilt_angle_degrees']}°")
print(f"Focal Length:   {round(focal_length, 2)} pixels")
print("----------------------------------------------------------------\n")

#---------Capturing Start--------!
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
renderer = mujoco.Renderer(model=model,height=image_height,width=image_width)


#-------Get Object Dimensions From MUJOCO
mesh_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_MESH,"wine_bottle")
# Number of vertices in this mesh
vertex_count = model.mesh_vertnum[mesh_id]

# Starting index of this mesh's vertices
vertex_start = model.mesh_vertadr[mesh_id]

# Get all vertices belonging to this mesh
vertices = model.mesh_vert[
    vertex_start : vertex_start + vertex_count
]

# Find minimum and maximum XYZ coordinates
mesh_min = vertices.min(axis=0)
mesh_max = vertices.max(axis=0)

# Physical dimensions after MuJoCo's mesh scaling
bottle_dimensions = mesh_max - mesh_min

bottle_width = bottle_dimensions[0]
bottle_depth = bottle_dimensions[1]
bottle_height = bottle_dimensions[2]

print("\nBottle dimensions:")
print(f"Width  : {bottle_width:.3f} m")
print(f"Depth  : {bottle_depth:.3f} m")
print(f"Height : {bottle_height:.3f} m")

# ============================================================
# GROUND TRUTH VALIDATOR
# ============================================================

ground_truth_validator = GroundTruthValidator(
    model=model,
    data=data,
    camera_name="main_camera",
    bottle_height=bottle_height,
    image_width=image_width,
    image_height=image_height,
    focal_length=focal_length
)

while True:
    # ---------------------------------------------
    # MuJoCo updates the scene
    # ---------------------------------------------
    mujoco.mj_forward(model, data)
    renderer.update_scene(data,camera="main_camera")

    # ---------------------------------------------
    # Render camera image
    # ---------------------------------------------
    frame = renderer.render()
    frame = np.asarray(frame)

    # MuJoCo → RGB
    # OpenCV → BGR

    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    boxes,labels,scores,mask,track_ids,class_names = segmentor.segment(frame=frame)
    print("track_id:",track_ids)
    overlay = frame.copy()

    if mask is not None and track_ids is not None:
        polygons = mask.xy
        used_bottles = set()

        # Get MuJoCo ground truth for all bottles
        ground_truth_bottles = (
            ground_truth_validator.get_bottle_ground_truth()
        )

        # Project all GT bottle centers into the image
        ground_truth_pixels = (
            ground_truth_validator.get_ground_truth_pixels(
                ground_truth_bottles
            )
        )

        for polygon, box, label, track_id, class_name, score in zip(
            polygons,
            boxes,
            labels,
            track_ids,
            class_names,
            scores
        ):

            if score < 0.1:
                continue

            polygon = polygon.astype(np.int32)

            track_id = int(track_id)

            color = colors[track_id % len(colors)]

            cv2.fillPoly(
                overlay,
                [polygon],
                color
            )

            x1, y1, x2, y2 = box.int().tolist()

            class_name = class_names[int(label)]

            # ============================================================
            # 2D DETECTION INFORMATION
            # ============================================================

            calibrated_y2 = y2 - 5

            distance = (
                distance_estimator.get_distance_to_base(
                    calibrated_y2
                )
            )

            bbox_center_x = (x1 + x2) / 2.0

            horizontal_meters = (
                (bbox_center_x - center_x)
                * distance
                / focal_length
            )

            # Camera-relative 3D position
            x = horizontal_meters
            y = distance
            z = bottle_height / 2.0

            # ============================================================
            # MATCH YOLO DETECTION TO MUJOCO GROUND TRUTH
            # ============================================================

            detection_center = np.array([
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0
            ])

            matched_bottle, pixel_error = (
                ground_truth_validator.match_detection(
                    detection_center,
                    ground_truth_pixels,
                    used_bottles
                )
            )

            if matched_bottle is None:
                print(
                    f"\nTrack_ID {track_id}: "
                    f"No MuJoCo ground-truth match"
                )
                continue

            # Prevent another detection from using
            # the same ground-truth bottle
            used_bottles.add(matched_bottle)

            # Get the 3D ground-truth position of
            # the bottle matched using ONLY 2D information
            ground_truth_center = (
                ground_truth_bottles[matched_bottle]
            )

            # ============================================================
            # 2D IMAGE POINT -> 3D BIN / WORLD POSITION
            # ============================================================

            # Detection center pixel
            u = (x1 + x2) / 2.0
            v = (y1 + y2) / 2.0

            # IMPORTANT:
            # Do NOT redefine focal_length here.
            # We already calculated it dynamically above.

            # Ray in MuJoCo camera coordinates
            ray_camera = np.array([
                (u - center_x) / focal_length,
                -(v - center_y) / focal_length,
                -1.0
            ])

            # Camera pose in world coordinates
            camera_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                "main_camera"
            )

            camera_position = (
                data.cam_xpos[camera_id].copy()
            )

            camera_rotation = (
                data.cam_xmat[camera_id]
                .reshape(3, 3)
            )

            # Convert camera ray -> world ray
            ray_world = (
                camera_rotation @ ray_camera
            )

            # Bottle center height in world coordinates
            floor_z = 0.05

            bottle_center_z = (
                floor_z
                + bottle_height / 2.0
            )

            # Intersect camera ray with
            # bottle-center height plane
            scale = (
                bottle_center_z
                - camera_position[2]
            ) / ray_world[2]

            world_position = (
                camera_position
                + scale * ray_world
            )

            world_x = world_position[0]
            world_y = world_position[1]
            world_z = world_position[2]

            # ============================================================
            # 3D POSITION COMPARISON
            # ============================================================

            print(f"\nTrack_ID: {track_id}")

            print(
                f"Matched GT : {matched_bottle}"
            )

            print(
                f"2D Pixel Error : "
                f"{pixel_error:.2f} px"
            )

            print(
                "\n--- 3D POSITION COMPARISON ---"
            )

            print(
                f"Vision    : "
                f"X={world_x:.3f}, "
                f"Y={world_y:.3f}, "
                f"Z={world_z:.3f}"
            )

            print(
                f"MuJoCo GT : "
                f"X={ground_truth_center[0]:.3f}, "
                f"Y={ground_truth_center[1]:.3f}, "
                f"Z={ground_truth_center[2]:.3f}"
            )

            # ============================================================
            # 3D ERROR
            # ============================================================

            error_x = (
                world_x
                - ground_truth_center[0]
            )

            error_y = (
                world_y
                - ground_truth_center[1]
            )

            error_z = (
                world_z
                - ground_truth_center[2]
            )

            print(
                f"Error     : "
                f"X={error_x:.3f}, "
                f"Y={error_y:.3f}, "
                f"Z={error_z:.3f}"
            )

            # ============================================================
            # CAMERA-RELATIVE POSITION
            # ============================================================

            print(
                f"XYZ Distance from Camera | "
                f"X: {x:.3f}m | "
                f"Y: {y:.3f}m | "
                f"Z: {z:.3f}m"
            )

            # ============================================================
            # STRAIGHT-LINE DISTANCE
            # ============================================================

            true_diagonal_distance = math.sqrt(
                horizontal_meters ** 2
                + distance ** 2
            )

            true_diagonal_distance = round(
                true_diagonal_distance,
                2
            )

            distance_inmeter = (
                f"{true_diagonal_distance} meters"
            )

            print(
                f"Distance to Person: "
                f"{true_diagonal_distance} meters | "
                f"Score: {score:.4f}"
            )
            #frame = draw_detection(frame=frame,box=(x1, y1, x2, y2),class_name=distance_inmeter,score=float(score))
    alpha = 0.27
    frame = cv2.addWeighted(frame,1-alpha,overlay,alpha,0)

    if mask is not None and track_ids is not None:
        polygons = mask.xy
    
        for polygon,box,label,track_id,class_name,score in zip(polygons,boxes,labels,track_ids,class_names,scores):
          if score < 0.1:
              continue
                
          polygon = polygon.astype(np.int32)
          track_id = int(track_id)
          color = colors[track_id % len(colors)]

          cv2.polylines(frame,[polygon],isClosed=True,color=color,thickness=1,lineType=cv2.LINE_AA)

    cv2.imshow('Frame',frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cap.release()
cv2.destroyAllWindows()

