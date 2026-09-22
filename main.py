from ast import mod
import cv2
import mujoco.renderer
from draw_detection import draw_detection
from detection import Detector
from segmentation import Segmentation,colors
from pose import PoseEstimator
import numpy as np
from Calculation.DistanceEstiamtor import DistanceEstimator
import math
import mujoco
# from Scenes.single_person_scene import xml
# from Scenes.two_person_scene import xml
from Scenes.threemug_on_table_scene import xml
from camera_utils import get_camera_parameters

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

video_path = 'Video/streetwalkinghd.mp4' #'/Users/himanshuvyas/Downloads/vision-assistant/Video/streetwalking.mp4' #

#---------Capturing Start--------!
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model=model,height=image_height,width=image_width)

pose_estimator = PoseEstimator()
segmentor = Segmentation()

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
    boxes,labels,scores,masks,track_ids,class_names = segmentor.segment(frame=frame)
    print("boxes:", len(boxes))
    print("track_ids:", track_ids)

    for box, label, score in zip(boxes, labels, scores):
        if score < 0.1:
            continue

        x1, y1, x2, y2 = box.int().tolist()
        class_name = class_names[int(label)]

        calibrated_y2 = y2 - 5
        distance = distance_estimator.get_distance_to_base(calibrated_y2)
        bbox_center_x = (x1 + x2) / 2.0
        horizontal_meters = ((bbox_center_x - center_x) * distance) / focal_length

        # Step C: Compute the true straight-line diagonal distance (Hypotenuse)
        true_diagonal_distance = math.sqrt(horizontal_meters**2 + distance**2)
        true_diagonal_distance = round(true_diagonal_distance, 2)
        distance_inmeter = f"{true_diagonal_distance} meters"

        print(f"Distance to Person: {true_diagonal_distance} meters | Id: {id} | Score: {score}")
        frame = draw_detection(frame=frame,box=(x1, y1, x2, y2),class_name=distance_inmeter,score=float(score))


    cv2.imshow("Street Video",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cap.release()
cv2.destroyAllWindows()

