import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math
from Calculation.Distance_Calculation_Homanoid import get_real_world_camera_parameters
from YOLO.segmentation import Segmentation
from YOLO.draw_detection import draw_detection
from Visualisation.humanoid_draw_segmentation import Draw_Segment_Around
from Calculation.DistanceEstiamtor import DistanceEstimator
from Visualisation.Info_Panel import toggle_panel
import time


# ============================================================
# VIDEO PIPELINE SETUP
# ============================================================
video_path = "Video/streetwalkinghd.mp4"
segmentor = Segmentation()

if not Path(video_path).exists():
    folder = Path(video_path).parent
    found = sorted(p.name for p in folder.glob("*")) if folder.exists() else "folder not found"
    raise FileNotFoundError(
        f"Could not find: {video_path}\n"
        f"Files in '{folder}': {found}"
    )

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")

video_fps = cap.get(cv2.CAP_PROP_FPS)
print("Video FPS:", video_fps)

# ------------------------------------------------------------
# ONE-TIME CAMERA INITIALIZATION (Fixes Bug #2)
# ------------------------------------------------------------
# Read a single test frame to grab the native resolution specs
success, initial_frame = cap.read()
if not success:
    raise RuntimeError("Failed to read video stream properties.")

# Extract true resolution dimensions safely (Fixes Bug #1)
frame_height, frame_width = initial_frame.shape[:2]
print('frame_height: ',frame_height)
print('frame_width: ',frame_width)


# Call helper to get static camera configurations
cam_params = get_real_world_camera_parameters(frame_height, frame_width)
print(cam_params["camera_height"])
print(cam_params["tilt_angle_degrees"])

# Initialize your core tracker module once
distance_estimator = DistanceEstimator(
    cam_height_meters=cam_params["camera_height"],
    tilt_angle_degrees=cam_params["tilt_angle_degrees"],
    focal_length_pixels=cam_params["focal_length_pixels"],
    optical_center_y=cam_params["optical_center_y"]
)

# Extract center tracking points for your 2D diagonal math block
center_x = cam_params["optical_center_x"]
focal_length_x = cam_params["focal_length_pixels"]

# Reset video pointer back to the beginning frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0

# ------------------------------------------------------------
# LIVE VIDEO TRACKING LOOP
# ------------------------------------------------------------
while True:
    ok, frame = cap.read()
    if not ok:  # End of video file reached
        break

    frame_count += 1
    video_time = frame_count / video_fps

    # Run your tracking segmentation model on the live frame
    boxes, labels, scores, masks, track_ids, class_names = segmentor.segment(frame=frame)
    # ------------------------------------------------------------
    # DISPLAY (press q to quit)
    # ------------------------------------------------------------
    frame = Draw_Segment_Around(boxes, labels, class_names, frame, track_ids, scores, masks,distance_estimator,center_x,focal_length_x,video_time)
    cv2.imshow("Video Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord("s"):
        toggle_panel()

cap.release()
cv2.destroyAllWindows()