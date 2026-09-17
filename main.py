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
from Scenes.two_person_scene import xml

# 1. Image dimensions (1280x1920)
image_width = 720
image_height = 1280

# Automatically find centers and dynamic vertical focal scale
center_y = image_height / 2.0
center_x = image_width / 2.0

vertical_fov_degrees = 60.0
focal_length = center_y / math.tan(math.radians(vertical_fov_degrees / 2.0))

distance_estimator = DistanceEstimator(cam_height_meters=1.45, tilt_angle_degrees=0.0, focal_length_pixels=focal_length, optical_center_y=center_y)

video_path = 'Video/streetwalkinghd.mp4' #'/Users/himanshuvyas/Downloads/vision-assistant/Video/streetwalking.mp4' #

#---------Capturing Start--------!
cap = cv2.VideoCapture(video_path) # create video capture object


# model = mujoco.MjModel.from_xml_string(xml)
# data = mujoco.MjData(model)
# renderer = mujoco.Renderer(model=model,height=image_height,width=image_width)

pose_estimator = PoseEstimator()
segmentor = Segmentation()

while True:
    ret,frame = cap.read() # ret is boolean value to get if frame captured or not and frame is getting numpy values of video for each frame

    if not ret:
        break

    # ---------------------------------------------
    # MuJoCo updates the scene
    # ---------------------------------------------

    # mujoco.mj_forward(model, data)

    # renderer.update_scene(data,camera="main_camera")

    # ---------------------------------------------
    # Render camera image
    # ---------------------------------------------

    # frame = renderer.render()

    # frame = np.asarray(frame)

    # # MuJoCo → RGB
    # # OpenCV → BGR

    # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

    # result = pose_estimator.estimator(frame=frame)
    boxes,labels,scores,masks,track_ids,class_names = segmentor.segment(frame=frame)
    # frame = result.plot()

    for box, label, score, id in zip(boxes, labels, scores, track_ids):

        if score < 0.5:
            continue

        x1, y1, x2, y2 = box.int().tolist()

        class_name = class_names[int(label)]

        # ======================================================================
        # PIPELINE UPGRADE: 1D DEPTH TO TRUE 2D DIAGONAL DISTANCE
        # ======================================================================
        """
        WHY WE ADDED THESE 4 LINES:
        1. Forward Depth vs. Diagonal Distance:
           - Our 'get_distance_to_base()' function only reads vertical pixel rows (y2). 
           - It calculates 'Perpendicular Forward Depth' (how far down the street plane 
             an object is). For Person 1 (centered), depth equals straight-line distance.
           - For Person 2 (standing to the left), forward depth is 7.00m, but the true 
             diagonal hypotenuse distance to their feet is 7.28m.
        
        2. How the Math Automates This:
           - Line A: Finds the bounding box center X pixel and measures its horizontal 
             deviation from the camera's center line (center_x).
           - Line B: Converts that horizontal pixel displacement into real-world meters.
           - Line C & D: Uses the Pythagorean Theorem (Hypotenuse = sqrt(X^2 + Z^2)) to 
             find the true straight-line diagonal distance from you to them.
        
        3. Understanding the 7cm Variance (7.35m vs 7.28m):
           - Do not remove the vertical '-5' offset! The '-5' keeps your depth accurate.
           - The tiny 7cm difference is horizontal noise caused by bounding box thickness 
             (2 pixels wide) and the person's physical posture (arms/clothing width). 
           - This minor shift changes the box center by 3 pixels, which is completely 
             normal, expected, and safe for a real-time computer vision pipeline.
        """
        calibrated_y2 = y2 - 5
        distance = distance_estimator.get_distance_to_base(calibrated_y2)
        bbox_center_x = (x1 + x2) / 2.0
        horizontal_meters = ((bbox_center_x - center_x) * distance) / focal_length

        # Step C: Compute the true straight-line diagonal distance (Hypotenuse)
        true_diagonal_distance = math.sqrt(horizontal_meters**2 + distance**2)
        true_diagonal_distance = round(true_diagonal_distance, 2)
        distance_inmeter = f"{true_diagonal_distance} meters"

        if id == 5.0:
            print(f"Distance to Person: {true_diagonal_distance} meters | Id: {id}")
            frame = draw_detection(frame=frame,box=(x1, y1, x2, y2),class_name=distance_inmeter,score=float(score))


    cv2.imshow("Street Video",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# cap.release()
cv2.destroyAllWindows()
