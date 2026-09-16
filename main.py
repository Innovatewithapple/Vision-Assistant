import cv2
from draw_detection import draw_detection
from detection import Detector
from segmentation import Segmentation,colors
from pose import PoseEstimator
import numpy as np
from Calculation.DistanceEstiamtor import DistanceEstimator
import math

image_path = "Video/Images/streetwalking.jpg"

frame = cv2.imread(image_path)

if frame is None:
    print("Could not load image")
    exit()

# 1. Image dimensions (1280x1920)
image_width = 2160
image_height = 3840

# 2. Automatically calculate Center Y
# For a 1920-tall image, this gives exactly 960
center_y = image_height / 2 
 
# 3. Automatically calculate Focal Length in pixels
# Most standard cameras have a vertical field of view (FOV) around 60 degrees.
# This formula scales focal length perfectly for your 1920 height.
vertical_fov_degrees = 60.0
focal_length = center_y / math.tan(math.radians(vertical_fov_degrees / 2))

pose_estimator = PoseEstimator()

distance_estimator = DistanceEstimator(cam_height_meters=1.4, tilt_angle_degrees=0.0, focal_length_pixels=focal_length, optical_center_y=center_y)

# result = pose_estimator.estimator(frame=frame)

# boxes = result.boxes.xyxy
# labels = result.boxes.cls
# scores = result.boxes.conf
# class_names = result.names

# for box, label, score in zip(boxes, labels, scores):

#     if score < 0.5:
#         continue

#     x1, y1, x2, y2 = box.int().tolist()

#     class_name = class_names[int(label)]

#     print(class_name, float(score), (x1, y1, x2, y2))
#     frame = draw_detection(frame=frame,box=(x1, y1, x2, y2),class_name=class_name,score=float(score))
#     distance = distance_estimator.get_distance_to_base(y2)

#     print(f"Distance to Person: {distance} meters")

#     break

# cv2.imshow("Image", frame)

# cv2.waitKey(0)
# cv2.destroyAllWindows()

video_path = 'Video/streetwalking.mp4' #'/Users/himanshuvyas/Downloads/vision-assistant/Video/streetwalking.mp4' #

#---------Capturing Start--------!
cap = cv2.VideoCapture(video_path) # create video capture object
pose_estimator = PoseEstimator()
segmentor = Segmentation()

while True:
    ret,frame = cap.read() # ret is boolean value to get if frame captured or not and frame is getting numpy values of video for each frame

    if not ret:
        break

    # result = pose_estimator.estimator(frame=frame)
    boxes,labels,scores,masks,track_ids,class_names = segmentor.segment(frame=frame)
    # frame = result.plot()

    for box, label, score, id in zip(boxes, labels, scores, track_ids):

        if score < 0.5:
            continue

        x1, y1, x2, y2 = box.int().tolist()

        class_name = class_names[int(label)]

        # print(class_name, float(score), (x1, y1, x2, y2))
        distance = distance_estimator.get_distance_to_base(y2)
        distance_inmeter = f"{distance} meters"
        if id == 7.0:
            print(f"Distance to Person: {distance} meters | Id: {id}")
            frame = draw_detection(frame=frame,box=(x1, y1, x2, y2),class_name=distance_inmeter,score=float(score))

    

    cv2.imshow("Street Video",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
