import cv2
from draw_detection import draw_detection
from detection import Detector
from segmentation import Segmentation,colors
from pose import PoseEstimator
import numpy as np

video_path = 'Video/streetwalkinghd.mp4' #'/Users/himanshuvyas/Downloads/vision-assistant/Video/streetwalking.mp4' #

#---------Capturing Start--------!
cap = cv2.VideoCapture(video_path) # create video capture object
pose_estimator = PoseEstimator()

while True:
    ret,frame = cap.read() # ret is boolean value to get if frame captured or not and frame is getting numpy values of video for each frame

    if not ret:
        break

    result = pose_estimator.estimator(frame=frame)
    frame = result.plot()

    cv2.imshow("Street Video",frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()


#======================================================xxxxxxxxxxxxxxxxxxxxxx====================================================xxxxxxxxxxxxx==================================

# video_path = "/Users/himanshuvyas/Downloads/vision-assistant/Video/streetwalking.mp4"

# cap = cv2.VideoCapture(video_path)

# ret, frame = cap.read()

# if not ret:
#     print("Could not read video")
#     cap.release()
#     exit()

# #-------Load Pretrained Fast R-CNN----------!
# weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT

# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)

# model.eval()


# #-----Prepare the Frame--------!
# frame_rgb = cv2.cvtColor(src=frame,code=cv2.COLOR_BGR2RGB)
# image_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float() / 255.0


# #-----Run Detection---------!
# with torch.no_grad():
#     predictions = model([image_tensor])

# prediction = predictions[0]


# #-----Inspect the Output-------!
# boxes = prediction['boxes']
# print("Boxes:")
# print(prediction["boxes"])

# labels = prediction['labels']
# print("\nLabels:")
# print(prediction["labels"])

# scores = prediction['scores']
# print("\nScores:")
# print(prediction["scores"])

# #----------Draw boxes and put labels with score-------------!
# for box,label,score in zip(boxes,labels,scores):
#     if score < 0.5:
#         break

#     x1,y1,x2,y2 = box.int().tolist()

#     cv2.rectangle(img=frame,pt1=(x1,y1),pt2=(x2,y2),color=(0,255,0),thickness=2)

#     text = f"Class {label.item()} : {score.item():.2f}"

#     cv2.putText(img=frame,text=text,org=(x1,y1-10),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=0.8,color=(0,255,0),thickness=2)


# #-----Show Result-----!
# cv2.imshow("Street Video",frame)
# cv2.waitKey(0)
# cap.release()
# cv2.destroyAllWindows()
