import cv2
from draw_detection import draw_detection
from detection import Detector
from segmentation import Segmentation
import numpy as np

video_path = '/Users/himanshuvyas/Downloads/vision-assistant/Video/streetwalkinghd.mp4' #'/Users/himanshuvyas/Downloads/vision-assistant/Video/streetwalking.mp4' #

#Different colors for different instances
colors = [
    (77, 255, 77),       # Soft Green
    (180, 120, 255),     # Soft Purple
    (255, 100, 100),     # Soft Red
    (100, 180, 255),     # Soft Sky Blue
    (255, 220, 100),     # Soft Yellow
    (100, 255, 220),     # Soft Teal
    (255, 160, 100),     # Soft Orange
    (255, 120, 200),     # Soft Pink
    (180, 255, 100),     # Soft Lime
    (100, 255, 255),     # Soft Cyan
    (255, 140, 140),     # Soft Coral
    (200, 160, 255),     # Soft Violet
]

#---------Capturing Start--------!
cap = cv2.VideoCapture(video_path) # create video capture object
# detector = Detector()
segmentor = Segmentation()

while True:
    ret,frame = cap.read() # ret is boolean value to get if frame captured or not and frame is getting numpy values of video for each frame

    if not ret:
        break

    boxes,labels,scores,masks,track_ids,class_names = segmentor.segment(frame=frame)

    #create the seperate object for the mask
    overlay = frame.copy()

    if masks is not None and track_ids is not None:
        polygons = masks.xy
        for polygon,label,score,track_id in zip(polygons,labels,scores,track_ids):
            if score < 0.4:
                continue


            # convert polygen coordinates to pixel coordinates
            polygon = polygon.astype(np.int32)

            track_id = int(track_id)

            #select color for the instance
            color = colors[track_id % len(colors)]

            # Fill the mask on the overlay
            cv2.fillPoly(overlay,[polygon],color)

            #Draw the mask boundary on the overlay
            # cv2.polylines(overlay,[polygon],isClosed=True,color=color,thickness=1)

    #Blend original frame with mask overlay
    alpha = 0.35

    frame = cv2.addWeighted(frame,1 - alpha,overlay,alpha,0)

    if masks is not None and track_ids is not None:
            polygons = masks.xy
            for polygon,label,score,track_id in zip(polygons,labels,scores,track_ids):
                if score < 0.4:
                    continue
    
    
                # convert polygen coordinates to pixel coordinates
                polygon = polygon.astype(np.int32)
    
                track_id = int(track_id)
    
                #select color for the instance
                color = colors[track_id % len(colors)]
                #Draw the mask boundary on the overlay
                cv2.polylines(frame,[polygon],isClosed=True,color=color,thickness=1,lineType=cv2.LINE_AA)   


    


    # for box,label,score in zip(boxes,labels,scores):
    #     if score < 0.5:
    #         continue

    #     x1,y1,x2,y2 = box.int().tolist()
    #     class_name = class_names[int(label)]

    #     frame = draw_detection(frame=frame,box=(x1, y1, x2, y2),class_name=class_name,score=float(score))

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
