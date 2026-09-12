import cv2
import torch
import torchvision
from ultralytics import YOLO

class Detector:

    def __init__(self):

        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):

        results = self.model(frame, verbose=False,device="mps")

        result = results[0]

        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        scores = result.boxes.conf

        class_names = result.names

        return boxes, labels, scores, class_names
    # def __init__(self):
    #     weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    #     self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #     print("Using device:", self.device)
    #     self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights,pretrained=True,rpn_post_nms_top_n_test=150,box_detections_per_img=10)
    #     # self.model = self.model.to(self.device)

    #     self.model.eval()
    #     self.class_names = weights.meta['categories']

    # def detect(self,frame):
    #     #----Resize for faster inference-----!
    #     original_height, original_width = frame.shape[:2]

    #     new_width = 800
    #     new_height = int(original_height * new_width / original_width)

    #     resized_frame = cv2.resize(frame,(new_width,new_height))

    #     #BGR->RGB
    #     frame_rgb = cv2.cvtColor(resized_frame,cv2.COLOR_BGR2RGB)
    #     image_tensor = torch.from_numpy(frame_rgb).permute(2,0,1).float() / 255.0
    #     # image_tensor = image_tensor.to(self.device)

    #     with torch.no_grad():
    #         predicton = self.model([image_tensor])[0]

    #     boxes= predicton['boxes']
    #     labels = predicton['labels']
    #     scores = predicton['scores']

    #     #Convert boxes back to original frame coordinates
    #     scale_x = original_width / new_width
    #     scale_y = original_height / new_height

    #     boxes[:, 0] *= scale_x
    #     boxes[:, 1] *= scale_y
    #     boxes[:, 2] *= scale_x
    #     boxes[:, 3] *= scale_y

    #     return boxes, labels, scores, self.class_names 