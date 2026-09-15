import cv2
import torch
import torchvision
from ultralytics import YOLO

class Detector:

    def __init__(self):

        self.model = YOLO("yolo26n.pt")

    def detect(self, frame):

        results = self.model(frame, verbose=False,device="mps")

        result = results[0]

        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        scores = result.boxes.conf

        class_names = result.names

        return boxes, labels, scores, class_names
   