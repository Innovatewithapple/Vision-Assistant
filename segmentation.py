from ultralytics import YOLO

class Segmentation:
    def __init__(self):
        self.model = YOLO("yolo26n-seg.pt")

    def segment(self,frame):
        results = self.model.track(frame,persist=True,verbose=False,device='mps')
        result = results[0]

        boxes = result.boxes.xyxy
        labels = result.boxes.cls
        scores = result.boxes.conf
        mask = result.masks
        track_ids = result.boxes.id
        class_names = result.names

        return boxes,labels,scores,mask,track_ids,class_names


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