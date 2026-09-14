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
        class_names = labels.names

        return boxes,labels,scores,mask,track_ids,class_names