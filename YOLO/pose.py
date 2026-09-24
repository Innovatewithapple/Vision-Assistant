from ultralytics import YOLO

class PoseEstimator:
    def __init__(self):
        self.model = YOLO('yolo26n-pose.pt')

    def estimator(self,frame):
        results = self.model(frame,verbose=False,device='mps')
        result = results[0]
        #keypoints = result.keypoints
        return result