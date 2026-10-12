import math
import numpy as np
from ultralytics import YOLO

# COCO keypoint indices
NOSE, L_EYE, R_EYE = 0, 1, 2
L_ANKLE, R_ANKLE = 15, 16


class PoseEstimator:
    def __init__(self, model_path="yolo11n-pose.pt", device="mps",every_n=2, min_box_h=150, max_people=4,window_s=1.0, imgsz=320):
        self.model_path = model_path
        self.device = device
        self.every_n = every_n          # run pose every N frames
        self.min_box_h = min_box_h      # ignore people smaller than this (px)
        self.max_people = max_people    # cap per run, nearest first
        self.window_s = window_s        # window for the leg-swing score
        self.imgsz = imgsz
        self.model = None               # loaded on first use
        self.frame_idx = 0
        self.ankle_hist = {}            # track_id -> [(t, ankle_gap / body_height)]
        self.results = {}               # track_id -> {"legs", "facing", "t"}
        self.samples = {}               # track_id -> [(t, legs, facing)]

    def _load(self):
        if self.model is None:
            self.model = YOLO(self.model_path)

    def update(self, frame, candidates, now):
        """candidates: list of (track_id, (x1, y1, x2, y2)), nearest first.
        frame must be the raw frame, without drawings."""
        self.frame_idx += 1
        if self.frame_idx % self.every_n != 0:
            return                      # keep the previous results

        H, W = frame.shape[:2]
        crops, meta = [], []
        for tid, (x1, y1, x2, y2) in candidates:
            bh = y2 - y1
            if bh < self.min_box_h:
                continue
            pad = int(0.1 * bh)
            cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
            cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
            if cx2 - cx1 < 10 or cy2 - cy1 < 10:
                continue
            crops.append(frame[cy1:cy2, cx1:cx2])
            meta.append((tid, bh))
            if len(crops) >= self.max_people:
                break
        if not crops:
            return

        self._load()
        outs = self.model.predict(crops, imgsz=self.imgsz, conf=0.25,
                                  device=self.device, verbose=False)

        for (tid, bh), r in zip(meta, outs):
            if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            i = int(np.argmax(areas))                  # biggest person in the crop
            kxy = r.keypoints.xy.cpu().numpy()[i]
            kcf = r.keypoints.conf.cpu().numpy()[i]

            # --- legs: distance between ankles / body height ---
            legs = None
            if kcf[L_ANKLE] >= 0.4 and kcf[R_ANKLE] >= 0.4:
                gap = math.hypot(*(kxy[L_ANKLE] - kxy[R_ANKLE])) / bh
                hist = self.ankle_hist.setdefault(tid, [])
                hist.append((now, gap))
                hist[:] = [h for h in hist if now - h[0] <= self.window_s]
                if len(hist) >= 8:
                    vals = [h[1] for h in hist]
                    legs = max(vals) - min(vals)       # big swing = legs moving

            # --- facing: nose and an eye visible ---
            facing = bool(kcf[NOSE] >= 0.5 and (kcf[L_EYE] >= 0.5 or kcf[R_EYE] >= 0.5))

            self.results[tid] = {"legs": legs, "facing": facing, "t": now}
            if legs is not None:
                buf = self.samples.setdefault(tid, [])
                buf.append((now, legs, facing))
                buf[:] = [s for s in buf if now - s[0] <= 3.0]

    def summary(self, track_id, now, span=1.5, min_n=8):
        buf = [s for s in self.samples.get(track_id, []) if now - s[0] <= span]
        if len(buf) < min_n or now - buf[0][0] < 0.8 * span:
            return None
        return {"legs": float(np.median([s[1] for s in buf])),
                "facing": float(np.mean([s[2] for s in buf]))}

    def get(self, track_id, now, max_age=1.0):
        r = self.results.get(track_id)
        if r and now - r["t"] <= max_age:
            return r
        return None

    def forget(self, track_id):
        self.ankle_hist.pop(track_id, None)
        self.samples.pop(track_id, None)
        self.results.pop(track_id, None)