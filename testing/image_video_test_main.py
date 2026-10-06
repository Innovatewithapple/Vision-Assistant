import cv2
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from YOLO.segmentation import Segmentation
from YOLO.draw_detection import draw_detection


# ============================================================
# VIDEO
# ============================================================

video_path = "Video/streetwalkinghd.mp4"     # <- must be a video file (.mp4), not .mp3

if not Path(video_path).exists():
    folder = Path(video_path).parent
    found = sorted(p.name for p in folder.glob("*")) if folder.exists() else "folder not found"
    raise FileNotFoundError(
        f"Could not find: {video_path}\n"
        f"Files in '{folder}': {found}"
    )

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {video_path}")


# ============================================================
# SEGMENTATION MODEL
# ============================================================

segmentor = Segmentation()


# ============================================================
# RUN ON EVERY FRAME
# ============================================================

frame_count = 0

while True:
    ok, frame = cap.read()
    if not ok:                      # end of video
        break

    frame_count += 1

    boxes, labels, scores, masks, track_ids, class_names = segmentor.segment(
        frame=frame
    )

    if frame_count % 30 == 0:       # a short status line twice a second, not every frame
        print(f"frame {frame_count}: {len(boxes)} detections, track IDs: {track_ids}")

    # ------------------------------------------------------------
    # DRAW DETECTIONS
    # ------------------------------------------------------------
    for box, label, score in zip(boxes, labels, scores):

        if score < 0.5:
            continue

        x1, y1, x2, y2 = box.int().tolist()

        class_name = class_names[int(label)]

        frame = draw_detection(
            frame=frame,
            box=(x1, y1, x2, y2),
            class_name=class_name,
            score=float(score)
        )

    # ------------------------------------------------------------
    # DISPLAY  (press q to quit)
    # ------------------------------------------------------------
    cv2.imshow("Video Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()