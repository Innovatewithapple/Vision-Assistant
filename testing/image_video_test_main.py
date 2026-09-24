import cv2

from YOLO.segmentation import Segmentation
from YOLO.draw_detection import draw_detection


# ============================================================
# IMAGE
# ============================================================

image_path = "Video/Images/bottlec.png"

frame = cv2.imread(image_path)

if frame is None:
    raise FileNotFoundError(
        f"Could not load image: {image_path}"
    )


# ============================================================
# SEGMENTATION MODEL
# ============================================================

segmentor = Segmentation()


# ============================================================
# RUN SEGMENTATION
# ============================================================

boxes, labels, scores, masks, track_ids, class_names = segmentor.segment(
    frame=frame
)


print("Number of detections:", len(boxes))
print("Track IDs:", track_ids)


# ============================================================
# DRAW DETECTIONS
# ============================================================

for box, label, score in zip(
    boxes,
    labels,
    scores
):

    if score < 0.5:
        continue

    x1, y1, x2, y2 = box.int().tolist()

    class_name = class_names[int(label)]

    print(
        f"Detected: {class_name} "
        f"| Confidence: {float(score):.2f}"
    )

    frame = draw_detection(
        frame=frame,
        box=(x1, y1, x2, y2),
        class_name=class_name,
        score=float(score)
    )


# ============================================================
# DISPLAY
# ============================================================

cv2.imshow(
    "Image Detection",
    frame
)

cv2.waitKey(0)

cv2.destroyAllWindows()