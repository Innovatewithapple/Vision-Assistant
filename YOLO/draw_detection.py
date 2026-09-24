import cv2

def draw_detection(frame, box, class_name, score):

    x1, y1, x2, y2 = box

    frame_height, frame_width = frame.shape[:2]

    # Bounding box
    box_thickness = max(1, int(frame_width / 640))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        box_thickness
    )

    # Label settings
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.5
    text_thickness = max(1, int(frame_width / 640))

    class_text = class_name
    score_text = f"{score:.2f}"

    # Measure class name
    (class_width, text_height), baseline = cv2.getTextSize(
        class_text,
        font,
        font_scale,
        text_thickness
    )

    # Measure score
    (score_width, _), _ = cv2.getTextSize(
        score_text,
        font,
        font_scale,
        text_thickness
    )

    padding = 6
    gap = 6

    label_width = (
        class_width
        + gap
        + score_width
        + padding * 2
    )

    label_height = text_height + baseline + padding * 2

    # Make label fit inside bounding box
    box_width = x2 - x1

    while label_width > box_width and font_scale > 0.2:

        font_scale -= 0.05

        (class_width, text_height), baseline = cv2.getTextSize(
            class_text,
            font,
            font_scale,
            text_thickness
        )

        (score_width, _), _ = cv2.getTextSize(
            score_text,
            font,
            font_scale,
            text_thickness
        )

        label_width = (
            class_width
            + gap
            + score_width
            + padding * 2
        )

        label_height = text_height + baseline + padding * 2

    # Label position
    label_x1 = x1
    label_y2 = y1
    label_x2 = x1 + label_width
    label_y1 = y1 - label_height

    # If label doesn't fit above the box,
    # put it inside the top of the box
    if label_y1 < 0:
        label_y1 = y1
        label_y2 = y1 + label_height

    # Black background
    cv2.rectangle(
        frame,
        (label_x1, label_y1),
        (label_x2, label_y2),
        (0, 0, 0),
        -1
    )

    text_x = label_x1 + padding
    text_y = label_y2 - padding - baseline

    # White class name
    cv2.putText(
        frame,
        class_text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA
    )

    # Yellow confidence score
    cv2.putText(
        frame,
        score_text,
        (text_x + class_width + gap, text_y),
        font,
        font_scale,
        (0, 255, 255),
        text_thickness,
        cv2.LINE_AA
    )

    return frame