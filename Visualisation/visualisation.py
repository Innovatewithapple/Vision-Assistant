import cv2
from YOLO.segmentation import colors
import numpy as np
import mujoco
seen_track_ids = set()

def Calculate_World_Position(
    box,
    model,
    data,
    bottle_height=0.30,
    counter_z=0.485
):
    # ------------------------------------------------------------
    # Detection center pixel
    # ------------------------------------------------------------
    x1, y1, x2, y2 = box.int().tolist()

    u = (x1 + x2) / 2.0
    v = y2

    # ------------------------------------------------------------
    # Wrist camera
    # ------------------------------------------------------------

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "wrist_camera"
    )

    camera_position = data.cam_xpos[camera_id].copy()

    camera_rotation = (
        data.cam_xmat[camera_id]
        .reshape(3, 3)
    )

    # ------------------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------------------

    IMAGE_WIDTH = 1280
    IMAGE_HEIGHT = 960

    fovy = model.cam_fovy[camera_id]

    center_x = IMAGE_WIDTH / 2.0
    center_y = IMAGE_HEIGHT / 2.0

    fovy_rad = np.radians(fovy)
    fx = (
    0.5 * IMAGE_WIDTH
    / np.tan(fovy_rad / 2.0)
    )
    fy = (
    0.5 * IMAGE_HEIGHT
    / np.tan(fovy_rad / 2.0)
    )
    ray_camera = np.array([
    (u - center_x) / fx,
    -(v - center_y) / fy,
    -1.0
    ])
    # ------------------------------------------------------------
    # Camera ray -> world ray
    # ------------------------------------------------------------

    ray_world = camera_rotation @ ray_camera

    # ------------------------------------------------------------
    # Ray-plane intersection
    # ------------------------------------------------------------

    scale = (
        counter_z - camera_position[2]
    ) / ray_world[2]

    world_position = (
        camera_position
        + scale * ray_world
    )

    return world_position

def Draw_Segmentation(frame, boxes, labels, scores, mask, track_ids, class_names,model,data):
    wrist_overlay = frame.copy()
    if mask is not None:
        polygons = mask.xy
        
        for idx, (polygon, box, label, score) in enumerate(zip(polygons, boxes, labels, scores)):
            class_name = class_names[int(label)]
            if class_name != 'bottle':
                continue

            polygon = polygon.astype(np.int32)
            
            # --- color selection (testing mode) ---
            track_id = None
            if track_ids is not None:
                track_id = int(track_ids[idx])
                if track_id not in seen_track_ids:
                    world_position = Calculate_World_Position(box,model,data)
                    print(f"Track_ID {track_id}: "f"World Position -> "f"X={world_position[0]:.3f}, "f"Y={world_position[1]:.3f}, "f"Z={world_position[2]:.3f}")
                    print(f"Class: {class_name} | "f"score: {score:.2f} | "f"trackID: {track_ids[idx]}"f"box: {box.tolist()}")
                    seen_track_ids.add(track_id)
                color = colors[track_id % len(colors)]
            else:
                color = colors[int(label) % len(colors)]

            cv2.fillPoly(wrist_overlay, [polygon], color=color)
            cv2.polylines(wrist_overlay, [polygon], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    alpha = 0.27
    wrist_frame = cv2.addWeighted(frame, 1 - alpha, wrist_overlay, alpha, 0)

    return wrist_frame

