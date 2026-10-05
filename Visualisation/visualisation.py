import cv2
from YOLO.segmentation import colors
import numpy as np
import mujoco
from Calculation.panda_ik import PandaIK

BOTTLE_RADIUS = 0.0368 - 0.005
BOTTLE_HEIGHT = 0.30
APPROACH_CLEARANCE = 0.10
REGISTER_MATCH_RADIUS = 0.05   # meters — how close a new reading must be to count as "same bottle"
detected_bottles = {}   # key: bottle_id (int, 0,1,2...), value: {"pos": [x,y,z], "confirmed": True}
detection_enabled = True
_last_print_cam = None

TRUTH_XY = None

def set_truth(xy):
    global TRUTH_XY
    TRUTH_XY = np.array(xy)

TRUTH_ALL = {}            # {"bottle_1": array([x, y]), ...}  true positions from the simulator
MATCH_WARN_MM = 30.0      # farther than this from every real bottle = suspicious

def set_truth_all(model, data):
    """Snapshot the true XY of every bottle_N body in the scene."""
    TRUTH_ALL.clear()
    i = 1
    while True:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bottle_{i}")
        if bid < 0:
            break
        TRUTH_ALL[f"bottle_{i}"] = data.xpos[bid][:2].copy()
        i += 1
    print(f"Ground truth snapshot ({len(TRUTH_ALL)} bottles):")
    for name, xy in TRUTH_ALL.items():
        print(f"   sim {name}: X={xy[0]:.3f}, Y={xy[1]:.3f}")

def nearest_truth(xy):
    if not TRUTH_ALL:
        return None, None, None
    name = min(TRUTH_ALL, key=lambda n: np.linalg.norm(TRUTH_ALL[n] - xy))
    return name, TRUTH_ALL[name], float(np.linalg.norm(TRUTH_ALL[name] - xy))

def report_vs_truth(bottle_id, pos):
    name, t, err = nearest_truth(np.asarray(pos[:2]))
    if name is None:
        return
    flag = "" if err * 1000 < MATCH_WARN_MM else "   <-- no real bottle close by (ghost / bad estimate)"
    print(f"    registered #{bottle_id} matches sim {name}: truth X={t[0]:.3f}, Y={t[1]:.3f}  err={err*1000:.1f} mm{flag}")

def print_registration_report():
    """Final table: every registered bottle vs its nearest real bottle, plus real bottles never detected."""
    if not TRUTH_ALL:
        return
    print("\n===== REGISTERED vs GROUND TRUTH =====")
    used = {}
    for bid, info in detected_bottles.items():
        p = np.array(info["pos"])
        name, t, err = nearest_truth(p[:2])
        note = ""
        if err * 1000 >= MATCH_WARN_MM:
            note = "  <-- ghost?"
        elif name in used:
            note = f"  <-- duplicate of registered #{used[name]}"
        used.setdefault(name, bid)
        print(f"  #{bid}: est=({p[0]:.3f}, {p[1]:.3f})  sim {name} truth=({t[0]:.3f}, {t[1]:.3f})  err={err*1000:.1f} mm{note}")
    missed = [n for n in TRUTH_ALL if n not in used]
    print(f"  never detected: {missed if missed else 'none'}")
    print("======================================\n")


def disable_detection_and_get_bottles():
    """Call this once scanning is done. Freezes detection and returns
    the final registered bottle list."""
    set_detection_enabled(False)
    for bottle_id, info in detected_bottles.items():
        x, y, z = info["pos"]
        print(f"FINAL bottle_{bottle_id} (used for approach): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
    print_registration_report()
    return detected_bottles


def approach_bottle(bottle_id, model, data, ik, bottle_height=BOTTLE_HEIGHT, clearance=APPROACH_CLEARANCE):
    """
    Computes an IK target positioned above the given registered bottle,
    solves for it, and returns the joint target (does NOT apply it to
    data.qpos permanently — caller decides how to drive toward it).
    """
    if bottle_id not in detected_bottles:
        raise ValueError(f"bottle_id {bottle_id} not found in detected_bottles")

    bx, by, bz = detected_bottles[bottle_id]["pos"]

    approach_position = np.array([bx, by, bz + bottle_height + clearance])

    # don't let ik.solve's internal qpos scratch-work affect the live sim
    saved_qpos = data.qpos[:7].copy()
    approach_q = ik.solve(approach_position)
    data.qpos[:7] = saved_qpos
    mujoco.mj_forward(model, data)

    return approach_q, approach_position

def set_detection_enabled(enabled: bool):
    global detection_enabled
    detection_enabled = enabled

def Calculate_World_Position(polygon, model, data, counter_z=0.485, bottle_radius=BOTTLE_RADIUS):
    v_max = polygon[:, 1].max()
    bottom = polygon[polygon[:, 1] >= v_max - 2.0]          # points within 2 px of the lowest row
    u = (bottom[:, 0].min() + bottom[:, 0].max()) / 2.0 + 0.5
    v = v_max + 0.5

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    cam_pos = data.cam_xpos[cam_id].copy()
    R = data.cam_xmat[cam_id].reshape(3, 3)

    W, H = 1280, 960   # MUST equal the renderer's width/height (see point 3)
    fy = 0.5 * H / np.tan(np.radians(model.cam_fovy[cam_id]) / 2.0)
    fx = fy             # square pixels
    cx, cy = W / 2.0, H / 2.0

    a = (u - cx) / fx
    b = -(v - cy) / fy
    ray_world = R @ np.array([a, b, -1.0])

    t = (counter_z - cam_pos[2]) / ray_world[2]
    edge = cam_pos + t * ray_world

    # Ground line of this image row: plane normal x world-up
    plane_normal = R @ np.array([0.0, 1.0, b])
    line_dir = np.cross(plane_normal, [0.0, 0.0, 1.0])[:2]
    n = np.array([-line_dir[1], line_dir[0]])
    n /= np.linalg.norm(n)
    if np.dot(n, edge[:2] - cam_pos[:2]) < 0:   # point away from the camera
        n = -n

    center_xy = edge[:2] + bottle_radius * n
    return np.array([center_xy[0], center_xy[1], counter_z])


def register_bottle(world_position, radius=REGISTER_MATCH_RADIUS):
    for bottle_id, info in detected_bottles.items():
        known_xy = np.array(info["pos"][:2])
        if np.linalg.norm(world_position[:2] - known_xy) < radius:
            info["pos"] = world_position.tolist()
            return bottle_id, False

    new_id = len(detected_bottles)
    detected_bottles[new_id] = {"pos": world_position.tolist(), "confirmed": True}
    return new_id, True


def Draw_Segmentation(frame, boxes, labels, scores, mask, track_ids, class_names, model, data):
    wrist_overlay = frame.copy()

    if detection_enabled and mask is not None:
        polygons = mask.xy

        for idx, (polygon, box, label, score) in enumerate(zip(polygons, boxes, labels, scores)):
            class_name = class_names[int(label)]
            if class_name != 'bottle':
                continue

            world_position = Calculate_World_Position(polygon, model, data)
            if not (0.20 <= world_position[1] <= 0.35):
                # print("Ghost bottle Detected!!!")
                continue
            if TRUTH_ALL:
                global _last_print_cam
                cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
                cam_pos = data.cam_xpos[cam_id].copy()
                if _last_print_cam is None or np.linalg.norm(cam_pos - _last_print_cam) > 0.01:
                    _last_print_cam = cam_pos
                    name, t, err = nearest_truth(world_position[:2])
                    print(f"pose cam=({cam_pos[0]:.2f},{cam_pos[1]:.2f},{cam_pos[2]:.2f}) "
                          f"est=({world_position[0]:.3f},{world_position[1]:.3f}) "
                          f"nearest sim {name}=({t[0]:.3f},{t[1]:.3f}) err={err*1000:.1f} mm")
            bottle_id, is_new = register_bottle(world_position)

            if is_new:
                print(f"Registered bottle_{bottle_id}: "
                      f"X={world_position[0]:.3f}, Y={world_position[1]:.3f}, Z={world_position[2]:.3f}")
                report_vs_truth(bottle_id, world_position)
                print_registration_report()

            color = colors[bottle_id % len(colors)]

            polygon_draw = polygon.astype(np.int32)
            cv2.fillPoly(wrist_overlay, [polygon_draw], color=color)
            cv2.polylines(wrist_overlay, [polygon_draw], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)

    alpha = 0.27
    wrist_frame = cv2.addWeighted(frame, 1 - alpha, wrist_overlay, alpha, 0)

    return wrist_frame
