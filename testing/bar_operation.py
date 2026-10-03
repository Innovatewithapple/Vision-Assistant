import mujoco
import numpy as np
import cv2
from YOLO.segmentation import Segmentation
from Visualisation.visualisation import Draw_Segmentation
from Visualisation.visualisation import disable_detection_and_get_bottles, approach_bottle, set_truth, BOTTLE_HEIGHT
from Calculation.panda_ik import PandaIK, drive_to_target
from Calculation.side_ik import side_reach_test
from Calculation.side_grasp import plan_side_grasp, GRIPPER_CLOSED

#-----Segmentation---@
segmentor = Segmentation()

# ============================================================
# GRASP SETTINGS  (tune these)
# ============================================================
GRASP_DEPTH_BELOW_TOP = 0.03   # m: how far the finger center goes BELOW the top of the bottle cap
                               # bigger number = lower / deeper
DESCENT_STEP = 0.01            # m: go down this much at a time (keeps the hand centered)
WAYPOINT_TOL = 0.004           # m: how close to each waypoint before going to the next one
SETTLED_VEL = 0.02             # rad/s: arm must be nearly still before next waypoint


def solve_ik_safe(model, data, ik, target):
    """Solve IK starting from the live arm pose, then put the live pose back.
    Returns (joint targets, error in meters)."""
    saved = data.qpos[:7].copy()
    q = ik.solve(target)
    center = (data.xpos[ik.left_finger_id] + data.xpos[ik.right_finger_id]) / 2
    err = np.linalg.norm(center - target)
    data.qpos[:7] = saved
    mujoco.mj_forward(model, data)
    return q, err

def reach_test(model, data, ik, xy, z_list):
    jr = np.array([model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
                   for i in range(1, 8)])
    for z in z_list:
        q, err = solve_ik_safe(model, data, ik, np.array([xy[0], xy[1], z]))
        at_limit = [i + 1 for i in range(7) if q[i] <= jr[i, 0] + 1e-3 or q[i] >= jr[i, 1] - 1e-3]
        print(f"z={z:.3f}  ik_err={err*1000:6.1f} mm  joints at limit: {at_limit}")

# bar_operation.py (top of file, after imports)
def bottle_truth(model, data, name="bottle_1_geom"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    mid = model.geom_dataid[gid]
    adr, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[adr:adr + n]
    R = data.geom_xmat[gid].reshape(3, 3)
    w = verts @ R.T + data.geom_xpos[gid]
    center_xy = (w[:, :2].min(0) + w[:, :2].max(0)) / 2
    print("body origin    :", data.xpos[model.geom_bodyid[gid]])
    print("base z / top z :", w[:, 2].min(), w[:, 2].max())
    print("axis center xy :", center_xy)
    print("radius         :", (w[:, :2].max(0) - w[:, :2].min(0)).max() / 2)
    print("height         :", w[:, 2].max() - w[:, 2].min())
    return center_xy, w[:, 2].min()

def bottle_profile(model, data, name="bottle_1_geom"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    mid = model.geom_dataid[gid]
    adr, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    w = model.mesh_vert[adr:adr + n] @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
    c = (w[:, :2].min(0) + w[:, :2].max(0)) / 2
    base = w[:, 2].min()
    for h in np.arange(0.02, 0.30, 0.02):
        sel = np.abs(w[:, 2] - (base + h)) < 0.005
        if sel.any():
            r = np.linalg.norm(w[sel, :2] - c, axis=1).max()
            print(f"height above base {h:.2f} m  ->  radius {r*100:.1f} cm   (z = {base + h:.3f})")

def grip_report(model, data, label, bottle_geom="bottle_1_geom"):
    j1 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
    j2 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")]
    print(f"[{label}] finger opening = {(data.qpos[j1] + data.qpos[j2]) * 1000:.1f} mm  (0 = closed, 80 = fully open)")
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, bottle_geom)
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if gid in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == gid else c.geom1
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other]))
            print(f"    touching '{body}'   normal force {f[0]:.2f} N")
            n += 1
    if n == 0:
        print("    no contacts with the bottle")

def run_operation(model, data, viewer, renderer):
    print("Operation connected to scene.")
    bottle_truth(model, data)   
    bottle_profile(model,data) 
    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    ik = PandaIK(model, data)

    phase = "scan"
    step_count = 0
    start_q = None
    approach_steps = 0
    APPROACH_STEPS = 1500     # about 3 s of simulated time (2 ms per step)
    arm_dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
            for i in range(1, 8)]
    arm_qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
            for i in range(1, 8)]
    SCAN_HOLD_STEPS = 300   # how long to sit at the scan pose before moving on

    # exact actuator values you gave
    SCAN_POSE_CTRL = np.array([1.59, -0.494, 0, -0.1, 0, 1.04, 0, 255])

    target_q = None
    gripper_ctrl = None
    settle_counter = 0
    grasp_xy = None
    grasp_z = None
    waypoint_z = None

    path = []
    lift_path = []
    idx = 0
    wp_steps = 0
    close_steps = 0
    CLOSE_STEPS = 400   

    while viewer.is_running():
        mujoco.mj_step(model, data)
        step_count += 1
        if step_count == 500:   
            true_xy, base_z = bottle_truth(model, data)  
            bottle_profile(model,data) 
            set_truth(true_xy) 
        renderer.update_scene(data, camera='wrist_camera')
        wrist_frame = renderer.render()
        wrist_frame = np.asarray(wrist_frame)
        wrist_frame_bgr = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)

        if phase == "scan":
            boxes, labels, scores, mask, track_ids, class_names = segmentor.segment(frame=wrist_frame_bgr)
            wrist_frame_bgr = Draw_Segmentation(wrist_frame_bgr, boxes, labels, scores, mask, track_ids, class_names, model, data)
        display_frame = cv2.cvtColor(wrist_frame_bgr, cv2.COLOR_BGR2RGB)

        is_wrist = (
            viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
            and viewer.cam.fixedcamid == wrist_camera_id
        )
        if is_wrist:
            vp = viewer.viewport
            resized_frame = cv2.resize(display_frame, (vp.width, vp.height))
            viewport_rect = mujoco.MjrRect(vp.left, vp.bottom, vp.width, vp.height)
            viewer.set_images((viewport_rect, resized_frame))
        else:
            viewer.clear_images()

        # --------------------------------------------------
        # PHASE LOGIC:  scan -> go_in -> close -> lift -> hold
        # --------------------------------------------------
        if phase == "scan":
            data.ctrl[:] = SCAN_POSE_CTRL
            joint_err = np.abs(data.qpos[arm_qpos] - SCAN_POSE_CTRL[:7])
            joint_err[3] = 0          # joint 4 sits at its limit, ignore it

            if joint_err.max() < 0.05:
                settle_counter += 1
            else:
                settle_counter = 0

            if settle_counter >= SCAN_HOLD_STEPS:
                bottles = disable_detection_and_get_bottles()
                print(f"Scan complete. {len(bottles)} bottles registered.")

                if len(bottles) == 0:
                    print("No bottles found - staying idle.")
                    phase = "idle"
                else:
                    plan = plan_side_grasp(model, data, ik, np.array(bottles[0]["pos"][:2]), arm_qpos)
                    if plan is None:
                        print("Side grasp planning failed - staying idle.")
                        phase = "idle"
                    else:
                        path, lift_path = plan
                        idx = 0
                        wp_steps = 0
                        phase = "go_in"

        elif phase in ("go_in", "lift"):
            wp_list = path if phase == "go_in" else lift_path
            q_wp, tgt, grip = wp_list[idx]
            drive_to_target(model, data, q_wp, gripper_ctrl=grip)
            wp_steps += 1

            center = (data.xpos[ik.left_finger_id] + data.xpos[ik.right_finger_id]) / 2
            still = np.abs(data.qvel[arm_dofs]).max() < SETTLED_VEL

            if np.linalg.norm(center - tgt) < WAYPOINT_TOL and still:
                print(f"  {phase}: waypoint {idx + 1}/{len(wp_list)} reached")
                idx += 1
                wp_steps = 0
                if idx >= len(wp_list):
                    if phase == "go_in":
                        close_steps = 0
                        phase = "close"
                    else:
                        bz = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle_1")][2]
                        print(f"Lift done. Bottle z = {bz:.3f}  (starts at 0.485)")
                        grip_report(model, data, "after lift")
                        phase = "hold"
            elif wp_steps > 2500:
                print(f"{phase}: waypoint {idx + 1} timed out - stopping.")
                phase = "idle"

        elif phase == "close":
            drive_to_target(model, data, path[-1][0], gripper_ctrl=GRIPPER_CLOSED)
            close_steps += 1
            if close_steps >= CLOSE_STEPS:
                grip_report(model, data, "after close")
                idx = 0
                wp_steps = 0
                phase = "lift"

        elif phase == "hold":
            drive_to_target(model, data, lift_path[-1][0], gripper_ctrl=GRIPPER_CLOSED)

        viewer.sync()