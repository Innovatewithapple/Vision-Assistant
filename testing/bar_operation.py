import mujoco
import numpy as np
import cv2
import time
from YOLO.segmentation import Segmentation
from Visualisation.visualisation import (
    Draw_Segmentation, disable_detection_and_get_bottles, detected_bottles,
    nearest_truth, set_truth, set_truth_all, print_registration_report,
)
from Calculation.panda_ik import PandaIK, drive_to_target
from Calculation.side_grasp import (
    plan_side_grasp, GRIPPER_CLOSED, GRIPPER_OPEN, plan_place, RELEASE_STEPS,
    SmoothTrajectory, CARRY_JOINT_SPEED, draw_slot_markers,
)

#-----Segmentation---@
segmentor = Segmentation()

# ============================================================
# SCAN POSES  (tune these)
# Each pose is held until the camera has registered `need` bottles in total
# (waiting SCAN_AFTER_FOUND_SECONDS after that), or until SCAN_MAX_SECONDS pass,
# then the arm moves to the next pose.  Times are real (wall-clock) seconds.
# ctrl = actuator1..7 (joint targets), actuator8 (gripper, 255 = open)
# ============================================================
SCAN_POSES = [
    dict(ctrl=np.array([2.00, -0.494, 0, -0.1, 0, 0.793, 0, 255]), need=2),   # first two bottles
    dict(ctrl=np.array([2.64, -0.494, 0, -0.1, 0, 0.793, 0, 255]), need=4),   # remaining two bottles
]
SCAN_MAX_SECONDS = 10.0          # most time spent at one scan pose; then move on with whatever was found
SCAN_AFTER_FOUND_SECONDS = 5.0   # once the expected bottles are registered, keep looking this long, then move on
SCAN_SEGMENT_EVERY = 10          # while scanning, run the camera + YOLO only every N sim steps
                                 # (the scene is still, and YOLO on every step is what makes scanning slow)

# ============================================================
# RETURN SPEED  (tune this)
# ============================================================
EMPTY_JOINT_SPEED = 2.0   # rad/s: ONLY the retreat + home moves after releasing a bottle (arm is empty).
                          # Carrying a bottle still uses CARRY_JOINT_SPEED. 1.1 = old speed.

# ============================================================
# GRASP SETTINGS  (tune these)
# ============================================================
WAYPOINT_TOL = 0.004           # m: how close to each waypoint before going to the next one
SETTLED_VEL = 0.02             # rad/s: arm must be nearly still before next waypoint
# Closing the gripper: quick while the fingers are still far from the bottle, very slow through the contact point.
# (the fingers touch a neck that is centred between them at a gripper command of about 94)
CLOSE_SLOW_START = 140         # gripper command where the slow part begins (255 = open, GRIPPER_CLOSED = firm grip)
CLOSE_FAST_STEPS = 120         # steps to go from fully open to CLOSE_SLOW_START (~0.25 s)
CLOSE_SLOW_STEPS = 600         # steps for the slow squeeze from CLOSE_SLOW_START to GRIPPER_CLOSED (more = slower)
CLOSE_STEPS = CLOSE_FAST_STEPS + CLOSE_SLOW_STEPS + 100   # total close phase, includes a short hold
RELEASE_FINGER_OPEN = 0.036    # m: after letting go, retreat only when EACH finger is at least this open (40 mm = full)


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


def grip_report(model, data, label, bottle_body="bottle_1"):
    j1 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
    j2 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")]
    print(f"[{label}] finger opening = {(data.qpos[j1] + data.qpos[j2]) * 1000:.1f} mm  (0 = closed, 80 = fully open)"
          f"   finger1 = {data.qpos[j1] * 1000:.1f} mm, finger2 = {data.qpos[j2] * 1000:.1f} mm  (40 each = fully open)")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bottle_body)
    bottle_geoms = {g for g in range(model.ngeom) if model.geom_bodyid[g] == bid}
    n = 0
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in bottle_geoms or c.geom2 in bottle_geoms:
            other = c.geom2 if c.geom1 in bottle_geoms else c.geom1
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, f)
            body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[other]))
            print(f"    touching '{body}'   normal force {f[0]:.2f} N   gap {c.dist * 1000:+.1f} mm  (negative = finger inside the bottle)")
            n += 1
    if n == 0:
        print("    no contacts with the bottle")


def run_operation(model, data, viewer, renderer):
    print("Operation connected to scene.")

    # ground truth of every bottle, saved immediately (used for the comparison prints)
    true_xy, base_z = bottle_truth(model, data)
    set_truth(true_xy)
    set_truth_all(model, data)

    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    ik = PandaIK(model, data)
    # base_xy = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_base")][:2].copy()
    # draw_slot_markers(viewer, model, base_xy)

    arm_dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
                for i in range(1, 8)]
    arm_qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
                for i in range(1, 8)]
    dt = model.opt.timestep
    f1 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")]
    f2 = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint2")]

    HOME_Q = SCAN_POSES[0]["ctrl"][:7].copy()     # the arm returns here between bottles

    # ---- state ----
    phase = "scan"
    scan_idx = 0
    settle_counter = 0
    scan_t0 = 0.0           # wall-clock time the arm arrived at the current scan pose
    found_t = None          # wall-clock time the expected bottles were all registered
    loop_i = 0
    display_frame = None

    queue = []              # registered bottle ids still to be moved
    n_total = 0
    n_placed = 0
    skipped = []
    cur_id = None
    cur_name = "bottle_1"   # simulator body name of the bottle being moved

    path, lift_path, carry_path, retreat_path = [], [], [], []
    idx = 0
    wp_steps = 0
    close_steps = 0
    release_steps = 0
    traj = None
    traj_steps = 0
    wp_key = None           # which waypoint wp_reach belongs to
    wp_reach = None         # where the fingers end up at that waypoint's joint targets

    while viewer.is_running():
        mujoco.mj_step(model, data)

        loop_i += 1
        scanning = phase == "scan" and settle_counter > 0

        # while holding still at a scan pose, look only every SCAN_SEGMENT_EVERY steps
        if (not scanning) or display_frame is None or loop_i % SCAN_SEGMENT_EVERY == 0:
            renderer.update_scene(data, camera='wrist_camera')
            wrist_frame = np.asarray(renderer.render())
            wrist_frame_bgr = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)

            if scanning:
                boxes, labels, scores, mask, track_ids, class_names = segmentor.segment(frame=wrist_frame_bgr)
                wrist_frame_bgr = Draw_Segmentation(wrist_frame_bgr, boxes, labels, scores, mask, track_ids,
                                                    class_names, model, data)
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

        # --------------------------------------------------------------
        # scan (pose 1, pose 2, ...) -> next_bottle -> go_in -> close -> lift
        #   -> carry -> release -> retreat -> home -> next_bottle -> ... -> done
        # --------------------------------------------------------------
        if phase == "scan":
            pose = SCAN_POSES[scan_idx]
            data.ctrl[:] = pose["ctrl"]
            joint_err = np.abs(data.qpos[arm_qpos] - pose["ctrl"][:7])
            joint_err[3] = 0          # joint 4 sits at its limit, ignore it
            settle_counter = settle_counter + 1 if joint_err.max() < 0.05 else 0

            if settle_counter == 1:                      # arm just arrived and is holding still
                scan_t0 = time.perf_counter()
                found_t = None

            if settle_counter > 0:
                now = time.perf_counter()
                found = len(detected_bottles)
                if found >= pose["need"] and found_t is None:
                    found_t = now                        # all expected bottles are registered
                waited = now - scan_t0

                all_found_and_waited = found_t is not None and now - found_t >= SCAN_AFTER_FOUND_SECONDS
                timed_out = waited >= SCAN_MAX_SECONDS

                if all_found_and_waited or timed_out:
                    why = "expected bottles found" if all_found_and_waited else "time limit"
                    print(f"Scan pose {scan_idx + 1}/{len(SCAN_POSES)} done after {waited:.1f}s ({why}): "
                          f"{found} bottles registered so far (expected {pose['need']}).")
                    scan_idx += 1
                    settle_counter = 0
                    if scan_idx >= len(SCAN_POSES):
                        disable_detection_and_get_bottles()
                        queue = list(detected_bottles.keys())
                        n_total = len(queue)
                        print(f"Scan complete. {n_total} bottles to move.")
                        phase = "next_bottle"

        elif phase == "next_bottle":
            if not queue:
                print(f"All done: {n_placed}/{n_total} bottles placed"
                      + (f", skipped registered ids {skipped}" if skipped else "") + ".")
                phase = "done"
            else:
                cur_id = queue.pop(0)                      # take it out of the queue: it is handled once
                pos = np.array(detected_bottles[cur_id]["pos"])
                near = nearest_truth(pos[:2])[0]
                cur_name = near if near else "bottle_1"
                print(f"\n=== Bottle {n_total - len(queue)}/{n_total}: registered #{cur_id} "
                      f"({cur_name}) at X={pos[0]:.3f}, Y={pos[1]:.3f} -> slot {n_placed + 1} ===")

                plan = plan_side_grasp(model, data, ik, pos[:2], arm_qpos)
                if plan is None:
                    print(f"Grasp planning failed for registered #{cur_id} - skipping it.")
                    skipped.append(cur_id)
                else:
                    path, lift_path, R_des, off = plan
                    # bottles still standing on the counter (waiting in the queue, or skipped earlier)
                    obstacles = [detected_bottles[i]["pos"][:2] for i in queue + skipped]
                    place = plan_place(model, data, ik, lift_path[-1][0], lift_path[-1][1],
                                       R_des, off, base_z, slot_index=n_placed, obstacles=obstacles)
                    if place is None:
                        print(f"Placement planning failed for registered #{cur_id} - skipping it.")
                        skipped.append(cur_id)
                    else:
                        carry_path, retreat_path = place
                        idx = 0
                        wp_steps = 0
                        phase = "go_in"

        elif phase in ("go_in", "lift"):
            wp_list = path if phase == "go_in" else lift_path
            q_wp, tgt, grip = wp_list[idx]
            drive_to_target(model, data, q_wp, gripper_ctrl=grip)
            wp_steps += 1

            # Once per waypoint: where will the fingers really be at these joint targets?
            # (the planner's IK can be a few mm off the ideal target; the arm cannot do better than that)
            if wp_key != (cur_id, phase, idx):
                wp_key = (cur_id, phase, idx)
                saved_q = data.qpos[arm_qpos].copy()
                data.qpos[arm_qpos] = q_wp
                mujoco.mj_kinematics(model, data)
                wp_reach = (data.xpos[ik.left_finger_id] + data.xpos[ik.right_finger_id]) / 2
                data.qpos[arm_qpos] = saved_q
                mujoco.mj_kinematics(model, data)

            center = (data.xpos[ik.left_finger_id] + data.xpos[ik.right_finger_id]) / 2
            still = np.abs(data.qvel[arm_dofs]).max() < SETTLED_VEL

            last = idx == len(wp_list) - 1
            strict = (last and phase != "lift") or (phase == "go_in" and idx == 0)
            dist = np.linalg.norm(center - wp_reach)      # distance to the planned pose (not the ideal target)
            reached = (dist < WAYPOINT_TOL and still) if strict else (dist < 0.010)

            if reached:
                print(f"  {phase}: waypoint {idx + 1}/{len(wp_list)} reached")
                idx += 1
                wp_steps = 0
                if idx >= len(wp_list):
                    idx = 0
                    if phase == "go_in":
                        grip_report(model, data, "before close", cur_name)   # opening should be ~80 mm, no contacts yet
                        close_steps = 0
                        phase = "close"
                    else:
                        bz = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cur_name)][2]
                        print(f"Lift done. {cur_name} z = {bz:.3f}  (starts at {base_z:.3f})")
                        grip_report(model, data, "after lift", cur_name)
                        phase = "carry"
            elif wp_steps > 2500:
                print(f"{phase}: waypoint {idx + 1} timed out - stopping.  "
                      f"(distance to planned pose {dist * 1000:.1f} mm, "
                      f"arm speed {np.abs(data.qvel[arm_dofs]).max():.3f} rad/s)")
                phase = "idle"

        elif phase == "close":
            if close_steps < CLOSE_FAST_STEPS:        # fast part: fingers are still far from the bottle
                grip = GRIPPER_OPEN + (CLOSE_SLOW_START - GRIPPER_OPEN) * close_steps / CLOSE_FAST_STEPS
            else:                                     # slow part: through the contact and up to the firm grip
                frac = min((close_steps - CLOSE_FAST_STEPS) / CLOSE_SLOW_STEPS, 1.0)
                grip = CLOSE_SLOW_START + (GRIPPER_CLOSED - CLOSE_SLOW_START) * frac
            drive_to_target(model, data, path[-1][0], gripper_ctrl=grip)
            close_steps += 1
            if close_steps >= CLOSE_STEPS:
                grip_report(model, data, "after close", cur_name)
                idx = 0
                wp_steps = 0
                phase = "lift"

        elif phase in ("carry", "retreat"):
            if traj is None:
                if phase == "carry":
                    traj = SmoothTrajectory([w[0] for w in carry_path], CARRY_JOINT_SPEED, q_first=lift_path[-1][0])
                else:
                    traj = SmoothTrajectory([w[0] for w in retreat_path], EMPTY_JOINT_SPEED, q_first=carry_path[-1][0])
                traj_steps = 0
                print(f"{phase}: smooth trajectory, {traj.T:.1f} s")

            grip = GRIPPER_CLOSED if phase == "carry" else GRIPPER_OPEN
            drive_to_target(model, data, traj.q_at(traj_steps * dt), gripper_ctrl=grip)
            traj_steps += 1

            t = traj_steps * dt
            settled = np.abs(data.qvel[arm_dofs]).max() < SETTLED_VEL
            if t >= traj.T and settled:
                traj = None
                if phase == "carry":
                    release_steps = 0
                    phase = "release"
                    bz = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cur_name)][2]
                    center = (data.xpos[ik.left_finger_id] + data.xpos[ik.right_finger_id]) / 2
                    print(f"Release start: bottle base z={bz:.3f}, planned finger z={carry_path[-1][1][2]:.3f}, "
                          f"actual finger z={center[2]:.3f}")
                else:
                    b = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cur_name)]
                    n_placed += 1
                    print(f"Placed {cur_name} (registered #{cur_id}) at {b.round(3)}  [{n_placed}/{n_total}]")
                    grip_report(model, data, "after retreat", cur_name)    # any 'touching' line here = still stuck
                    phase = "home"
            elif t > traj.T + 5.0:
                print(f"{phase}: did not settle - stopping.")
                traj = None
                phase = "idle"

        elif phase == "release":
            drive_to_target(model, data, carry_path[-1][0], gripper_ctrl=GRIPPER_OPEN)
            release_steps += 1
            both_open = min(data.qpos[f1], data.qpos[f2]) >= RELEASE_FINGER_OPEN     # one stuck finger blocks this
            if both_open or release_steps >= RELEASE_STEPS:   # RELEASE_STEPS = max wait
                grip_report(model, data, "after release", cur_name)
                idx = 0
                wp_steps = 0
                phase = "retreat"

        elif phase == "home":
            # smooth move back to the first scan pose before planning the next bottle
            if traj is None:
                traj = SmoothTrajectory([HOME_Q], EMPTY_JOINT_SPEED, q_first=data.qpos[arm_qpos].copy())
                traj_steps = 0
                print(f"home: smooth trajectory, {traj.T:.1f} s")
            drive_to_target(model, data, traj.q_at(traj_steps * dt), gripper_ctrl=GRIPPER_OPEN)
            traj_steps += 1

            t = traj_steps * dt
            settled = np.abs(data.qvel[arm_dofs]).max() < SETTLED_VEL
            if t >= traj.T and settled:
                traj = None
                phase = "next_bottle"
            elif t > traj.T + 5.0:
                print("home: did not settle - stopping.")
                traj = None
                phase = "idle"

        elif phase == "done":
            drive_to_target(model, data, HOME_Q, gripper_ctrl=GRIPPER_OPEN)

        viewer.sync()

    print_registration_report()     # final registered-vs-ground-truth table when the window is closed