import mujoco
import numpy as np
from Calculation.side_ik import side_orientation, solve_side, solve_side_multi   # use the same package path style as your panda_ik import

# ============================================================
# SIDE GRASP SETTINGS (tune these)
# ============================================================
SIDE_GRASP_Z = 0.745      # world z of the finger pads during the grasp (bottle neck region)
PRE_BACKOFF = 0.12       # m: pre-grasp pose is this far back from the bottle, toward the robot
STEP_LEN = 0.02          # m: spacing of waypoints while sliding in and lifting
LIFT_HEIGHT = 0.10       # m: how far to lift the bottle
GRIPPER_OPEN = 255
GRIPPER_CLOSED = 0
POS_TOL = 0.003          # m: a planned pose must be this accurate or planning fails
ROT_TOL_DEG = 1.0


def pad_offset(model, data, ik):
    """Distance from the finger-body origins to the middle of the finger pads, along the hand z axis.
    The IK target is the midpoint of the two finger bodies, which sits this far BEHIND the pads."""
    z = data.xmat[ik.hand_body_id].reshape(3, 3)[:, 2]
    vals = []
    for fid in (ik.left_finger_id, ik.right_finger_id):
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == fid and model.geom_contype[g] > 0:
                vals.append(float(np.dot(data.geom_xpos[g] - data.xpos[fid], z)))
    off = float(np.mean(vals)) if vals else 0.04
    if not (0.0 < off < 0.08):
        off = 0.04
    return off


def plan_side_grasp(model, data, ik, bottle_xy, arm_qpos):
    """Plan: pre-grasp pose -> slide in toward the bottle -> (close) -> lift.
    Returns (path, lift_path); each entry is (joint targets, finger-midpoint target, gripper command).
    Returns None if any pose cannot be solved accurately."""
    bottle_xy = np.asarray(bottle_xy, dtype=float)
    base = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_base")][:2]
    d = bottle_xy - base
    d_unit = d / np.linalg.norm(d)

    off = pad_offset(model, data, ik)
    axis_xy = bottle_xy - off * d_unit            # finger-body midpoint when the pads surround the bottle axis
    pre_xy = axis_xy - PRE_BACKOFF * d_unit
    print(f"[plan] pad offset {off*1000:.0f} mm, grasp height {SIDE_GRASP_Z:.3f}")

    q_now = data.qpos[arm_qpos].copy()
    pre_target = np.array([pre_xy[0], pre_xy[1], SIDE_GRASP_Z])

    best = None
    for flip in (False, True):                    # two hand rotations (180 deg apart); keep the one closer to the current pose
        R_des = side_orientation(d, flip)
        q_pre, pe, re = solve_side_multi(model, data, ik, pre_target, R_des)
        print(f"[plan] flip={flip!s:5} pre-grasp: pos_err={pe*1000:.1f} mm  tilt_err={re:.1f} deg")
        if pe < POS_TOL and re < ROT_TOL_DEG:
            travel = float(np.abs(q_pre - q_now).sum())
            if best is None or travel < best[0]:
                best = (travel, flip, R_des, q_pre)
    if best is None:
        print("[plan] no accurate pre-grasp pose found")
        return None
    _, flip, R_des, q_pre = best
    print(f"[plan] using flip={flip}")

    path = [(q_pre, pre_target, GRIPPER_OPEN)]
    q = q_pre
    n = int(np.ceil(PRE_BACKOFF / STEP_LEN))
    for k in range(1, n + 1):
        xy = pre_xy + (axis_xy - pre_xy) * k / n
        tgt = np.array([xy[0], xy[1], SIDE_GRASP_Z])
        q, pe, re = solve_side(model, data, ik, tgt, R_des, q_start=q)   # start from the previous pose: stays continuous
        if pe > POS_TOL or re > ROT_TOL_DEG:
            print(f"[plan] slide-in step {k}/{n} failed: pos_err={pe*1000:.1f} mm tilt_err={re:.1f} deg")
            return None
        path.append((q, tgt, GRIPPER_OPEN))

    lift_path = []
    nl = int(np.ceil(LIFT_HEIGHT / STEP_LEN))
    for k in range(1, nl + 1):
        tgt = np.array([axis_xy[0], axis_xy[1], SIDE_GRASP_Z + LIFT_HEIGHT * k / nl])
        q, pe, re = solve_side(model, data, ik, tgt, R_des, q_start=q)
        if pe > POS_TOL or re > ROT_TOL_DEG:
            print(f"[plan] lift step {k}/{nl} failed: pos_err={pe*1000:.1f} mm tilt_err={re:.1f} deg")
            return None
        lift_path.append((q, tgt, GRIPPER_CLOSED))

    print(f"[plan] OK: {len(path)} waypoints in, {len(lift_path)} to lift")
    return path, lift_path