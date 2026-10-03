import mujoco
import numpy as np


def _arm_ids(model):
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 8)]
    qadr = np.array([model.jnt_qposadr[j] for j in jids])
    dadr = np.array([model.jnt_dofadr[j] for j in jids])
    jr = model.jnt_range[jids]
    return qadr, dadr, jr


def _finger_center(data, ik):
    return (data.xpos[ik.left_finger_id] + data.xpos[ik.right_finger_id]) / 2


def _rot_error_vec(R, R_des):
    return 0.5 * (np.cross(R[:, 0], R_des[:, 0])
                  + np.cross(R[:, 1], R_des[:, 1])
                  + np.cross(R[:, 2], R_des[:, 2]))


def _rot_angle(R, R_des):
    return np.arccos(np.clip((np.trace(R_des.T @ R) - 1) / 2, -1, 1))


def side_orientation(approach_dir_xy, flip=False):
    """Hand z points horizontally along approach_dir; hand y (finger opening) is horizontal."""
    z = np.array([approach_dir_xy[0], approach_dir_xy[1], 0.0])
    z /= np.linalg.norm(z)
    y = np.array([-z[1], z[0], 0.0])
    if flip:
        y = -y
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def best_seeds(model, data, ik, target, R_des, n_best=6):
    """Screen a grid of arm poses with forward kinematics only (no IK) and return the
    poses whose hand is closest to the wanted position and orientation."""
    qadr, dadr, jr = _arm_ids(model)
    saved = data.qpos.copy()

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_base")
    local = data.xmat[base_id].reshape(3, 3).T @ (target - data.xpos[base_id])
    q1_0 = np.arctan2(local[1], local[0])          # joint 1 that faces the target

    q1s = q1_0 + np.array([-0.3, 0.0, 0.3])
    q2s = [-0.3, 0.0, 0.3, 0.6]
    q4s = [-1.2, -1.6, -2.0, -2.4, -2.8]
    q6s = np.linspace(0.2, 3.5, 12)
    q7s = np.linspace(-2.6, 2.8, 8)

    scored = []
    for q1 in q1s:
        for q2 in q2s:
            for q4 in q4s:
                for q6 in q6s:
                    for q7 in q7s:
                        q = np.array([q1, q2, 0.0, q4, 0.0, q6, q7])
                        if np.any(q < jr[:, 0]) or np.any(q > jr[:, 1]):
                            continue
                        data.qpos[qadr] = q
                        mujoco.mj_kinematics(model, data)
                        R = data.xmat[ik.hand_body_id].reshape(3, 3)
                        score = (np.linalg.norm(_finger_center(data, ik) - target)
                                 + 0.3 * _rot_angle(R, R_des))
                        scored.append((score, q))

    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    scored.sort(key=lambda s: s[0])
    return [q for _, q in scored[:n_best]]


def solve_side(model, data, ik, target, R_des, q_start=None,
               iterations=1500, step=0.3, damping=0.05):
    """6D IK: finger center -> target, hand orientation -> R_des.
    Starts from q_start (or the live arm pose) and restores the live pose afterwards.
    Returns (joint targets, position error in m, orientation error in degrees)."""
    qadr, dadr, jr = _arm_ids(model)
    saved = data.qpos.copy()
    q = data.qpos[qadr].copy() if q_start is None else np.array(q_start, dtype=float)

    for _ in range(iterations):
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)

        pl = data.xpos[ik.left_finger_id].copy()
        pr = data.xpos[ik.right_finger_id].copy()
        center = (pl + pr) / 2

        jpl = np.zeros((3, model.nv)); jrl = np.zeros((3, model.nv))
        jpr = np.zeros((3, model.nv)); jrr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jpl, jrl, pl, ik.left_finger_id)
        mujoco.mj_jac(model, data, jpr, jrr, pr, ik.right_finger_id)
        Jp = ((jpl + jpr) / 2)[:, dadr]

        jph = np.zeros((3, model.nv)); jrh = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jph, jrh, data.xpos[ik.hand_body_id].copy(), ik.hand_body_id)
        Jr = jrh[:, dadr]

        R = data.xmat[ik.hand_body_id].reshape(3, 3)
        e_rot = _rot_error_vec(R, R_des)
        e_pos = target - center

        J = np.vstack([Jp, Jr])
        e = np.concatenate([e_pos, e_rot])
        dq = J.T @ np.linalg.solve(J @ J.T + damping ** 2 * np.eye(6), e)
        q = np.clip(q + step * dq, jr[:, 0], jr[:, 1])

        if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_rot) < 1e-3:
            break

    data.qpos[qadr] = q
    mujoco.mj_forward(model, data)
    pos_err = np.linalg.norm(target - _finger_center(data, ik))
    R = data.xmat[ik.hand_body_id].reshape(3, 3)
    rot_err = np.degrees(_rot_angle(R, R_des))

    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return q, pos_err, rot_err


def solve_side_multi(model, data, ik, target, R_des, n_seeds=6):
    """Try IK from several good starting poses and keep the best result."""
    best = None
    for seed in best_seeds(model, data, ik, target, R_des, n_best=n_seeds):
        q, pe, re = solve_side(model, data, ik, target, R_des, q_start=seed)
        score = pe + 0.3 * np.radians(re)
        if best is None or score < best[3]:
            best = (q, pe, re, score)
        if pe < 0.002 and re < 1.0:
            break
    return best[0], best[1], best[2]


def side_reach_test(model, data, ik, bottle_xy, z_list, backoff=0.12):
    bottle_xy = np.asarray(bottle_xy, dtype=float)
    base = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_base")][:2]
    d = bottle_xy - base
    d_unit = d / np.linalg.norm(d)
    for flip in (False, True):
        R_des = side_orientation(d, flip)
        for z in z_list:
            for name, xy in (("pre-grasp", bottle_xy - backoff * d_unit), ("grasp", bottle_xy)):
                target = np.array([xy[0], xy[1], z])
                _, pe, re = solve_side_multi(model, data, ik, target, R_des)
                print(f"flip={flip!s:5}  z={z:.3f}  {name:9}  pos_err={pe*1000:6.1f} mm  tilt_err={re:5.1f} deg",
                      flush=True)