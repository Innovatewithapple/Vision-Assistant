import mujoco
import numpy as np
from Calculation.side_ik import side_orientation, solve_side, solve_side_multi   # use the same package path style as your panda_ik import

# ============================================================
# SIDE GRASP SETTINGS (tune these)
# ============================================================
SIDE_GRASP_Z = 0.745      # world z of the finger pads during the grasp (bottle neck region)
PRE_BACKOFF = 0.12       # m: pre-grasp pose is this far back from the bottle, toward the robot
STEP_LEN = 0.06          # m: spacing of waypoints while sliding in and lifting
LIFT_HEIGHT = 0.10       # m: how far to lift the bottle
GRIPPER_OPEN = 255
GRIPPER_CLOSED = 70
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
    return path, lift_path, R_des, off


#----------Carry to the Bin----------!
import mujoco
import numpy as np
from Calculation.side_ik import solve_side
# ============================================================
# PLACE SETTINGS (tune these)
# ============================================================
CARRY_STEP = 0.06          # m: spacing of waypoints while carrying
MAX_YAW_STEP = np.radians(12)   # max hand turn between two waypoints
HAND_CLEAR = 0.06          # m: finger height must stay at least this far above the bin rim
BOTTLE_BASE_CLEAR = 0.04   # m: bottle base must be this far above the bin rim while carrying
RELEASE_GAP = 0.005         # m: bottle base is this far above the bin floor when released
RETREAT_DIST = 0.12        # m: how far the hand backs away after letting go
POS_TOL = 0.005            # m: carry poses may be slightly looser than the grasp poses
ROT_TOL_DEG = 2.0
RELEASE_STEPS = 300        # sim steps to hold still while the gripper opens
FINAL_YAW_OFFSETS_DEG = (0, 15, -15, 30, -30, 45, -45, 60, -60, 90, -90)   # final hand directions tried, relative to "pointing at the bin from the base"
PLACE_EXTRA_DROP = (0.0, 0.04, 0.08)   # m: if the lowest release height is unreachable, release a bit higher
ALLOW_LONG_WAY = False     # True = last resort: turn the hand the long way round (swings the arm AWAY from the bin first)
 
 
def bin_info(model):
    """Read the bin collider geoms created in bar_bottle_view.py.
    Returns (center_xy, floor_top_z, rim_z) or None if the colliders are missing."""
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_floor_col")
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_wall_xm")
    if fid < 0 or wid < 0:
        return None
    center_xy = model.geom_pos[fid][:2].copy()
    floor_top = float(model.geom_pos[fid][2] + model.geom_size[fid][2])
    rim = float(model.geom_pos[wid][2] + model.geom_size[wid][2])
    return center_xy, floor_top, rim
 
 
# ============================================================
# BIN SLOTS (2 x 2 grid, one bottle per corner)
# ============================================================
SLOT_FILL_ORDER = "far_first"   # "far_first": far row first (the arm never reaches over a placed bottle)
                                # "near_first": near row first (carry height is raised automatically to clear placed bottles)
BOTTLE_HEIGHT_M = 0.30
SLOT_SPREAD = 1.0               # 1.0 = bottles go right into the corners; 0.5 = halfway between the corners and the bin center
SLOT_MARGIN_DEPTH = 0.075       # m: closest the bottle AXIS may get to the far/near wall (bottle radius 0.037 + 1.3 cm)
SLOT_MARGIN_LATERAL = 0.057     # m: closest the axis may get to the left/right walls (smaller = pairs spread further apart)
 
# ---- direct control: move the slots yourself (all in meters) ----
SLOT_PITCH_DEPTH = None         # distance between the far row and the near row (center to center). None = from the margins above
SLOT_PITCH_LATERAL = None       # distance between left and right bottle in a row (center to center). None = from the margins above
SLOT_NUDGE = {}                 # move one slot in WORLD x / y, e.g. {1: (0.00, 0.02), 4: (-0.01, 0.0)}  (slot number: (dx, dy))
 
 
def slot_centers(model, base_xy, verbose=True):
    """The 4 slot centers (xy), sorted in fill order, as seen by the robot looking toward the bin:
    far_first -> 1: far-left, 2: far-right, 3: near-left, 4: near-right."""
    fid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_floor_col")
    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_wall_xm")
    C = model.geom_pos[fid][:2].copy()
    wall_t = 2.0 * model.geom_size[wid][0]
    ihx = model.geom_size[fid][0] - wall_t               # inside half sizes of the bin
    ihy = model.geom_size[fid][1] - wall_t
 
    u = C - np.asarray(base_xy, float)
    u = u / np.linalg.norm(u)                            # direction robot -> bin
    depth_is_x = abs(u[0]) >= abs(u[1])
    mx = SLOT_MARGIN_DEPTH if depth_is_x else SLOT_MARGIN_LATERAL
    my = SLOT_MARGIN_LATERAL if depth_is_x else SLOT_MARGIN_DEPTH
    ox = SLOT_SPREAD * max(ihx - mx, 0.0)
    oy = SLOT_SPREAD * max(ihy - my, 0.0)
    p_depth, p_lat = SLOT_PITCH_DEPTH, SLOT_PITCH_LATERAL          # explicit spacing overrides the margins
    px, py = (p_depth, p_lat) if depth_is_x else (p_lat, p_depth)
    if px is not None:
        ox = px / 2
    if py is not None:
        oy = py / 2
 
    items = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = C + np.array([sx * ox, sy * oy])
            row = sx * np.sign(u[0]) if depth_is_x else sy * np.sign(u[1])    # +1 = far side
            d = p - C
            lat = u[0] * d[1] - u[1] * d[0]                                   # > 0 = left as seen from the robot
            items.append((row, lat, p))
    far = SLOT_FILL_ORDER == "far_first"
    items.sort(key=lambda it: ((-it[0] if far else it[0]), -it[1]))
    items = [(r, l, p + np.array(SLOT_NUDGE.get(k, (0.0, 0.0)), float)) for k, (r, l, p) in enumerate(items, 1)]
 
    if verbose:
        print(f"[place] bin inside {2*ihx*100:.1f} x {2*ihy*100:.1f} cm (x by y); "
              f"slot offsets from bin center {ox*100:.1f} x {oy*100:.1f} cm; order = {SLOT_FILL_ORDER}")
        for k, (row, lat, p) in enumerate(items, 1):
            name = ("far" if row > 0 else "near") + "-" + ("left" if lat > 0 else "right")
            print(f"   slot {k}: {name:10}  xy = ({p[0]:.3f}, {p[1]:.3f})")
        if 2 * ox < 0.085 or 2 * oy < 0.085:
            print("   WARNING: slots are closer together than one bottle diameter + gap - the bin is too small for 4 bottles with these margins")
    return [it[2] for it in items]
 
 
SLOT_COLORS = [(0.1, 0.9, 0.2, 0.8), (0.2, 0.5, 1.0, 0.8), (1.0, 0.9, 0.1, 0.8), (1.0, 0.2, 0.2, 0.8)]   # slot 1..4 = green, blue, yellow, red
 
 
def draw_slot_markers(viewer, model, base_xy):
    """Draw a flat colored disc on the bin floor at every slot, so you can SEE where bottles will go."""
    info = bin_info(model)
    if info is None:
        return
    _, floor_top, _ = info
    scn = viewer.user_scn
    scn.ngeom = 0
    for k, p in enumerate(slot_centers(model, base_xy, verbose=False)):
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_CYLINDER,
                            np.array([0.037, 0.002, 0.0]), np.array([p[0], p[1], floor_top + 0.003]),
                            np.eye(3).flatten(), np.array(SLOT_COLORS[k % 4], dtype=np.float32))
        scn.ngeom += 1
 
 
def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi
 
 
def _dir(psi):
    return np.array([np.cos(psi), np.sin(psi)])
 
 
def _pose(axis_xy, z, psi, off, flip):
    """Finger-midpoint target + hand orientation that hold the bottle axis at axis_xy
    while the hand points along yaw psi."""
    d = _dir(psi)
    tgt = np.array([axis_xy[0] - off * d[0], axis_xy[1] - off * d[1], z])
    return tgt, side_orientation(d, flip)
 
 
PULL_BACK_TRIES = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)   # m: pull-back distances tried, smallest first
BOTTLE_CLEAR_DIST = 0.10   # m: held-bottle axis must stay this far from the axis of every other bottle


def _seg_dist(p, a, b):
    """Distance from point p to the segment a-b (2D)."""
    ab = b - a
    L2 = float(np.dot(ab, ab))
    t = 0.0 if L2 < 1e-12 else min(max(float(np.dot(p - a, ab)) / L2, 0.0), 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _try_variant(model, data, ik, q_start, A0, V, C, z0, carry_z, place_z, psi0, dpsi, flip, off):
    """Rise -> pull back to V -> carry to C (turning the hand while moving) -> descend.
    Returns (path, last_pose, None, False) or (None, None, why, failed_in_rise_or_pullback)."""
    poses = []
    if carry_z - z0 > 1e-6:
        n = int(np.ceil((carry_z - z0) / CARRY_STEP))
        for k in range(1, n + 1):
            poses.append(_pose(A0, z0 + (carry_z - z0) * k / n, psi0, off, flip))
    dv = float(np.linalg.norm(V - A0))
    if dv > 1e-6:                                    # pull back, hand keeps pointing the same way
        n_v = int(np.ceil(dv / CARRY_STEP))
        for k in range(1, n_v + 1):
            poses.append(_pose(A0 + (V - A0) * k / n_v, carry_z, psi0, off, flip))
    n_prefix = len(poses)
    dist = np.linalg.norm(C - V)
    n_h = max(1, int(np.ceil(dist / CARRY_STEP)), int(np.ceil(abs(dpsi) / MAX_YAW_STEP)))
    for k in range(1, n_h + 1):
        s = k / n_h
        poses.append(_pose(V + (C - V) * s, carry_z, psi0 + dpsi * s, off, flip))
    n_d = max(1, int(np.ceil((carry_z - place_z) / CARRY_STEP)))
    for k in range(1, n_d + 1):
        poses.append(_pose(C, carry_z + (place_z - carry_z) * k / n_d, psi0 + dpsi, off, flip))

    q = q_start
    path = []
    for k, (tgt, R) in enumerate(poses, 1):
        q_new, pe, re = solve_side(model, data, ik, tgt, R, q_start=q)
        if pe > POS_TOL or re > ROT_TOL_DEG:      # retry harder before giving up
            q_new, pe, re = solve_side(model, data, ik, tgt, R, q_start=q,
                                       iterations=5000, step=0.2, damping=0.03)
        q = q_new
        if pe > POS_TOL or re > ROT_TOL_DEG:
            return None, None, (f"step {k}/{len(poses)} at {tgt.round(3)}: pos_err={pe*1000:.1f} mm "
                                f"tilt_err={re:.1f} deg"), k <= n_prefix
        path.append((q, tgt, GRIPPER_CLOSED))
    return path, (q, poses[-1][0], poses[-1][1]), None, False


def plan_place(model, data, ik, q_start, start_target, R_des, off, bottle_base_z, slot_index=0, obstacles=None):
    """Plan: rise -> pull back from the counter row -> carry over the bin while turning the hand -> descend
    -> release -> retreat. `obstacles` = xy of the other bottles still standing on the counter.
    Returns (carry_path, retreat_path) with entries (joint targets, finger target, gripper command), or None."""
    info = bin_info(model)
    if info is None:
        print("[place] bin colliders not found in the model")
        return None
    C, floor_top, rim = info

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_base")
    base_xy = data.xpos[base_id][:2].copy()

    slots = slot_centers(model, base_xy)
    C = slots[slot_index % len(slots)]
    print(f"[place] slot {slot_index % len(slots) + 1}/{len(slots)} ({SLOT_FILL_ORDER}) -> {C.round(3)}")

    start = np.asarray(start_target, float)
    d0 = R_des[:2, 2] / np.linalg.norm(R_des[:2, 2])
    psi0 = float(np.arctan2(d0[1], d0[0]))
    flip = bool(R_des[2, 0] > 0)                      # hand x up <=> flip=True in side_orientation
    A0 = start[:2] + off * d0                         # bottle axis while held

    b2f = SIDE_GRASP_Z - bottle_base_z                # finger height above the bottle base
    carry_z = max(start[2], rim + BOTTLE_BASE_CLEAR + b2f, rim + HAND_CLEAR)
    if slot_index > 0 and SLOT_FILL_ORDER == "near_first":      # must clear the bottles already standing in the bin
        carry_z = max(carry_z, floor_top + BOTTLE_HEIGHT_M + BOTTLE_BASE_CLEAR + b2f)
    place_z = max(floor_top + RELEASE_GAP + b2f, rim + HAND_CLEAR)
    drop = place_z - b2f - floor_top
    print(f"[place] bin target {C.round(3)}  floor top z={floor_top:.3f}  rim z={rim:.3f}")
    print(f"[place] carry z={carry_z:.3f}  place z={place_z:.3f}  drop height={drop*100:.1f} cm")

    # ---- pull-back candidates: go back toward the robot, away from the bottles on the counter ----
    others = [np.asarray(o, float)[:2] for o in (obstacles or [])]
    back = base_xy - A0
    back = back / np.linalg.norm(back)
    vias = []
    for pull in PULL_BACK_TRIES:
        V = A0 + back * pull
        clear = min([min(_seg_dist(o, A0, V), _seg_dist(o, V, C)) for o in others], default=99.0)
        if clear >= BOTTLE_CLEAR_DIST:
            vias.append((pull, V))
    if not vias:
        print("[place] no pull-back distance keeps the held bottle clear of the other bottles")
        return None
    print(f"[place] {len(others)} other bottles on the counter; trying pull-backs "
          f"{[round(p * 100) for p, _ in vias]} cm")

    psi_out = float(np.arctan2(*(C - base_xy)[::-1]))  # hand pointing from the robot base toward the bin
    variants = []
    for extra in PLACE_EXTRA_DROP:
        for off_deg in FINAL_YAW_OFFSETS_DEG:
            psi1 = psi_out + np.radians(off_deg)
            variants.append((extra, psi1, "short way", _wrap(psi1 - psi0)))
    if ALLOW_LONG_WAY:
        for extra in PLACE_EXTRA_DROP:
            for off_deg in FINAL_YAW_OFFSETS_DEG:
                psi1 = psi_out + np.radians(off_deg)
                ds = _wrap(psi1 - psi0)
                variants.append((extra, psi1, "long way", ds - np.sign(ds) * 2 * np.pi if ds != 0 else 2 * np.pi))

    winner = None
    for pull, V in vias:
        last_why = None
        for extra, psi1, name, dpsi in variants:
            path, last, why, early = _try_variant(model, data, ik, q_start, A0, V, C, start[2],
                                                  carry_z, place_z + extra, psi0, dpsi, flip, off)
            if path is not None:
                print(f"[place] pull-back {pull*100:.0f} cm, final yaw {np.degrees(psi1):6.1f} deg, {name}, "
                      f"turn {np.degrees(dpsi):6.1f} deg, drop {(drop + extra)*100:.0f} cm: OK")
                winner = (path, last, psi0 + dpsi)
                break
            last_why = why
            if early:          # the rise / pull-back itself is unreachable: every yaw would fail the same way
                break
        if winner:
            break
        print(f"[place] pull-back {pull*100:.0f} cm: no variant worked (last: {last_why})")
    if winner is None:
        print("[place] no carry variant worked")
        return None

    carry_path, (q, tgt_end, R_end), psi_f = winner
    d_f = _dir(psi_f)

    retreat_path = []
    n = int(np.ceil(RETREAT_DIST / CARRY_STEP))
    for k in range(1, n + 1):
        tgt = np.array([tgt_end[0] - RETREAT_DIST * d_f[0] * k / n,
                        tgt_end[1] - RETREAT_DIST * d_f[1] * k / n, tgt_end[2]])
        q, pe, re = solve_side(model, data, ik, tgt, R_end, q_start=q)
        if pe > POS_TOL or re > ROT_TOL_DEG:
            print(f"[place] retreat step {k}/{n} failed: pos_err={pe*1000:.1f} mm tilt_err={re:.1f} deg")
            return None
        retreat_path.append((q, tgt, GRIPPER_OPEN))

    print(f"[place] OK: {len(carry_path)} carry waypoints, {len(retreat_path)} retreat waypoints")
    return carry_path, retreat_path
 
 
# ============================================================
# SMOOTH TRAJECTORY (use this instead of waypoint-by-waypoint driving while carrying)
# ============================================================
CARRY_JOINT_SPEED = 1.1    # rad/s: cruise speed of the joint that moves most (lower = gentler)
CARRY_RAMP_TIME = 0.4      # s: time to speed up from rest / slow down to rest (shorter = snappier start)
 
 
class SmoothTrajectory:
    """One continuous joint-space path through a list of joint targets.
    Speed profile: quick ramp up, constant cruise, ramp down. No stop-and-go at the waypoints."""
 
    def __init__(self, q_list, v_cruise=CARRY_JOINT_SPEED, q_first=None, ramp=CARRY_RAMP_TIME):
        pts = [np.asarray(q, float) for q in q_list]
        if q_first is not None:
            pts = [np.asarray(q_first, float)] + pts
        self.pts = np.array(pts)
        seg = np.abs(np.diff(self.pts, axis=0)).max(axis=1) / v_cruise   # seconds per segment at cruise speed
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.L = max(float(self.cum[-1]), 1e-3)          # path length measured in cruise-seconds
        self.tr = min(ramp, self.L)
        self.T = self.L + self.tr                        # total duration in seconds
 
    def _progress(self, t):
        t = min(max(t, 0.0), self.T)
        if t < self.tr:
            return 0.5 * t * t / self.tr
        if t < self.T - self.tr:
            return 0.5 * self.tr + (t - self.tr)
        r = self.T - t
        return self.L - 0.5 * r * r / self.tr
 
    def q_at(self, t):
        s = self._progress(t)
        i = int(np.searchsorted(self.cum, s, side="right")) - 1
        i = min(max(i, 0), len(self.pts) - 2)
        span = self.cum[i + 1] - self.cum[i]
        f = (s - self.cum[i]) / span if span > 1e-9 else 1.0
        return self.pts[i] + (self.pts[i + 1] - self.pts[i]) * min(max(f, 0.0), 1.0)