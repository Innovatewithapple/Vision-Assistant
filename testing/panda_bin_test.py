import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)
from Calculation.panda_ik import PandaIK
from pathlib import Path
from Scenes.panda_scene import SCENE_XML
from YOLO.segmentation import Segmentation, colors
import mujoco
import numpy as np
import cv2
import math


#-------Load Segmentation--------@
segmentor = Segmentation()
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


#-------Load XML---------@
model = mujoco.MjModel.from_xml_string(SCENE_XML)
print("cone:", model.opt.cone, "impratio:", model.opt.impratio, "noslip:", model.opt.noslip_iterations)


#-------OFFSCREEN FRAMEBUFFER--------!
model.vis.global_.offwidth = IMAGE_WIDTH
model.vis.global_.offheight = IMAGE_HEIGHT
data = mujoco.MjData(model)

# ========================================================
# PRINT ALL GEOM NAMES
# ========================================================

print("\n==============================")
print("GEOM NAMES")
print("==============================")

for geom_id in range(model.ngeom):

    geom_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id
    )

    print(
        geom_id,
        ":",
        geom_name
    )

print("==============================")

print("\n==============================")
print("FINGER GEOMS")
print("==============================")

for geom_id in range(model.ngeom):

    body_id = model.geom_bodyid[geom_id]

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
    )

    if body_name in ["left_finger", "right_finger"]:

        print(
            "Geom ID:",
            geom_id,
            "| Body:",
            body_name
        )

print("==============================")


#------SIMULATION DATA----------@
# Rotate Panda base
joint1_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "joint2"
)

joint1_qpos = model.jnt_qposadr[joint1_id]

data.qpos[joint1_qpos] = -1.8  # radians

# Rotate Panda base
joint6_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "joint6"
)

joint6_qpos = model.jnt_qposadr[joint6_id]

data.qpos[joint6_qpos] = 1.2  # radians

# Rotate Panda base
joint4_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "joint4"
)

joint4_qpos = model.jnt_qposadr[joint4_id]

data.qpos[joint4_qpos] = -1.8 # radians

mujoco.mj_forward(model, data)

bottle_4_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "bottle_4"
)

print("\n==============================")
print("BOTTLE 4 POSITION")
print("==============================")
print("Bottle 4 world position:")
print(data.xpos[bottle_4_id])
print("==============================")


#--------MODEL INFORMATION---------!
print("\n==============================")
print("PANDA + BIN SCENE")
print("==============================")

print(f"Bodies    : {model.nbody}")
print(f"Joints    : {model.njnt}")
print(f"Actuators : {model.nu}")
print(f"DOF       : {model.nv}")


#-------BOTTLE DIMENSIONS-------!
mesh_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_MESH,
    "wine_bottle"
)

vertex_count = model.mesh_vertnum[mesh_id]

vertex_start = model.mesh_vertadr[mesh_id]

vertices = model.mesh_vert[
    vertex_start:
    vertex_start + vertex_count
]

mesh_min = vertices.min(axis=0)
mesh_max = vertices.max(axis=0)

bottle_dimensions = mesh_max - mesh_min

print("\n==============================")
print("BOTTLE")
print("==============================")

print(f"Width  : {bottle_dimensions[0]:.3f} m")
print(f"Depth  : {bottle_dimensions[1]:.3f} m")
print(f"Height : {bottle_dimensions[2]:.3f} m")


#------BODY INFORMATION------!
print("\n==============================")
print("BODIES")
print("==============================")

for body_id in range(model.nbody):

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
    )

    print(
        f"{body_id}: {body_name}"
    )


#---------RENDERER----------!
renderer = mujoco.Renderer(
    model=model,
    height=IMAGE_HEIGHT,
    width=IMAGE_WIDTH
)


#------BOTTLE DIMENSIONS-----!
mesh_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_MESH,
    "wine_bottle"
)

vertex_count = model.mesh_vertnum[mesh_id]

vertex_start = model.mesh_vertadr[mesh_id]

vertices = model.mesh_vert[
    vertex_start : vertex_start + vertex_count
]

mesh_min = vertices.min(axis=0)
mesh_max = vertices.max(axis=0)

bottle_dimensions = mesh_max - mesh_min

bottle_width = bottle_dimensions[0]
bottle_depth = bottle_dimensions[1]
bottle_height = bottle_dimensions[2]

print("\nBottle dimensions:")
print(f"Width  : {bottle_width:.3f} m")
print(f"Depth  : {bottle_depth:.3f} m")
print(f"Height : {bottle_height:.3f} m")


#------WRIST CAMERA PARAMETERS-----!
wrist_camera_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "wrist_camera"
)

center_x = IMAGE_WIDTH / 2.0
center_y = IMAGE_HEIGHT / 2.0

wrist_fovy = model.cam_fovy[wrist_camera_id]

focal_length = (
    center_y /
    math.tan(
        math.radians(wrist_fovy / 2.0)
    )
)

print("\nWrist Camera:")
print(f"FOV          : {wrist_fovy:.2f} degrees")
print(f"Focal Length : {focal_length:.2f} pixels")


#-------- WRIST CAMERA WORLD POSE - DEBUG-------@
# Make sure MuJoCo has calculated the current camera pose
mujoco.mj_forward(model, data)

camera_position = (
    data.cam_xpos[wrist_camera_id].copy()
)

camera_rotation = (
    data.cam_xmat[wrist_camera_id]
    .reshape(3, 3)
    .copy()
)

print("\nWrist Camera World Position:")
print(
    f"X={camera_position[0]:.3f}, "
    f"Y={camera_position[1]:.3f}, "
    f"Z={camera_position[2]:.3f}"
)

print("\nWrist Camera World Rotation:")
print(camera_rotation)

print()


#------OVERVIEW CAMERA CONTROLS FOR LIVE--------!
overview_camera_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_CAMERA,
    "overview_camera"
)

# Current camera position
camera_pos = model.cam_pos[overview_camera_id].copy()

# Keep Euler angles ourselves.
# These match the values in the XML.
camera_euler = np.array([
    0.65,      # X rotation
    0.0,       # Y rotation
    -1.0       # Z rotation
])

# Current field of view
camera_fovy = model.cam_fovy[overview_camera_id]


def print_camera_settings():

    print("\n==============================")
    print("OVERVIEW CAMERA")
    print("==============================")

    print(
        f"Position : "
        f"X={camera_pos[0]:.3f}, "
        f"Y={camera_pos[1]:.3f}, "
        f"Z={camera_pos[2]:.3f}"
    )

    camera_quaternion = model.cam_quat[overview_camera_id].copy()

    print(
        f"Quaternion : "
        f"{camera_quaternion[0]:.6f} "
        f"{camera_quaternion[1]:.6f} "
        f"{camera_quaternion[2]:.6f} "
        f"{camera_quaternion[3]:.6f}"
    )

    print(f"FOV      : {camera_fovy:.3f}")

    print("==============================\n")


# ============================================================
# LIVE RENDER LOOP
# ============================================================

print("\n==============================")
print("CAMERA CONTROLS")
print("==============================")
print("W / S : Move Y")
print("A / D : Move X")
print("Q / E : Move Z")
print("I / K : Rotate X")
print("J / L : Rotate Z")
print("+ / - : Change FOV")
print("P     : Print camera values")
print("ESC   : Exit")
print("==============================\n")

# ============================================================
# IK TEST
# ============================================================
ik = PandaIK(model,data)
# ============================================================
# SAVE MANUALLY DEFINED STARTING POSE
# ============================================================

starting_qpos = data.qpos[:7].copy()


# ============================================================
# APPROACH TARGET
# ============================================================

# Position above the bottle

approach_position = np.array([0.30, -0.15, 0.73])


# ============================================================
# GRASP TARGET
# ============================================================

# Bottle center is approximately Z = 0.20 m
grasp_position = np.array([0.33, -0.15, 0.68])


# ============================================================
# SOLVE IK FOR APPROACH
# ============================================================

approach_q = ik.solve(
    approach_position
)


print("\n==============================")
print("APPROACH IK RESULT")
print("==============================")

print("Target:")
print(approach_position)

print("\nIK joint solution:")
print(approach_q)

print("\nLink7 position produced by IK:")
print(
    data.xpos[
        ik.link7_body_id
    ].copy()
)

print("\nIK position error:")
print(
    approach_position
    - data.xpos[ik.link7_body_id]
)

print("\nIK distance:")
print(
    np.linalg.norm(
        approach_position
        - data.xpos[ik.link7_body_id]
    )
)

print("==============================")


# ============================================================
# CHECK APPROACH FK
# ============================================================

approach_hand_position = data.xpos[
    ik.hand_body_id
].copy()

left_finger_position = data.xpos[
        ik.left_finger_id
    ].copy()

right_finger_position = data.xpos[
        ik.right_finger_id
    ].copy()

current_gripper_center = (
        left_finger_position
        + right_finger_position
    ) / 2.0


approach_error = (
    approach_position
    - current_gripper_center
)


print("\nPredicted hand position:")
print(approach_hand_position)

print("\nRemaining error:")
print(approach_error)

print(
    "\nDistance to approach target:",
    np.linalg.norm(approach_error)
)


# ============================================================
# RESTORE REAL STARTING POSITION
# ============================================================

# IK changed qpos while calculating.
# Put Panda back where YOU manually positioned it.

data.qpos[:7] = starting_qpos

mujoco.mj_forward(
    model,
    data
)


# ============================================================
# MOVEMENT STATE
# ============================================================

phase = "open"

target_q = None

step_count = 0

max_steps = 5000
grasp_step_count = 0

target_tolerance = 0.13 #0.01
center_tolerance = 0.001
lift_position = np.array([0.33, -0.15, 0.85])
grasp_q = None

# ========================================================
# GRASP TARGET
# ========================================================

floor_z = 0.05

# Bottle center height
bottle_center_z = (
    floor_z +
    bottle_height / 2.0
)

# Bottle top height
bottle_top_z = (
    floor_z +
    bottle_height
)

# Grasp slightly below the bottle top
grasp_offset = 0.03   # 3 cm

grasp_z = (
    bottle_top_z -
    grasp_offset
)

# Keep the same X/Y position as the bottle
grasp_position = np.array([
    0.30,
    -0.15,
    grasp_z
])

print("\n==============================")
print("GRASP TARGET")
print("==============================")

print("Bottle center Z:", bottle_center_z)
print("Bottle top Z:", bottle_top_z)
print("Grasp Z:", grasp_z)
print("Grasp position:", grasp_position)

print("==============================")

# ============================================================
# GRIPPER
# ============================================================

GRIPPER_OPEN = 255
GRIPPER_CLOSE = 0
gripper_commanded = False

# ============================================================
# WRIST CAMERA + OVERVIEW CAMERA LOOP
# ============================================================

orientation_test_applied = False
orientation_test_steps = 0

while step_count < max_steps:

    # ========================================================
    # PHYSICS
    # ========================================================

    mujoco.mj_step(
        model,
        data
    )

    step_count += 1

    # ========================================================
    # PHASE 0 — OPEN GRIPPER
    # ========================================================

    if phase == "open":

        data.ctrl[7] = GRIPPER_OPEN

        left_finger_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "finger_joint1"
        )

        right_finger_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "finger_joint2"
        )

        left_qpos = data.qpos[
            model.jnt_qposadr[left_finger_joint_id]
        ]

        right_qpos = data.qpos[
            model.jnt_qposadr[right_finger_joint_id]
        ]

        left_open_limit = model.jnt_range[left_finger_joint_id, 1]
        right_open_limit = model.jnt_range[right_finger_joint_id, 1]

        if (
            left_qpos >= left_open_limit - 1e-5
            and
            right_qpos >= right_open_limit - 1e-5
        ):

            print("\n==============================")
            print("GRIPPER FULLY OPEN")
            print("==============================")
            print(f"Left  = {left_qpos:.6f}")
            print(f"Right = {right_qpos:.6f}")
            print(f"Left limit  = {left_open_limit:.6f}")
            print(f"Right limit = {right_open_limit:.6f}")
            print("==============================")

            left_finger_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "left_finger"
            )

            right_finger_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                "right_finger"
            )

            left_finger_pos = data.xpos[left_finger_id].copy()
            right_finger_pos = data.xpos[right_finger_id].copy()

            gripper_center = (left_finger_pos + right_finger_pos) / 2.0

            # ============================================================
            # BOTTLE 3 CENTER
            # ============================================================

            bottle_geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "bottle_3_geom"
            )

            bottle_center = data.geom_xpos[
                bottle_geom_id
            ].copy()


            # ============================================================
            # PRINT BOTH
            # ============================================================

            print("Left finger position : ", left_finger_pos)
            print("Right finger position: ", right_finger_pos)

            print("Gripper center       : ", gripper_center)

            print("Bottle 3 center      : ", bottle_center)

            print("==============================")

            target_q = approach_q
            phase = "approach"
            step_count = 0

        continue

    # CURRENT HAND POSITION
    current_hand_position = data.xpos[ik.hand_body_id].copy()
    current_link7_position = data.xpos[ik.link7_body_id].copy()

    left_finger_position = data.xpos[ik.left_finger_id].copy()
    right_finger_position = data.xpos[ik.right_finger_id].copy()
    current_gripper_center = (left_finger_position + right_finger_position) / 2.0


    # ========================================================
    # PHASE 1 — MOVE ABOVE BOTTLE
    # ========================================================

    if phase == "approach":
        if gripper_commanded == True:
            print("Gripper command true in approach")

        position_error = (
            approach_position
            - current_gripper_center
        )

        distance_to_target = np.linalg.norm(
            position_error
        )


        # ----------------------------------------------------
        # Reached actual approach target
        # ----------------------------------------------------

        if distance_to_target < target_tolerance:
            print("\n==============================")
            print("APPROACH DEBUG")
            print("==============================")

            print("Target:")
            print(approach_position)

            print("Current Link7:")
            print(current_link7_position)

            print("current_hand_position:")
            print(current_hand_position)

            print("gripper center position:")
            print(current_gripper_center)

            print("Position error:")
            print(position_error)

            print("Distance:")
            print(distance_to_target)

            print("Tolerance:")
            print(target_tolerance)

            print("CURRENT finger center:", current_gripper_center)
            print("APPROACH TARGET:", approach_position)
            print(
                "ERROR:",
                approach_position - current_gripper_center
            )

            print("==============================")

            # =================================================
            # APPROACH FINISHED
            # =================================================

            current_qpos = data.qpos[:7].copy()


            # =================================================
            # CURRENT GRIPPER CENTER
            # =================================================

            left_finger_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"left_finger")
            right_finger_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,"right_finger")

            left_finger_position = data.xpos[left_finger_id].copy()
            right_finger_position = data.xpos[right_finger_id].copy()
            gripper_center = (left_finger_position + right_finger_position) / 2.0

            # =================================================
            # BOTTLE 3 CENTER
            # =================================================

            bottle_geom_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,"bottle_3_geom")
            bottle_center = data.geom_xpos[bottle_geom_id].copy()

            # ============================================================
            # PRINT BOTH
            # ============================================================

            print("Left finger position : ", left_finger_pos)
            print("Right finger position: ", right_finger_pos)

            print("Gripper center       : ", gripper_center)

            print("Bottle 3 center      : ", bottle_center)

            print("==============================")


            # =================================================
            # CENTER TARGET
            # Keep current Z
            # Change only X and Y
            # =================================================

            center_position = np.array([
                bottle_center[0],
                bottle_center[1],
                gripper_center[2]
            ])


            # =================================================
            # CENTER IK
            # =================================================
            center_q = ik.solve(center_position)

            # =================================================
            # RESTORE REAL CURRENT PHYSICAL POSITION
            # =================================================

            data.qpos[:7] = current_qpos
            mujoco.mj_forward(model,data)

            # =================================================
            # START CENTER PHASE
            # =================================================

            target_q = center_q

            phase = "center"

            step_count = 0

    # ========================================================
    # PHASE 3 — MOVE Center TO BOTTLE
    # ========================================================
    elif phase == "center":

        # --------------------------------------------------------
        # CURRENT FINGER CENTER
        # --------------------------------------------------------

        left_finger_position = data.xpos[
            ik.left_finger_id
        ].copy()

        right_finger_position = data.xpos[
            ik.right_finger_id
        ].copy()

        current_gripper_center = (
            left_finger_position +
            right_finger_position
        ) / 2.0


        # --------------------------------------------------------
        # REMAINING DISTANCE TO CENTER
        # --------------------------------------------------------

        position_error = (
            center_position -
            current_gripper_center
        )

        distance_to_target = np.linalg.norm(
            position_error
        )


        # --------------------------------------------------------
        # CENTER REACHED
        # --------------------------------------------------------

        if distance_to_target < center_tolerance:

            print("\n==============================")
            print("CENTER POSITION REACHED")
            print("==============================")

            print("Target:")
            print(center_position)

            print("Current gripper center:")
            print(current_gripper_center)

            print("Remaining error:")
            print(position_error)

            print("Distance:")
            print(distance_to_target)

            print("==============================")

            grasp_target = current_gripper_center.copy()
            grasp_target[2] -= 0.09

            phase = "grasp"
            

        # --------------------------------------------------------
        # STILL AWAY → MOVE THE REMAINING DISTANCE
        # --------------------------------------------------------

        else:
            print("\n==============================")
            print("Center not reached, trying again")
            print("==============================")
            print("Target:")
            print(center_position)

            print("Current gripper center:")
            print(current_gripper_center)

            print("Remaining error:")
            print(position_error)

            print("Distance:")
            print(distance_to_target)

            print("==============================")

            correction_target = (
                current_gripper_center +
                position_error
            )

            current_qpos = data.qpos[:7].copy()

            center_q = ik.solve(
                correction_target
            )

            data.qpos[:7] = current_qpos

            mujoco.mj_forward(
                model,
                data
            )

            target_q = center_q

    # ========================================================
    # PHASE 4 — MOVE 2 CM DOWN
    # ========================================================

    elif phase == "grasp":

        # ----------------------------------------------------
        # Current gripper center
        # ----------------------------------------------------

        left_finger_position = (
            data.xpos[ik.left_finger_id].copy()
        )

        right_finger_position = (
            data.xpos[ik.right_finger_id].copy()
        )

        current_gripper_center = (
            left_finger_position +
            right_finger_position
        ) / 2.0

        # ----------------------------------------------------
        # Distance to the fixed 2 cm lower target
        # ----------------------------------------------------

        position_error = (
            grasp_target -
            current_gripper_center
        )

        distance_to_target = np.linalg.norm(
            position_error
        )

        # ----------------------------------------------------
        # 2 CM DOWN REACHED
        # ----------------------------------------------------

        print("distance_to_target: ",distance_to_target)
        if distance_to_target < center_tolerance:

            print("\n==============================")
            print("2 CM DOWN REACHED")
            print("==============================")

            print("\nGrasp target:")
            print(grasp_target)

            print("\nCurrent gripper center:")
            print(current_gripper_center)

            print("\nRemaining error:")
            print(position_error)

            print("\nDistance:")
            print(distance_to_target)

            print("==============================")
            print("READY FOR GRIPPER.")
            # Close the fingers
            phase = "close"

            print("CLOSING GRIPPER")

            

        # ----------------------------------------------------
        # MOVE TOWARD FIXED 2 CM LOWER TARGET
        # ----------------------------------------------------

        else:
            # print("Going down!!!!!")

            current_qpos = data.qpos[:7].copy()

            grasp_q = ik.solve(
                grasp_target
            )

            # IK temporarily changes qpos.
            # Restore the real physical position.
            data.qpos[:7] = current_qpos

            mujoco.mj_forward(
                model,
                data
            )

            target_q = grasp_q

    elif phase == "close":

        orientation_test_steps += 1

        # --------------------------------
        # LET THE ARM REACH THE IK POSE
        # --------------------------------
        # if orientation_test_steps < 500:
        #     continue

        # --------------------------------
        # CLOSE GRIPPER
        # --------------------------------
        data.ctrl[7] = GRIPPER_CLOSE

        print("\n==============================")
        print("GRIPPER CLOSING")
        print("==============================")

        # Give the fingers time to physically close
        if orientation_test_steps < 700:
            continue

        # --------------------------------
        # CHECK ACTUAL HAND TILT
        # --------------------------------

        hand_z = data.xmat[ik.hand_body_id].reshape(3, 3)[:, 2]

        desired_z = np.array([0.0, 0.0, -1.0])

        cos_angle = np.clip(
            np.dot(hand_z, desired_z),
            -1.0,
            1.0
        )

        tilt_angle = np.degrees(np.arccos(cos_angle))

        print("\n==============================")
        print("ACTUAL HAND ORIENTATION")
        print("==============================")
        print("Hand Z:", hand_z)
        print(f"Tilt from vertical: {tilt_angle:.3f} degrees")
        print("==============================")

        # --------------------------------
        # CHECK BOTTLE CONTACT
        # --------------------------------
        left_touch = False
        right_touch = False

        bottle_geom = 88

        for i in range(data.ncon):

            contact = data.contact[i]

            geom1 = contact.geom1
            geom2 = contact.geom2

            # Left finger geoms: 70-77
            if (
                (70 <= geom1 <= 77 and geom2 == bottle_geom)
                or
                (70 <= geom2 <= 77 and geom1 == bottle_geom)
            ):
                left_touch = True

            # Right finger geoms: 78-85
            if (
                (78 <= geom1 <= 85 and geom2 == bottle_geom)
                or
                (78 <= geom2 <= 85 and geom1 == bottle_geom)
            ):
                right_touch = True

        print("\n==============================")
        print("GRIPPER CONTACT RESULT")
        print("==============================")
        print("Left finger contact :", left_touch)
        print("Right finger contact:", right_touch)
        print("==============================")


        # ==========================================
        # CALCULATE FINGER → BOTTLE DISTANCE
        # ==========================================

        def get_finger_bottle_distance(finger_geoms, bottle_geom):

            min_distance = float("inf")

            for finger_geom in finger_geoms:

                distance = mujoco.mj_geomDistance(
                    model,
                    data,
                    finger_geom,
                    bottle_geom,
                    1.0,
                    None
                )

                min_distance = min(min_distance, distance)

            return min_distance


        left_distance = get_finger_bottle_distance(
            range(70, 78),
            bottle_geom
        )

        right_distance = get_finger_bottle_distance(
            range(78, 86),
            bottle_geom
        )


        print("\n==============================")
        print("FINGER → BOTTLE DISTANCE")
        print("==============================")
        print(f"Left finger → bottle : {left_distance:.6f} m")
        print(f"Right finger → bottle: {right_distance:.6f} m")
        print("==============================")

        # ==============================
        # START LIFT TEST
        # ==============================

        left_finger_pos = data.xpos[ik.left_finger_id].copy()
        right_finger_pos = data.xpos[ik.right_finger_id].copy()

        current_gripper_center = (
            left_finger_pos + right_finger_pos
        ) / 2.0

        # target_q = lift_q
        orientation_test_steps = 0

        print("\n==============================")
        print("STARTING LIFT TEST")
        print("==============================")
        print("Current:", current_gripper_center)
        # print("Target :", lift_target)
        print("Lift   : +5 cm Z")
        print("==============================")

        phase = "lift"

    elif phase == "lift":

        if orientation_test_steps == 0:
            # compute lift target WITHOUT disturbing the real state
            saved_qpos = data.qpos[:7].copy()
            start_q = saved_qpos.copy()

            lift_target = current_gripper_center.copy()
            lift_target[2] += 0.10
            lift_q = ik.solve(lift_target)

            data.qpos[:7] = saved_qpos          # <-- restore
            mujoco.mj_forward(model, data)

        # smooth ramp over 400 steps
        alpha = min(orientation_test_steps / 400.0, 1.0)
        target_q = (1 - alpha) * start_q + alpha * lift_q

        orientation_test_steps += 1


    # ========================================================
    # SEND CURRENT TARGET TO ARM
    # ========================================================

    mujoco.mj_forward(
        model,
        data
    )

    kp = model.actuator_gainprm[:7, 0]

    bias_compensation = (
        data.qfrc_bias[:7] / kp
    )

    corrected_ctrl = (
        target_q
        + bias_compensation
    )

    data.ctrl[:7] = corrected_ctrl


    # ========================================================
    # JOINT MOVEMENT DEBUG
    # ========================================================
    # print("Step: ",step_count)
    if step_count % 100 == 0 and phase == 'center':

        current_q = data.qpos[:7].copy()

        joint_error = (
            target_q
            - current_q
        )

        print("\n==============================")
        print("JOINT MOVEMENT DEBUG")
        print("==============================")

        print("TARGET Q:")
        print(target_q)

        print("\nCURRENT Q:")
        print(current_q)

        print("\nJOINT ERROR:")
        print(joint_error)

        print("\nQFRc BIAS:")
        print(data.qfrc_bias[:7])

        print("\nKP:")
        print(kp)

        print("\nBIAS COMPENSATION:")
        print(bias_compensation)

        print("\nCORRECTED CTRL:")
        print(corrected_ctrl)

        print("\nACTUATOR FORCE:")
        print(data.actuator_force[:7])

        print("\nQCRc ACTUATOR:")
        print(data.qfrc_actuator[:7])

        print("\nACTUATOR FORCE LIMIT:")
        print(model.actuator_forcerange[:7])

        joint_error = target_q - data.qpos[:7]

        print("TARGET Q:", target_q)
        print("CURRENT Q:", data.qpos[:7])
        print("JOINT ERROR:", joint_error)
        print("==============================")


    # ========================================================
    # WRIST CAMERA
    # ========================================================

    mujoco.mj_forward(
        model,
        data
    )

    renderer.update_scene(
        data,
        camera="wrist_camera"
    )

    wrist_frame = renderer.render()

    wrist_frame = np.asarray(
        wrist_frame
    )

    wrist_frame = cv2.cvtColor(
        wrist_frame,
        cv2.COLOR_RGB2BGR
    )


    # ========================================================
    # YOLO SEGMENTATION
    # ========================================================

    boxes, labels, scores, masks, track_ids, class_names = (
        segmentor.segment(
            frame=wrist_frame
        )
    )


    wrist_overlay = wrist_frame.copy()


    # ========================================================
    # PROCESS DETECTIONS
    # ========================================================

    if masks is not None and track_ids is not None:
        polygons = masks.xy
        for (polygon,box,label,track_id,score) in zip(polygons,boxes,labels,track_ids,scores):

            if score < 0.1:
                continue

            polygon = polygon.astype(np.int32)

            track_id = int(
                track_id
            )

            color = colors[
                track_id % len(colors)
            ]


            # ------------------------------------------------
            # Segmentation
            # ------------------------------------------------

            cv2.fillPoly(
                wrist_overlay,
                [polygon],
                color
            )


            # ------------------------------------------------
            # Bounding box
            # ------------------------------------------------

            x1, y1, x2, y2 = (
                box.int().tolist()
            )

            class_name = class_names[
                int(label)
            ]


            # ------------------------------------------------
            # Detection center
            # ------------------------------------------------

            u = (
                x1 + x2
            ) / 2.0

            v = (
                y1 + y2
            ) / 2.0


            # =================================================
            # CAMERA → WORLD
            # =================================================

            ray_camera = np.array([
                (u - center_x)
                / focal_length,

                -(v - center_y)
                / focal_length,

                -1.0
            ])


            camera_position = (
                data.cam_xpos[
                    wrist_camera_id
                ].copy()
            )


            camera_rotation = (
                data.cam_xmat[
                    wrist_camera_id
                ]
                .reshape(3, 3)
            )


            ray_world = (
                camera_rotation
                @ ray_camera
            )


            # ------------------------------------------------
            # Bottle center height
            # ------------------------------------------------

            floor_z = 0.05

            bottle_center_z = (
                floor_z
                + bottle_height / 2.0
            )


            # ------------------------------------------------
            # Ray intersection
            # ------------------------------------------------

            if abs(ray_world[2]) < 1e-6:
                continue


            scale = (
                bottle_center_z
                - camera_position[2]
            ) / ray_world[2]


            world_position = (
                camera_position
                + scale * ray_world
            )


            world_x = world_position[0]
            world_y = world_position[1]
            world_z = world_position[2]


            # ------------------------------------------------
            # Print world position
            # ------------------------------------------------

            # print(
            #     f"Track {track_id} | "
            #     f"{class_name} | "
            #     f"World XYZ: "
            #     f"X={world_x:.3f}, "
            #     f"Y={world_y:.3f}, "
            #     f"Z={world_z:.3f}"
            # )


            # ------------------------------------------------
            # Polygon outline
            # ------------------------------------------------

            cv2.polylines(
                wrist_overlay,
                [polygon],
                isClosed=True,
                color=color,
                thickness=1,
                lineType=cv2.LINE_AA
            )


            # ------------------------------------------------
            # ID + XYZ
            # ------------------------------------------------

            cv2.putText(
                wrist_overlay,

                f"ID:{track_id} "
                f"XYZ:"
                f"({world_x:.2f},"
                f"{world_y:.2f},"
                f"{world_z:.2f})",

                (
                    x1,
                    max(y1 - 10, 20)
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.5,

                color,

                2
            )


    # ========================================================
    # BLEND SEGMENTATION
    # ========================================================

    alpha = 0.27

    wrist_frame = cv2.addWeighted(
        wrist_frame,
        1 - alpha,
        wrist_overlay,
        alpha,
        0
    )


    # ========================================================
    # OVERVIEW CAMERA
    # ========================================================

    renderer.update_scene(
        data,
        camera="overview_camera"
    )

    overview_frame = renderer.render()

    overview_frame = np.asarray(
        overview_frame
    )

    overview_frame = cv2.cvtColor(
        overview_frame,
        cv2.COLOR_RGB2BGR
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "Panda Movement - Overview",
        overview_frame
    )


    # Keep wrist camera available too

    # cv2.imshow(
    #     "Wrist Camera - Segmentation",
    #     wrist_frame
    # )


    # ========================================================
    # STATUS
    # ========================================================

    if step_count % 100 == 0:

        print(
            f"Step: {step_count} | "
            f"Phase: {phase} | "
            f"Distance: "
            f"{distance_to_target:.4f} m"
        )


    # ========================================================
    # ESC / Q
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == 27 or key == ord("q"):
        break

# ============================================================
# FINAL
# ============================================================

print("\n==============================")
print("MOVEMENT FINISHED")
print("==============================")

print(
    "Final hand position:",
    data.xpos[
        ik.hand_body_id
    ]
)

print(
    "Target position:",
    (
        approach_position
        if phase == "approach"
        else grasp_position
    )
)

cv2.destroyAllWindows()