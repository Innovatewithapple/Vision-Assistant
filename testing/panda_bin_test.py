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


#-------OFFSCREEN FRAMEBUFFER--------!
model.vis.global_.offwidth = IMAGE_WIDTH
model.vis.global_.offheight = IMAGE_HEIGHT
data = mujoco.MjData(model)


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

approach_position = np.array([0.30, -0.15, 0.70])


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


approach_error = (
    approach_position
    - approach_hand_position
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
lift_position = np.array([0.33, -0.15, 0.85])
grasp_q = None


# ============================================================
# GRIPPER
# ============================================================

GRIPPER_OPEN = 255
GRIPPER_CLOSE = 0
gripper_commanded = False

# ============================================================
# WRIST CAMERA + OVERVIEW CAMERA LOOP
# ============================================================

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
        if gripper_commanded == True:
            print("Gripper command true in open")

        data.ctrl[7] = GRIPPER_OPEN

        if step_count >= 100:

            print("\n==============================")
            print("GRIPPER OPEN")
            print("==============================")

            target_q = approach_q

            phase = "approach"

            step_count = 0

        continue


    # ========================================================
    # CURRENT HAND POSITION
    # ========================================================

    current_hand_position = data.xpos[
        ik.hand_body_id
    ].copy()


    current_link7_position = data.xpos[
        ik.link7_body_id
    ].copy()


    # ========================================================
    # PHASE 1 — MOVE ABOVE BOTTLE
    # ========================================================

    if phase == "approach":
        if gripper_commanded == True:
            print("Gripper command true in approach")

        position_error = (
            approach_position
            - current_hand_position
        )

        distance_to_target = np.linalg.norm(
            position_error
        )


        # ----------------------------------------------------
        # Reached approach position?
        # ----------------------------------------------------

        if step_count == 100:

            print("\n==============================")
            print("APPROACH DEBUG")
            print("==============================")

            print("Target:")
            print(approach_position)

            print("Current Link7:")
            print(current_link7_position)

            print("current_hand_position:")
            print(current_hand_position)

            print("Position error:")
            print(position_error)

            print("Distance:")
            print(distance_to_target)

            print("Tolerance:")
            print(target_tolerance)

            print("CURRENT HAND:", current_hand_position)
            print("APPROACH TARGET:", approach_position)
            print(
                "HAND ERROR:",
                approach_position - current_hand_position
            )

            print("==============================")


        # ----------------------------------------------------
        # Reached actual approach target
        # ----------------------------------------------------

        if distance_to_target < target_tolerance:

            print("\n==============================")
            print("APPROACH POSITION REACHED")
            print("==============================")

            print(
                "Hand position:",
                current_hand_position
            )

            print(
                "Approach target:",
                approach_position
            )

            print(
                "Distance:",
                distance_to_target
            )

            current_link7_position = data.xpos[
                ik.link7_body_id
            ].copy()

            print(
                "Link7 position:",
                current_link7_position
            )

            print(
                "Hand position:",
                current_hand_position
            )


            # =================================================
            # SOLVE SECOND IK
            # =================================================

            current_qpos = data.qpos[
                :7
            ].copy()


            # =================================================
            # HAND OFFSET FROM LINK7
            # =================================================

            hand_offset_local = np.array([
                0.0,
                0.0,
                0.107
            ])


            link7_rotation = data.xmat[
                ik.link7_body_id
            ].reshape(3, 3).copy()


            hand_offset_world = (
                link7_rotation
                @ hand_offset_local
            )


            grasp_link7_target = (
                grasp_position
                - hand_offset_world
            )


            grasp_q = ik.solve(
                grasp_link7_target
            )
            lift_q = ik.solve(
                lift_position
            )


            print("\n==============================")
            print("GRASP IK")
            print("==============================")

            print("Hand target:")
            print(grasp_position)

            print("\nLink7 IK target:")
            print(grasp_link7_target)

            print("\nJoint targets:")
            print(grasp_q)


            # ------------------------------------------------
            # Restore REAL current physical position
            # ------------------------------------------------

            data.qpos[:7] = current_qpos

            mujoco.mj_forward(
                model,
                data
            )


            target_q = grasp_q

            phase = "grasp"

            print(
                "\nSwitching to GRASP phase..."
            )


    # ========================================================
    # PHASE 2 — MOVE DOWN TO BOTTLE
    # ========================================================

    elif phase == "grasp":
        grasp_step_count += 1

        position_error = (
            grasp_position
            - current_hand_position
        )

        distance_to_target = np.linalg.norm(
            position_error
        )

        # print(f"\nInside Grasp==")
        # print("distance_to_target: ",distance_to_target)
        # print("target_tolerance: ",target_tolerance)
        if distance_to_target <= target_tolerance:

            print("\n==============================")
            print("GRASP POSITION REACHED")
            print("==============================")

            print("\nDesired hand position:")
            print(grasp_position)

            print("\nActual hand position:")
            print(
                data.xpos[
                    ik.hand_body_id
                ].copy()
            )

            print("\nActual link7 position:")
            print(
                data.xpos[
                    ik.link7_body_id
                ].copy()
            )

            print("\nHand - Link7 offset:")
            print(
                data.xpos[ik.hand_body_id]
                - data.xpos[ik.link7_body_id]
            )

            print("\nLink7 rotation:")
            print(
                data.xmat[
                    ik.link7_body_id
                ].reshape(3, 3)
            )
            # Close gripper
            print(f"grasp_step_count: {grasp_step_count}")
            if grasp_step_count >= 300:
                data.ctrl[7] = GRIPPER_CLOSE
                print("GRIPPER CLOSING...")
            if grasp_step_count >= 700:
                print("lifting")
                target_q = lift_q


                


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

    if step_count % 100 == 0 and phase == "approach":

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


        for (
            polygon,
            box,
            label,
            track_id,
            score
        ) in zip(
            polygons,
            boxes,
            labels,
            track_ids,
            scores
        ):

            if score < 0.1:
                continue


            polygon = polygon.astype(
                np.int32
            )

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

            print(
                f"Track {track_id} | "
                f"{class_name} | "
                f"World XYZ: "
                f"X={world_x:.3f}, "
                f"Y={world_y:.3f}, "
                f"Z={world_z:.3f}"
            )


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







































































# while True:

#     # --------------------------------------------------------
#     # Read keyboard input
#     # --------------------------------------------------------

#     key = cv2.waitKey(1) & 0xFF

#     # --------------------------------------------------------
#     # Camera movement
#     # --------------------------------------------------------

#     position_step = 0.02
#     rotation_step = 0.02
#     fov_step = 1.0

#     if key == ord("w"):
#         camera_pos[1] += position_step

#     elif key == ord("s"):
#         camera_pos[1] -= position_step

#     elif key == ord("a"):
#         camera_pos[0] -= position_step

#     elif key == ord("d"):
#         camera_pos[0] += position_step

#     elif key == ord("q"):
#         camera_pos[2] += position_step

#     elif key == ord("e"):
#         camera_pos[2] -= position_step

#     # --------------------------------------------------------
#     # Camera rotation
#     # --------------------------------------------------------

#     elif key == ord("j"):
#         camera_euler[2] -= rotation_step

#     elif key == ord("l"):
#         camera_euler[2] += rotation_step

#     elif key == ord("i"):
#         camera_euler[0] -= rotation_step

#     elif key == ord("k"):
#         camera_euler[0] += rotation_step

#     # --------------------------------------------------------
#     # Field of view
#     # --------------------------------------------------------

#     elif key == ord("+") or key == ord("="):
#         camera_fovy -= fov_step

#     elif key == ord("-"):
#         camera_fovy += fov_step

#     # --------------------------------------------------------
#     # Print current settings
#     # --------------------------------------------------------

#     elif key == ord("p"):
#         print_camera_settings()

#     # --------------------------------------------------------
#     # Exit
#     # --------------------------------------------------------

#     elif key == 27:
#         break

#     # --------------------------------------------------------
#     # Apply camera position
#     # --------------------------------------------------------

#     model.cam_pos[overview_camera_id] = camera_pos

#     # --------------------------------------------------------
#     # Convert Euler → quaternion
#     # --------------------------------------------------------

#     quaternion = np.zeros(4)

#     mujoco.mju_euler2Quat(
#         quaternion,
#         camera_euler,
#         "xyz"
#     )

#     model.cam_quat[overview_camera_id] = quaternion

#     # --------------------------------------------------------
#     # Apply FOV
#     # --------------------------------------------------------

#     model.cam_fovy[overview_camera_id] = camera_fovy

#     # --------------------------------------------------------
#     # IMPORTANT:
#     # Recalculate MuJoCo's derived state
#     # --------------------------------------------------------

#     mujoco.mj_forward(model, data)

#     # --------------------------------------------------------
#     # Render
#     # --------------------------------------------------------

#     renderer.update_scene(
#         data,
#         camera="overview_camera"
#     )

#     image = renderer.render()

#     image = np.asarray(image)

#     image = cv2.cvtColor(
#         image,
#         cv2.COLOR_RGB2BGR
#     )

#     cv2.imshow(
#         "Panda Bin Picking",
#         image
#     )


# cv2.destroyAllWindows()