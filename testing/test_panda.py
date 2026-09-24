from pathlib import Path
import cv2
import mujoco
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PANDA_SCENE = (
    PROJECT_DIR
    / "mujoco_menagerie"
    / "franka_emika_panda"
    / "scene.xml"
)


# ============================================================
# LOAD PANDA
# ============================================================

model = mujoco.MjModel.from_xml_path(
    str(PANDA_SCENE)
)
model.vis.global_.offwidth = 1280
model.vis.global_.offheight = 720
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)


# ============================================================
# BASIC MODEL INFORMATION
# ============================================================

print("\n==============================")
print("PANDA MODEL")
print("==============================")

print(f"Number of bodies   : {model.nbody}")
print(f"Number of joints   : {model.njnt}")
print(f"Number of actuators : {model.nu}")
print(f"Number of DoF      : {model.nv}")


# ============================================================
# JOINT INFORMATION
# ============================================================

print("\n==============================")
print("JOINTS")
print("==============================")

for joint_id in range(model.njnt):

    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_id
    )

    print(
        f"{joint_id}: {joint_name}"
    )


# ============================================================
# ACTUATOR INFORMATION
# ============================================================

print("\n==============================")
print("ACTUATORS")
print("==============================")

for actuator_id in range(model.nu):

    actuator_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        actuator_id
    )

    print(
        f"{actuator_id}: {actuator_name}"
    )


# ============================================================
# SET INITIAL ROBOT POSITION
# ============================================================

home_keyframe = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_KEY,
    "home"
)

if home_keyframe != -1:

    mujoco.mj_resetDataKeyframe(
        model,
        data,
        home_keyframe
    )

    mujoco.mj_forward(
        model,
        data
    )


# ============================================================
# RENDERER
# ============================================================

renderer = mujoco.Renderer(
    model=model,
    height=720,
    width=1280
)

# ============================================================
# RENDER ONE IMAGE
# ============================================================

renderer.update_scene(data)

image = renderer.render() 

image = np.asarray(image)

print("\n==============================")
print("RENDER")
print("==============================")

print(f"Image shape: {image.shape}")

print("\nPanda loaded successfully!")

cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
cv2.imshow("Panda",image)

cv2.waitKey(0)
cv2.destroyAllWindows()