import mujoco
import numpy as np
import cv2
from PIL import Image
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "human"
TEXTURE_DIR = MODEL_DIR / "tex"

OBJ_FILE = MODEL_DIR / "rp_dennis_posed_004_30k.OBJ"
DIFFUSE_JPG = TEXTURE_DIR / "rp_dennis_posed_004_dif.jpg"
DIFFUSE_PNG = TEXTURE_DIR / "rp_dennis_posed_004_dif.png"


# --------------------------------------------------
# Convert diffuse texture to PNG
# --------------------------------------------------

if not DIFFUSE_PNG.exists():

    print("Converting diffuse texture to PNG...")

    image = Image.open(DIFFUSE_JPG)
    image = image.convert("RGB")
    image.save(DIFFUSE_PNG)

    print("Created:", DIFFUSE_PNG)


# --------------------------------------------------
# MuJoCo model
# --------------------------------------------------

xml = f"""
<mujoco>

    <compiler
        meshdir="{MODEL_DIR}"
        texturedir="{TEXTURE_DIR}"
    />

    <visual>
        <global
            offwidth="1280"
            offheight="720"
        />
    </visual>


    <asset>

        <!-- Human OBJ mesh -->
        <mesh
            name="human"
            file="{OBJ_FILE.name}"
            scale="0.01 0.01 0.01"
        />

        <!-- Human diffuse / clothing texture -->
        <texture
            name="human_texture"
            type="2d"
            file="{DIFFUSE_PNG.name}"
        />

        <material
            name="human_material"
            texture="human_texture"
        />

    </asset>


    <worldbody>

        <!-- Lighting -->

        <light
            pos="0 -4 6"
            directional="false"
        />

        <light
            pos="3 -2 4"
            directional="false"
        />


        <!-- Ground -->

        <geom
            type="plane"
            size="20 20 0.1"
            rgba="0.65 0.65 0.65 1"
        />


        <!--
            HUMAN

            Camera:
                (0, -8, 1.45)

            Human:
                (0, -3, 0.011)

            Ground distance:
                5 meters

            The quaternion below:
                0.5 0.5 0.5 0.5

            keeps the human upright while rotating
            the human's original facing direction
            toward the camera.
        -->

        <body
            name="human"
            pos="0 -3 0.011"
            euler="0 0 -90"
        >

            <geom
                type="mesh"
                mesh="human"
                material="human_material"
                euler="90 0 0"
                contype="0"
                conaffinity="0"
            />

        </body>


        <!-- First-person camera -->

        <camera
            name="main_camera"
            pos="0 -8 1.45"
            euler="90 0 0"
            fovy="60"
        />

    </worldbody>

</mujoco>
"""


# --------------------------------------------------
# Create MuJoCo model
# --------------------------------------------------

print("Loading MuJoCo model...")

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

print("Model loaded successfully.")


# --------------------------------------------------
# Renderer
# --------------------------------------------------

renderer = mujoco.Renderer(
    model,
    height=720,
    width=1280
)


# --------------------------------------------------
# Forward simulation
# --------------------------------------------------

mujoco.mj_forward(model, data)


# --------------------------------------------------
# Render from first-person camera
# --------------------------------------------------

renderer.update_scene(
    data,
    camera="main_camera"
)

image = renderer.render()

image = np.asarray(image)

# MuJoCo gives RGB
# OpenCV expects BGR

image = cv2.cvtColor(
    image,
    cv2.COLOR_RGB2BGR
)


# --------------------------------------------------
# Display
# --------------------------------------------------

cv2.imshow(
    "MuJoCo Human",
    image
)

cv2.waitKey(0)
cv2.destroyAllWindows()