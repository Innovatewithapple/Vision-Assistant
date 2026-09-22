from pathlib import Path

import mujoco
import numpy as np
import cv2


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
BOTTLE_DIR = PROJECT_DIR / "bottle"


# Automatically find the OBJ file
obj_files = list(BOTTLE_DIR.glob("*.obj"))

if not obj_files:
    raise FileNotFoundError(
        f"No OBJ file found inside {BOTTLE_DIR}"
    )

BOTTLE_OBJ = obj_files[0]

print("Using bottle:")
print(BOTTLE_OBJ)


# ============================================================
# IMAGE SETTINGS
# ============================================================

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


# ============================================================
# MUJOCO XML
# ============================================================

xml = f"""
<mujoco model="wine_bottle_scene">

    <compiler
        meshdir="{BOTTLE_DIR}"
        texturedir="{BOTTLE_DIR}"
    />

    <asset>

        <!-- Bottle texture -->
        <texture
            name="wine_bottle_texture"
            type="2d"
            file="14042_750_mL_Wine_Bottle_dfinal.png"
        />

        <!-- Bottle material -->
        <material
            name="wine_bottle_material"
            texture="wine_bottle_texture"
        />

        <!-- Bottle mesh -->
        <mesh
            name="wine_bottle"
            file="{BOTTLE_OBJ.name}"
            scale="0.1 0.1 0.1"
        />

    </asset>


    <worldbody>

        <!-- ================================================= -->
        <!-- FLOOR -->
        <!-- ================================================= -->

        <geom
            name="floor"
            type="plane"
            size="5 5 0.01"
            rgba="0.85 0.85 0.85 1"
        />


        <!-- ================================================= -->
        <!-- LIGHT -->
        <!-- ================================================= -->

        <light
            name="main_light"
            pos="0 -3 5"
            dir="0 0 -1"
            diffuse="1 1 1"
            specular="1 1 1"
        />

        <light
            name="front_light"
            pos="0 -4 2"
            dir="0 1 -0.3"
            diffuse="1 1 1"
            specular="0.5 0.5 0.5"
        />


        <!-- ================================================= -->
        <!-- BOTTLE -->
        <!-- ================================================= -->

        <body
            name="bottle"
            pos="0 0 0"
        >

            <geom
                name="bottle_geom"
                type="mesh"
                mesh="wine_bottle"
                material="wine_bottle_material"
                contype="0"
                conaffinity="0"
            />

        </body>


        <!-- ================================================= -->
        <!-- CAMERA TARGET -->
        <!-- ================================================= -->

        <body
            name="camera_target"
            pos="0 0 0.8"
        />


        <!-- ================================================= -->
        <!-- CAMERA -->
        <!-- ================================================= -->

        <camera
            name="main_camera"
            mode="targetbody"
            target="camera_target"
            pos="0 -8 2.5"
            fovy="45"
        />

    </worldbody>


    <!-- ===================================================== -->
    <!-- RENDER SETTINGS -->
    <!-- ===================================================== -->

    <visual>

        <global
            offwidth="{IMAGE_WIDTH}"
            offheight="{IMAGE_HEIGHT}"
        />

    </visual>

</mujoco>
"""