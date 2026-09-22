from pathlib import Path

import mujoco
import numpy as np
import cv2


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
BOTTLE_DIR = PROJECT_DIR / "bottle"

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
# CAMERA CALIBRATION
# ============================================================

CAMERA_HEIGHT = 2.5
VERTICAL_FOV = 45.0

# Camera position:
# (0, -8, 2.5)
#
# Camera target:
# (0, 0, 0.8)
#
# Therefore the camera looks approximately 12 degrees downward.

CAMERA_Y = -8.0
CAMERA_Z = CAMERA_HEIGHT

CAMERA_TARGET_Z = 0.8


# ============================================================
# MUJOCO SCENE
# ============================================================

xml = f"""
<mujoco model="bin_picking_scene">

    <!-- ================================================== -->
    <!-- COMPILER -->
    <!-- ================================================== -->

    <compiler
        meshdir="{BOTTLE_DIR}"
        texturedir="{BOTTLE_DIR}"
    />


    <!-- ================================================== -->
    <!-- ASSETS -->
    <!-- ================================================== -->

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


    <!-- ================================================== -->
    <!-- WORLD -->
    <!-- ================================================== -->

    <worldbody>


        <!-- ================================================== -->
        <!-- LIGHTING -->
        <!-- ================================================== -->

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

        <light
            name="side_light"
            pos="4 -2 3"
            dir="-1 0 -0.5"
            diffuse="0.7 0.7 0.7"
            specular="0.4 0.4 0.4"
        />


        <!-- ================================================== -->
        <!-- BIN FLOOR -->
        <!-- ================================================== -->

        <!--
            Bin floor:

            Width  = 1.2 m
            Depth  = 1.2 m
            Thickness = 0.10 m
        -->

        <geom
            name="bin_floor"
            type="box"
            pos="0 0 0"
            size="0.6 0.6 0.05"
            rgba="0.35 0.35 0.35 1"
            contype="0"
            conaffinity="0"
        />


        <!-- ================================================== -->
        <!-- BIN WALLS -->
        <!-- ================================================== -->

        <!--
            Wall height = 0.35 m

            MuJoCo "size" is half-size.

            Therefore:
                Z size = 0.175
                Full height = 0.35 m
        -->


        <!-- Back wall -->

        <geom
            name="bin_back_wall"
            type="box"
            pos="0 0.57 0.175"
            size="0.6 0.03 0.175"
            rgba="0.25 0.25 0.25 1"
            contype="0"
            conaffinity="0"
        />


        <!-- Front wall -->

        <geom
            name="bin_front_wall"
            type="box"
            pos="0 -0.57 0.175"
            size="0.6 0.03 0.175"
            rgba="0.25 0.25 0.25 1"
            contype="0"
            conaffinity="0"
        />


        <!-- Left wall -->

        <geom
            name="bin_left_wall"
            type="box"
            pos="-0.57 0 0.175"
            size="0.03 0.6 0.175"
            rgba="0.25 0.25 0.25 1"
            contype="0"
            conaffinity="0"
        />


        <!-- Right wall -->

        <geom
            name="bin_right_wall"
            type="box"
            pos="0.57 0 0.175"
            size="0.03 0.6 0.175"
            rgba="0.25 0.25 0.25 1"
            contype="0"
            conaffinity="0"
        />


        <!-- ================================================== -->
        <!-- BOTTLE 1 -->
        <!-- ================================================== -->

        <!-- Upright -->

        <body
            name="bottle_1"
            pos="-0.25 0.10 0.25"
        >

            <geom
                name="bottle_1_geom"
                type="mesh"
                mesh="wine_bottle"
                material="wine_bottle_material"
                contype="0"
                conaffinity="0"
            />

        </body>


        <!-- ================================================== -->
        <!-- BOTTLE 2 -->
        <!-- ================================================== -->

        <!-- Tilted around Y and Z -->

        <body
            name="bottle_2"
            pos="0.10 0.10 0.25"
            euler="0 25 25"
        >

            <geom
                name="bottle_2_geom"
                type="mesh"
                mesh="wine_bottle"
                material="wine_bottle_material"
                contype="0"
                conaffinity="0"
            />

        </body>


        <!-- ================================================== -->
        <!-- BOTTLE 3 -->
        <!-- ================================================== -->

        <!-- Tilted in another direction -->

        <body
            name="bottle_3"
            pos="0.25 -0.25 0.25"
            euler="20 0 -35"
        >

            <geom
                name="bottle_3_geom"
                type="mesh"
                mesh="wine_bottle"
                material="wine_bottle_material"
                contype="0"
                conaffinity="0"
            />

        </body>


        <!-- ================================================== -->
        <!-- CAMERA TARGET -->
        <!-- ================================================== -->

        <body
            name="camera_target"
            pos="0 0 0.8"
        />


        <!-- ================================================== -->
        <!-- CAMERA -->
        <!-- ================================================== -->

        <camera
            name="main_camera"
            mode="targetbody"
            target="camera_target"
            pos="0 -8 2.5"
            fovy="45"
        />

    </worldbody>


    <!-- ================================================== -->
    <!-- RENDER SETTINGS -->
    <!-- ================================================== -->

    <visual>

        <global
            offwidth="{IMAGE_WIDTH}"
            offheight="{IMAGE_HEIGHT}"
        />

    </visual>

</mujoco>
"""