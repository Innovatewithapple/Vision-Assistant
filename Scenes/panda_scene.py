from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

PANDA_DIR = (
    PROJECT_DIR
    / "mujoco_menagerie"
    / "franka_emika_panda"
)

PANDA_XML = PANDA_DIR / "panda.xml"

PANDA_ASSETS = PANDA_DIR / "assets"

BOTTLE_DIR = PROJECT_DIR / "bottle"

BOTTLE_OBJ = (
    BOTTLE_DIR
    / "14042_750_mL_Wine_Bottle_r_v1_L3.obj"
)

BOTTLE_TEXTURE = (
    BOTTLE_DIR
    / "14042_750_mL_Wine_Bottle_dfinal.png"
)


# ============================================================
# LOAD PANDA XML
# ============================================================

panda_xml = PANDA_XML.read_text()


# ============================================================
# MAKE PANDA ASSET PATH ABSOLUTE
# ============================================================

panda_xml = panda_xml.replace(
    'meshdir="assets"',
    f'meshdir="{PANDA_ASSETS}"'
)

import re

m = re.search(r'<[^<>]*name="actuator8"[^<>]*>', panda_xml, re.S)
if m:
    tag = m.group(0)
    new_tag = re.sub(r'biasprm="[^"]*"', 'biasprm="0 -1000 -10"', tag)
    new_tag = re.sub(r'gainprm="[^"]*"', 'gainprm="0.1568627451 0 0"', new_tag)
    panda_xml = panda_xml.replace(tag, new_tag)
    print("Gripper actuator tag:", new_tag)

# ============================================================
# WRIST CAMERA
# ============================================================

wrist_camera = """
<camera
    name="wrist_camera"
    mode="fixed"
    pos="-0.08 0.08 0.12"
    euler="0 -35 -5.6"
    fovy="75"
/>
"""

panda_xml = panda_xml.replace(
    '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">',
    '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">'
    + wrist_camera
)


# ============================================================
# OVERVIEW CAMERA
# ============================================================

overview_camera = """
<camera
    name="overview_camera"
    pos="1.500 0.000 1.000"
    quat="0.500000 0.500000 0.500000 0.500000"
    fovy="72"
/>
"""

panda_xml = panda_xml.replace(
    "<worldbody>",
    "<worldbody>" + overview_camera
)


# ============================================================
# BOTTLE ASSETS
# ============================================================

bottle_assets = f"""

    <texture
        name="wine_bottle_texture"
        type="2d"
        file="{BOTTLE_TEXTURE}"
    />

    <material
        name="wine_bottle_material"
        texture="wine_bottle_texture"
    />

    <mesh
        name="wine_bottle"
        file="{BOTTLE_OBJ}"
        scale="0.01418 0.01418 0.01418"
    />
"""

panda_xml = panda_xml.replace(
    "</asset>",
    bottle_assets + "\n</asset>"
)


# ============================================================
# BIN + BOTTLES
# ============================================================

bin_and_bottles = """

    <!-- ================================================== -->
    <!-- BIN FLOOR                                          -->
    <!-- ================================================== -->

    <geom
        name="bin_floor"
        type="box"
        pos="0.45 0 0.35"
        size="0.30 0.30 0.05"
        rgba="0.35 0.35 0.35 1"
        contype="1"
        conaffinity="1"
    />


    <!-- ================================================== -->
    <!-- BIN WALLS                                          -->
    <!-- ================================================== -->

    <geom
        name="bin_back_wall"
        type="box"
        pos="0.45 0.2925 0.39"
        size="0.30 0.0075 0.04"
        rgba="0.35 0.35 0.35 1"
        contype="1"
        conaffinity="1"
    />

    <geom
        name="bin_front_wall"
        type="box"
        pos="0.45 -0.2925 0.39"
        size="0.30 0.0075 0.04"
        rgba="0.35 0.35 0.35 1"
        contype="1"
        conaffinity="1"
    />

    <geom
        name="bin_left_wall"
        type="box"
        pos="0.1575 0 0.39"
        size="0.0075 0.30 0.04"
        rgba="0.35 0.35 0.35 1"
        contype="1"
        conaffinity="1"
    />

    <geom
        name="bin_right_wall"
        type="box"
        pos="0.7425 0 0.39"
        size="0.0075 0.30 0.04"
        rgba="0.35 0.35 0.35 1"
        contype="1"
        conaffinity="1"
    />


    <!-- ================================================== -->
    <!-- BOTTLE 1                                           -->
    <!-- ================================================== -->

    <body
        name="bottle_1"
        pos="0.30 0.15 0.40"
    >

        <freejoint/>

        <geom
            name="bottle_1_geom"
            type="mesh"
            mesh="wine_bottle"
            material="wine_bottle_material"
            mass="0.3"
            friction="2.0 0.1 0.01"
            contype="1"
            conaffinity="1"
        />

    </body>


    <!-- ================================================== -->
    <!-- BOTTLE 2                                           -->
    <!-- ================================================== -->

    <body
        name="bottle_2"
        pos="0.60 0.15 0.40"
    >

        <freejoint/>

        <geom
            name="bottle_2_geom"
            type="mesh"
            mesh="wine_bottle"
            material="wine_bottle_material"
            mass="0.3"
            friction="2.0 0.1 0.01"
            contype="1"
            conaffinity="1"
        />

    </body>


    <!-- ================================================== -->
    <!-- BOTTLE 3                                           -->
    <!-- ================================================== -->

    <body
        name="bottle_3"
        pos="0.30 -0.15 0.40"
    >

        <freejoint/>

        <geom
            name="bottle_3_geom"
            type="mesh"
            mesh="wine_bottle"
            material="wine_bottle_material"
            mass="0.3"
            friction="2.0 0.1 0.01"
            contype="1"
            conaffinity="1"
        />

    </body>


    <!-- ================================================== -->
    <!-- BOTTLE 4                                           -->
    <!-- ================================================== -->

    <body
        name="bottle_4"
        pos="0.50 0.00 0.40"
    >

        <freejoint/>

        <geom
            name="bottle_4_geom"
            type="mesh"
            mesh="wine_bottle"
            material="wine_bottle_material"
            mass="0.3"
            friction="2.0 0.1 0.01"
            contype="1"
            conaffinity="1"
        />

    </body>
"""


panda_xml = panda_xml.replace(
    "</worldbody>",
    bin_and_bottles + "\n</worldbody>"
)


# ============================================================
# FINAL SCENE
# ============================================================

SCENE_XML = panda_xml