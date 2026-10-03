import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathlib import Path
import random
import re
import mujoco
import mujoco.viewer
import math
from testing.bar_operation import run_operation
# ============================================================
# PATHS
# ============================================================
WALL_PARTS = (
    set(range(1, 28))
    | {29}
    | set(range(31, 142))
    | set(range(146, 170))
    | set(range(180, 192))
)
WALL_OFFSET_Y = 0.25

BAR_EULER = f"{math.radians(90):.6f} 0 0"   # was "90 0 0"

PROJECT_DIR = Path(__file__).resolve().parent.parent
BAR_DIR = PROJECT_DIR / "Bar"
BAR_SCALE = 0.01
BAR_EULER = f"{math.radians(90):.6f} 0 0"

BOTTLE_DIR = PROJECT_DIR / "bottle"
BOTTLE_OBJ = BOTTLE_DIR / "14042_750_mL_Wine_Bottle_r_v1_L3.obj"
BOTTLE_TEXTURE = BOTTLE_DIR / "14042_750_mL_Wine_Bottle_dfinal.png"
BOTTLE_SCALE = 0.01418

BIN_DIR = PROJECT_DIR / "Bin"
BIN_OBJ = BIN_DIR / "Metal+Storage+Bin.obj"
BIN_SCALE = 1.0

STOOL_DIR = PROJECT_DIR / "Chair"
STOOL_OBJ = STOOL_DIR / "Chair.obj"
STOOL_SCALE = 0.2085

PANDA_DIR = PROJECT_DIR / "mujoco_menagerie" / "franka_emika_panda"
PANDA_XML = PANDA_DIR / "panda.xml"
PANDA_ASSETS = PANDA_DIR / "assets"


# ============================================================
# BAR COUNTER PARTS
# ============================================================

parts = sorted(
    (BAR_DIR / "parts").glob("part_*.obj"),
    key=lambda p: int(p.stem.split("_")[1])
)

DEFAULT_STYLE = dict(rgba="0.15 0.15 0.18 1", emission=0)
EXCLUDE_PARTS = {0}
NEON_PARTS = {2, 3}
RING_PARTS = set(range(4, 15))

bar_assets, bar_geoms = "", ""
for i, p in enumerate(parts):
    if i in EXCLUDE_PARTS:
        continue
    if i in NEON_PARTS:
        s = dict(rgba="0.1 0.4 1 1", emission=3)
    elif i in RING_PARTS:
        s = dict(rgba="1 0.7 0.2 1", emission=2.5)
    elif i == 15:
        s = dict(rgba="0.04 0.04 0.05 1", emission=0)
    else:
        s = DEFAULT_STYLE

    bar_assets += (
        f'<mesh name="bar_{i}" file="{p}" scale="{BAR_SCALE} {BAR_SCALE} {BAR_SCALE}" inertia="shell"/>\n'
        f'<material name="bar_mat_{i}" rgba="{s["rgba"]}" emission="{s["emission"]}" specular="0.8" shininess="0.9"/>\n'
    )
    geom_pos = (
    f'pos="0 {WALL_OFFSET_Y} 0"'
    if i in WALL_PARTS
    else 'pos="0 0 0"'
    )

    bar_geoms += (
        f'<geom name="bar_g{i}" type="mesh" mesh="bar_{i}" material="bar_mat_{i}" '
        f'{geom_pos} euler="{BAR_EULER}" contype="0" conaffinity="0"/>\n'
    )


# ============================================================
# COUNTER TOP COLLISION SURFACE (invisible)
# ============================================================

COUNTER_TOP_Z = 0.485
COUNTER_X_RANGE = (-1.5, 1.5)
COUNTER_Y_RANGE = (0.22, 0.30)

COLLIDER_Y_MARGIN = 0.15
COLLIDER_X_MARGIN = 0.20

counter_collision = f"""
<geom name="counter_top_collider" type="box"
      size="{(COUNTER_X_RANGE[1]-COUNTER_X_RANGE[0])/2 + COLLIDER_X_MARGIN} {(COUNTER_Y_RANGE[1]-COUNTER_Y_RANGE[0])/2 + COLLIDER_Y_MARGIN} 0.15"
      pos="0 {(COUNTER_Y_RANGE[0]+COUNTER_Y_RANGE[1])/2} {COUNTER_TOP_Z - 0.15}"
      rgba="0 0 0 0"
      contype="1" conaffinity="1"/>
"""


# ============================================================
# BOTTLES
# ============================================================

bottle_assets = f"""
<texture name="wine_bottle_texture" type="2d" file="{BOTTLE_TEXTURE}"/>
<material name="wine_bottle_material" texture="wine_bottle_texture"/>
<mesh name="wine_bottle" file="{BOTTLE_OBJ}" scale="{BOTTLE_SCALE} {BOTTLE_SCALE} {BOTTLE_SCALE}"/>
"""

N_BOTTLES = 1
random.seed(42)
MIN_SPACING = 0.12

placed = []
bottle_bodies = ""
for i in range(1, N_BOTTLES + 1):
    for attempt in range(100):
        x = random.uniform(*COUNTER_X_RANGE)
        y = random.uniform(*COUNTER_Y_RANGE)
        y += 0.10
        if all((x - px)**2 + (y - py)**2 >= MIN_SPACING**2 for px, py in placed):
            break
    placed.append((x, y))

    yaw = random.uniform(0, 360)
    spawn_z = COUNTER_TOP_Z

    bottle_bodies += f"""
    <body name="bottle_{i}" pos="{x:.3f} {y:.3f} {spawn_z:.3f}" euler="0 0 {yaw:.1f}">
        <freejoint/>
        <geom name="bottle_{i}_geom" type="mesh" mesh="wine_bottle" material="wine_bottle_material"
              mass="0.0001" contype="0" conaffinity="0"/>
        <geom type="cylinder" fromto="0 0 0 0 0 0.20"    size="0.0368" mass="0.22" friction="2.0 0.1 0.01" rgba="0 0 0 0"/>
        <geom type="cylinder" fromto="0 0 0.20 0 0 0.22" size="0.030"  mass="0.03" friction="2.0 0.1 0.01" rgba="0 0 0 0"/>
        <geom type="cylinder" fromto="0 0 0.22 0 0 0.24" size="0.019"  mass="0.02" friction="2.0 0.1 0.01" rgba="0 0 0 0"/>
        <geom type="cylinder" fromto="0 0 0.24 0 0 0.30" size="0.0148" mass="0.03" friction="2.0 0.1 0.01" rgba="0 0 0 0"/>
    </body>
    """


# ============================================================
# STOOL
# ============================================================

STOOL_POS = (1.3, 0.65, -0.35)

stool_parts = sorted((STOOL_DIR / "parts").glob("*.obj"))

stool_assets, stool_geom = "", ""
for i, p in enumerate(stool_parts):
    stool_assets += (
        f'<mesh name="stool_{i}" file="{p}" scale="{STOOL_SCALE} {STOOL_SCALE} {STOOL_SCALE}" inertia="shell"/>\n'
        f'<material name="stool_mat_{i}" rgba="0.08 0.08 0.08 1" specular="0.6" shininess="0.6"/>\n'
    )
    stool_geom += (
    f'<geom name="stool_g{i}" type="mesh" mesh="stool_{i}" material="stool_mat_{i}" '
    f'pos="{STOOL_POS[0]} {STOOL_POS[1]} {STOOL_POS[2]}" euler="{math.radians(90):.6f} 0 0" '
    f'contype="0" conaffinity="0"/>\n'
    )


# ============================================================
# STORAGE BIN
# ============================================================

bin_assets = f"""
<material name="bin_mat" rgba="0.05 0.05 0.05 1" specular="0.7" shininess="0.7"/>
<mesh name="storage_bin" file="{BIN_OBJ}" scale="{BIN_SCALE} {BIN_SCALE} {BIN_SCALE}" inertia="shell"/>
"""

BIN_POS = (1.3, 0.65, 0.25)

bin_geom = f"""
<geom name="storage_bin_geom" type="mesh" mesh="storage_bin" material="bin_mat"
      pos="{BIN_POS[0]} {BIN_POS[1]} {BIN_POS[2]}" euler="{math.radians(90):.6f} 0 0"
      contype="0" conaffinity="0"/>
"""


# ============================================================
# ROBOT (Franka Panda) — parsed properly with ElementTree
# ============================================================

import xml.etree.ElementTree as ET

panda_raw = PANDA_XML.read_text()
panda_raw = re.sub(r'meshdir="[^"]*"', f'meshdir="{PANDA_ASSETS}"', panda_raw)

# stronger gripper
m = re.search(r'<[^<>]*name="actuator8"[^<>]*>', panda_raw, re.S)
if m:
    tag = m.group(0)
    new_tag = re.sub(r'biasprm="[^"]*"', 'biasprm="0 -1000 -10"', tag)
    new_tag = re.sub(r'gainprm="[^"]*"', 'gainprm="0.1568627451 0 0"', new_tag)
    panda_raw = panda_raw.replace(tag, new_tag)

# wrist camera
wrist_camera = """
<camera name="wrist_camera" mode="fixed"
        pos="-0.08 0.08 0.12" euler="0 -35 -5.6" fovy="75"/>
"""
panda_raw = panda_raw.replace(
    '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">',
    '<body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">' + wrist_camera
)

root = ET.fromstring(panda_raw)

def block_to_string(elem):
    return ET.tostring(elem, encoding="unicode") if elem is not None else ""

def inner_to_string(elem):
    if elem is None:
        return ""
    return "".join(ET.tostring(child, encoding="unicode") for child in elem)

panda_compiler_block  = block_to_string(root.find("compiler"))
panda_default_block   = block_to_string(root.find("default"))
panda_assets_block    = inner_to_string(root.find("asset"))
panda_worldbody_block = inner_to_string(root.find("worldbody"))
panda_tendon_block    = block_to_string(root.find("tendon"))
panda_actuator_block  = block_to_string(root.find("actuator"))
panda_equality_block  = block_to_string(root.find("equality"))
panda_contact_block   = block_to_string(root.find("contact"))
panda_sensor_block    = block_to_string(root.find("sensor"))

ROBOT_POS = (0.63, 0.59, 0)
panda_worldbody_wrapped = (
    f'<body name="panda_base" '
    f'pos="{ROBOT_POS[0]} {ROBOT_POS[1]} {ROBOT_POS[2]}" '
    f'euler="0 0 {math.radians(180):.6f}">'
    f'{panda_worldbody_block}'
    f'</body>'
)


# ============================================================
# FULL SCENE
# ============================================================

XML = f"""
<mujoco>

  <option integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="5"/>

  {panda_compiler_block}
  {panda_default_block}

  <visual>
    <headlight ambient="0.65 0.65 0.65" diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15"/>
    <quality shadowsize="8192"/>
    <global offwidth="1280" offheight="960"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.06 0.07 0.10" rgb2="0 0 0" width="512" height="3072"/>

    <material name="floor_mat" rgba="0.08 0.08 0.1 1"
              specular="0.8" shininess="0.9" reflectance="0.3"/>

    {bar_assets}
    {bottle_assets}
    {stool_assets}
    {bin_assets}
    {panda_assets_block}
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"/>

    {bar_geoms}
    {counter_collision}
    {stool_geom}
    {bin_geom}
    {panda_worldbody_wrapped}
    {bottle_bodies}

    <light directional="true" pos="0 0 5" dir="0.3 0.4 -1"
           diffuse="1.2 1.15 1.1" specular="0.4 0.4 0.4" castshadow="true"/>
    <light directional="true" pos="0 0 5" dir="-0.3 0.4 -1"
           diffuse="0.7 0.7 0.75" specular="0.2 0.2 0.2" castshadow="false"/>
    <light directional="true" pos="0 0 5" dir="0 -1 -0.3"
           diffuse="0.5 0.55 0.6" specular="0.1 0.1 0.1" castshadow="false"/>
  </worldbody>

  {panda_tendon_block}
  {panda_actuator_block}
  {panda_equality_block}
  {panda_contact_block}
  {panda_sensor_block}

</mujoco>
"""

SCENE_XML = XML
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960


if __name__ == "__main__":

    model = mujoco.MjModel.from_xml_string(SCENE_XML)

    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    bottle_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "bottle_1"
    )

    bottle_position = data.xpos[bottle_id].copy()

    print(
        f"GROUND TRUTH bottle_1 -> "
        f"X={bottle_position[0]:.3f}, "
        f"Y={bottle_position[1]:.3f}, "
        f"Z={bottle_position[2]:.3f}"
    )

    renderer = mujoco.Renderer(
        model=model,
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:

        run_operation(model, data, viewer,renderer)