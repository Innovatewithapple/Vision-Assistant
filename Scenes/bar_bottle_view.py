from pathlib import Path
import random
import mujoco
import mujoco.viewer

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
BAR_DIR = PROJECT_DIR / "Bar"
BAR_SCALE = 0.01
BAR_EULER = "90 0 0"

BOTTLE_DIR = PROJECT_DIR / "bottle"
BOTTLE_OBJ = BOTTLE_DIR / "14042_750_mL_Wine_Bottle_r_v1_L3.obj"
BOTTLE_TEXTURE = BOTTLE_DIR / "14042_750_mL_Wine_Bottle_dfinal.png"
BOTTLE_SCALE = 0.01418

BIN_DIR = PROJECT_DIR / "Bin"
BIN_OBJ = BIN_DIR / "Metal+Storage+Bin.obj"
BIN_SCALE = 1.0


# ============================================================
# BAR COUNTER PARTS
# ============================================================

parts = sorted(
    (BAR_DIR / "parts").glob("part_*.obj"),
    key=lambda p: int(p.stem.split("_")[1])
)

DEFAULT_STYLE = dict(rgba="0.15 0.15 0.18 1", emission=0)
EXCLUDE_PARTS = {0}              # character silhouette
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
    bar_geoms += (
        f'<geom name="bar_g{i}" type="mesh" mesh="bar_{i}" material="bar_mat_{i}" '
        f'euler="{BAR_EULER}" contype="0" conaffinity="0"/>\n'
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

N_BOTTLES = 6
random.seed(42)

MIN_SPACING = 0.12

placed = []
bottle_bodies = ""
for i in range(1, N_BOTTLES + 1):
    for attempt in range(100):
        x = random.uniform(*COUNTER_X_RANGE)
        y = random.uniform(*COUNTER_Y_RANGE)
        if all((x - px)**2 + (y - py)**2 >= MIN_SPACING**2 for px, py in placed):
            break
    placed.append((x, y))

    yaw = random.uniform(0, 360)
    spawn_z = COUNTER_TOP_Z

    bottle_bodies += f"""
    <body name="bottle_{i}" pos="{x:.3f} {y:.3f} {spawn_z:.3f}" euler="0 0 {yaw:.1f}">
        <freejoint/>
        <geom name="bottle_{i}_geom" type="mesh" mesh="wine_bottle" material="wine_bottle_material"
              mass="0.3" friction="2.0 0.1 0.01"
              contype="1" conaffinity="1"/>
    </body>
    """


# ============================================================
# STORAGE BIN (behind counter, robot side)
# ============================================================

bin_assets = f"""
<material name="bin_mat" rgba="0.05 0.05 0.05 1" specular="0.7" shininess="0.7"/>
<mesh name="storage_bin" file="{BIN_OBJ}" scale="{BIN_SCALE} {BIN_SCALE} {BIN_SCALE}" inertia="shell"/>
"""

BIN_POS = (1.3, 0.55, 0.0)

bin_geom = f"""
<geom name="storage_bin_geom" type="mesh" mesh="storage_bin" material="bin_mat"
      pos="{BIN_POS[0]} {BIN_POS[1]} {BIN_POS[2]}" euler="90 0 0"
      contype="0" conaffinity="0"/>
"""

# ============================================================
# FULL SCENE — same lighting/floor as bar_view.py
# ============================================================

XML = f"""
<mujoco>

  <option cone="elliptic" impratio="10" noslip_iterations="5"/>

  <visual>
    <headlight ambient="0.65 0.65 0.65" diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>
    <texture name="sky" type="skybox" builtin="gradient"
             rgb1="0.06 0.07 0.10" rgb2="0 0 0" width="512" height="3072"/>

    <material name="floor_mat" rgba="0.08 0.08 0.1 1"
              specular="0.8" shininess="0.9" reflectance="0.3"/>

    {bar_assets}
    {bottle_assets}
    {bin_assets}
  </asset>

  <worldbody>
    <geom name="floor" type="plane" size="6 6 0.1" material="floor_mat"/>

    {bar_geoms}
    {counter_collision}
    {bottle_bodies}
    {bin_geom}

    <light directional="true" pos="0 0 5" dir="0.3 0.4 -1"
           diffuse="1.2 1.15 1.1" specular="0.4 0.4 0.4" castshadow="true"/>
    <light directional="true" pos="0 0 5" dir="-0.3 0.4 -1"
           diffuse="0.7 0.7 0.75" specular="0.2 0.2 0.2" castshadow="false"/>
    <light directional="true" pos="0 0 5" dir="0 -1 -0.3"
           diffuse="0.5 0.55 0.6" specular="0.1 0.1 0.1" castshadow="false"/>
  </worldbody>

</mujoco>
"""

SCENE_XML = XML

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)