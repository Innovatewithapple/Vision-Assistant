# scene_check.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer
from Scenes.bar_bottle_view import SCENE_XML

from pathlib import Path
import trimesh

PROJECT_DIR = Path(__file__).resolve().parent.parent
BOTTLE_DIR = PROJECT_DIR / "bottle"
BOTTLE_OBJ = BOTTLE_DIR / "14042_750_mL_Wine_Bottle_r_v1_L3.obj"
BOTTLE_SCALE = 0.01418

m = trimesh.load(BOTTLE_OBJ, force="mesh")
raw_radius = (m.bounds[1][0] - m.bounds[0][0]) / 2  # X-extent / 2
real_radius = raw_radius * BOTTLE_SCALE
print("Bottle base radius (m):", real_radius)