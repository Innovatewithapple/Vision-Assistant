# scene_check.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer
from Scenes.bar_bottle_view import SCENE_XML

model = mujoco.MjModel.from_xml_string(SCENE_XML)
data = mujoco.MjData(model)
mujoco.viewer.launch(model, data)