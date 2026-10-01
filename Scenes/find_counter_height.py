import sys
from pathlib import Path
import trimesh
STOOL_OBJ = Path(__file__).resolve().parent.parent / "Chair" / "Chair.obj"

# Load as a Scene first (preserves sub-objects/groups) instead of force="mesh"
scene = trimesh.load(STOOL_OBJ, force="scene")
print("Number of separate geometries in file:", len(scene.geometry))
for name, geom in scene.geometry.items():
    print(f"  {name}: faces={len(geom.faces)}, bounds={geom.bounds.tolist()}")