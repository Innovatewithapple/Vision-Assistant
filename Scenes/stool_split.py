
import trimesh
from pathlib import Path

STOOL_DIR = Path(__file__).resolve().parent.parent / "Chair"
OUT = STOOL_DIR / "parts"
OUT.mkdir(exist_ok=True)

scene = trimesh.load(STOOL_DIR / "Chair.obj", force="scene")
for name, geom in scene.geometry.items():
    fn = f"{name}.obj"
    geom.export(OUT / fn)
    print(f"wrote {fn}: faces={len(geom.faces)}")