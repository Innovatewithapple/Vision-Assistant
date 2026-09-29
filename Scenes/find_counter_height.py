import sys
from pathlib import Path
import trimesh
BIN_OBJ = Path(__file__).resolve().parent.parent / "Bin" / "Metal+Storage+Bin.obj"
m = trimesh.load(BIN_OBJ, force="mesh")
print("Bin bounds (min,max):", m.bounds)
print("Bin size (x,y,z):", m.extents)