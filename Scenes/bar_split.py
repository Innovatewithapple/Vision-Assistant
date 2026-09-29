import trimesh
from pathlib import Path

BAR_DIR = Path(__file__).resolve().parent.parent / "Bar"
OUT = BAR_DIR / "parts"
OUT.mkdir(exist_ok=True)

m = trimesh.load(BAR_DIR / "Barcounter.obj", force="mesh")
print("size:", m.extents, "\nbounds:\n", m.bounds)

pieces = sorted(m.split(only_watertight=False), key=lambda p: -len(p.faces))
print("total pieces:", len(pieces))

keep = pieces[:15]                      # the 15 biggest pieces stay separate
rest = pieces[15:]
if rest:
    keep.append(trimesh.util.concatenate(rest))   # all tiny bits merged into one

for i, p in enumerate(keep):
    trimesh.repair.fix_normals(p)
    p.export(OUT / f"part_{i}.obj")
    print(f"part_{i}: faces={len(p.faces)} size={p.extents}")