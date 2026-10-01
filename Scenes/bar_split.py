import trimesh
from pathlib import Path

BAR_DIR = Path(__file__).resolve().parent.parent / "Bar"
OUT = BAR_DIR / "parts"

OUT.mkdir(exist_ok=True)

m = trimesh.load(
    BAR_DIR / "Barcounter.obj",
    force="mesh"
)

print("size:", m.extents)
print("bounds:\n", m.bounds)

pieces = sorted(
    m.split(only_watertight=False),
    key=lambda p: -len(p.faces)
)

print("total pieces:", len(pieces))


# ------------------------------------------------------------
# EXPORT EVERY COMPONENT SEPARATELY
# ------------------------------------------------------------

for i, p in enumerate(pieces):

    trimesh.repair.fix_normals(p)

    p.export(
        OUT / f"part_{i}.obj"
    )

    print(
        f"part_{i}: "
        f"faces={len(p.faces)} "
        f"size={p.extents} "
        f"center={p.centroid} "
        f"bounds={p.bounds}"
    )