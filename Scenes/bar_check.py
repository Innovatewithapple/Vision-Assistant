import trimesh
from pathlib import Path

obj = Path(__file__).resolve().parent / "Bar" / "Barcounter.obj"
m = trimesh.load(obj, force="mesh")

print("bounds (min, max):\n", m.bounds)
print("size (x, y, z):", m.extents)
print("faces:", len(m.faces), " vertices:", len(m.vertices))

parts = m.split(only_watertight=False)
print("separate pieces:", len(parts))
for i, p in enumerate(sorted(parts, key=lambda p: -len(p.faces))[:8]):
    print(f"  piece {i}: faces={len(p.faces)}, size={p.extents}")

# write a double-sided copy (every face duplicated with flipped direction)
flipped = m.copy()
flipped.invert()
both = trimesh.util.concatenate([m, flipped])
both.export(obj.with_name("Barcounter_2side.obj"))
print("wrote Barcounter_2side.obj")