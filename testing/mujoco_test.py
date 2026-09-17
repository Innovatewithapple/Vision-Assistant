import trimesh
from pathlib import Path


obj_file = Path("human/rp_dennis_posed_004_30k.OBJ")


mesh = trimesh.load(
    obj_file,
    force="mesh"
)


print("Loaded successfully!")
print()

print("Vertices:", len(mesh.vertices))
print("Faces:", len(mesh.faces))
print()

print("Minimum XYZ:")
print(mesh.bounds[0])

print()

print("Maximum XYZ:")
print(mesh.bounds[1])

print()

print("Dimensions XYZ:")
print(mesh.extents)

print()

print("Has vertex normals:")
print(mesh.vertex_normals is not None)

print()

print("Has UV coordinates:")

if mesh.visual.uv is not None:
    print(True)
    print("UV shape:", mesh.visual.uv.shape)
else:
    print(False)