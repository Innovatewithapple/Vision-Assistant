from pathlib import Path
import numpy as np

BOTTLE_OBJ = Path("bottle/14042_750_mL_Wine_Bottle_r_v1_L3.obj")

vertices = []

with open(BOTTLE_OBJ, "r") as f:
    for line in f:
        if line.startswith("v "):
            _, x, y, z = line.split()
            vertices.append([float(x), float(y), float(z)])

vertices = np.array(vertices)

obj_min = vertices.min(axis=0)
obj_max = vertices.max(axis=0)

raw_dimensions = obj_max - obj_min

print("RAW OBJ:")
print("Min:", obj_min)
print("Max:", obj_max)
print("Dimensions:", raw_dimensions)

# XML scale = 0.1 0.1 0.1
scale = np.array([0.1, 0.1, 0.1])

scaled_dimensions = raw_dimensions * scale

print("\nAFTER XML SCALE:")
print(f"Width  : {scaled_dimensions[0]:.4f} m")
print(f"Depth  : {scaled_dimensions[1]:.4f} m")
print(f"Height : {scaled_dimensions[2]:.4f} m")