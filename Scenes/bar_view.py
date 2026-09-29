from pathlib import Path
import mujoco
import mujoco.viewer
import trimesh
import numpy as np
import re
import random

# ============================================================
# BAR PATHS
# ============================================================

BAR_DIR = Path(__file__).resolve().parent.parent / "Bar"

BAR_SCALE = 0.01
BAR_EULER = "90 0 0"


# ============================================================
# LOAD SPLIT BAR PARTS
# ============================================================

parts = sorted(
    (BAR_DIR / "parts").glob("part_*.obj"),
    key=lambda p: int(p.stem.split("_")[1])
)

m15 = trimesh.load(parts[15], force="mesh")
print("part 15 bounds min/max:", m15.bounds)

verts = m15.vertices

# raw y-axis = height (becomes world Z after rotation).
# Try to isolate just the counter/bar box, which should be
# much shorter than the tall wall panel behind it.
low_mask = verts[:, 1] < 20   # adjust threshold after seeing the printout
counter_verts = verts[low_mask]

print("Full mesh raw Y (height) range:", verts[:,1].min(), verts[:,1].max())
print("Full mesh raw Y (height) range:", verts[:,1].min(), verts[:,1].max())


# ============================================================
# MATERIAL STYLE FOR EACH BAR PART
# ============================================================

DEFAULT_STYLE = dict(
    rgba="0.15 0.15 0.18 1",
    emission=0
)

NEON_PARTS = {2, 3}             # thin outline + gold trim border
RING_PARTS = set(range(4, 15))  # 11 ring lights

EXCLUDE_PARTS = {0}   # the character silhouette panel by this index it no longer visible in the scene

part_assets, part_geoms = "", ""
for i, p in enumerate(parts):
    if i in EXCLUDE_PARTS:
        continue

    if i in NEON_PARTS:
        s = dict(rgba="0.1 0.4 1 1", emission=3)
    elif i in RING_PARTS:
        s = dict(rgba="1 0.7 0.2 1", emission=2.5)
    elif i == 15:
        s = dict(rgba="0.04 0.04 0.05 1", emission=0)
    else:
        s = DEFAULT_STYLE

    part_assets += (
        f'<mesh name="bar_{i}" file="{p}" scale="{BAR_SCALE} {BAR_SCALE} {BAR_SCALE}" inertia="shell"/>\n'
        f'<material name="bar_mat_{i}" rgba="{s["rgba"]}" emission="{s["emission"]}" specular="0.8" shininess="0.9"/>\n'
    )
    part_geoms += (
        f'<geom name="bar_g{i}" type="mesh" mesh="bar_{i}" material="bar_mat_{i}" '
        f'euler="{BAR_EULER}" contype="0" conaffinity="0"/>\n'
    )


# ============================================================
# TWO NEON STRIPS — top edge (gold) + bottom edge (pink)
# ============================================================

def raw_to_world(x, y, z):
    return np.array([x * BAR_SCALE, -z * BAR_SCALE, y * BAR_SCALE])

def trace_at_height(all_verts, y_center, initial_band=1.0):
    for band in [initial_band, 2, 4, 8, 16, 30]:
        mask = np.abs(all_verts[:, 1] - y_center) < band
        layer = all_verts[mask]
        if len(layer) >= 2:
            break
    else:
        raise ValueError(f"No vertices near y={y_center}")

    z_max = layer[:, 2].max()
    edge_verts = np.empty((0, 3))
    for relax in [0.5, 1, 2, 4, 8, 16, 30, 100]:
        m = layer[:, 2] > z_max - relax
        if m.sum() >= 2:
            edge_verts = layer[m]
            break

    order = np.argsort(edge_verts[:, 0])
    edge_verts = edge_verts[order]
    ex, ey, ez = edge_verts[:, 0], edge_verts[:, 1], edge_verts[:, 2]

    n_samples = 80
    sx = np.linspace(ex.min(), ex.max(), n_samples)
    sy = np.interp(sx, ex, ey)
    sz = np.interp(sx, ex, ez)
    pts = np.array([raw_to_world(x, y, z) for x, y, z in zip(sx, sy, sz)])
    pts[:, 1] += 0.005
    return pts

TOP_Y = 20
BOTTOM_Y = 0.15

top_pts = trace_at_height(verts, TOP_Y)
top_pts[:, 2] += 0.01

bottom_pts = trace_at_height(verts, BOTTOM_Y)
bottom_pts[:, 2] -= 0.01

def make_strip_geoms(pts, material_name):
    out = ""
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        out += (
            f'<geom type="capsule" fromto="{a[0]:.4f} {a[1]:.4f} {a[2]:.4f} '
            f'{b[0]:.4f} {b[1]:.4f} {b[2]:.4f}" size="0.008" '
            f'material="{material_name}" contype="0" conaffinity="0"/>\n'
        )
    return out

strip_assets = (
    '<material name="strip_mat_top" rgba="1.0 0.55 0.25 1" emission="1.2" specular="0.6" shininess="0.6"/>\n'
    '<material name="strip_mat_bottom" rgba="1.0 0.2 0.5 1" emission="1.2" specular="0.6" shininess="0.6"/>\n'
)

strip_geoms = (
    make_strip_geoms(top_pts, "strip_mat_top")
    + make_strip_geoms(bottom_pts, "strip_mat_bottom")
)
# ============================================================
# MUJOCO XML
# ============================================================

XML = f"""
<mujoco>

  <visual>
    <headlight ambient="0.65 0.65 0.65" diffuse="0.6 0.6 0.6" specular="0.15 0.15 0.15"/>
    <quality shadowsize="8192"/>
  </visual>

  <asset>

    <texture
        name="sky"
        type="skybox"
        builtin="gradient"
        rgb1="0.06 0.07 0.10"
        rgb2="0 0 0"
        width="512"
        height="3072"
    />

    <material
        name="floor_mat"
        rgba="0.08 0.08 0.1 1"
        specular="0.8"
        shininess="0.9"
        reflectance="0.3"
    />

    {part_assets}
    {strip_assets}

  </asset>

  <worldbody>

    <geom
        name="floor"
        type="plane"
        size="6 6 0.1"
        material="floor_mat"
    />

    {part_geoms}
    {strip_geoms}

    <light directional="true" pos="0 0 5" dir="0.3 0.4 -1"
           diffuse="1.2 1.15 1.1" specular="0.4 0.4 0.4" castshadow="true"/>

    <light directional="true" pos="0 0 5" dir="-0.3 0.4 -1"
           diffuse="0.7 0.7 0.75" specular="0.2 0.2 0.2" castshadow="false"/>

    <light directional="true" pos="0 0 5" dir="0 -1 -0.3"
           diffuse="0.5 0.55 0.6" specular="0.1 0.1 0.1" castshadow="false"/>

  </worldbody>

</mujoco>
"""


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)


# ============================================================
# LAUNCH VIEWER
# ============================================================

mujoco.viewer.launch(model, data)