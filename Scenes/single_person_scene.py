from pathlib import Path

# =====================================================
# MUJOCO SCENE
# =====================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_DIR / "human"
TEXTURE_DIR = MODEL_DIR / "tex"

OBJ_FILE = MODEL_DIR / "rp_dennis_posed_004_30k.OBJ"

DIFFUSE_PNG = TEXTURE_DIR / "rp_dennis_posed_004_dif.png"

xml = f"""
<mujoco>

    <compiler
        meshdir="{MODEL_DIR}"
        texturedir="{TEXTURE_DIR}"
    />

    <visual>
        <global
            offwidth="1280"
            offheight="720"
        />
    </visual>

    <asset>

        <!-- Human OBJ -->
        <mesh
            name="human"
            file="{OBJ_FILE.name}"
            scale="0.01 0.01 0.01"
        />

        <!-- Human clothing/body texture -->
        <texture
            name="human_texture"
            type="2d"
            file="{DIFFUSE_PNG.name}"
        />

        <material
            name="human_material"
            texture="human_texture"
        />

    </asset>

    <worldbody>

        <!-- Lighting -->

        <light
            pos="0 -4 6"
            directional="false"
        />

        <light
            pos="3 -2 4"
            directional="false"
        />

        <!-- Ground -->

        <geom
            type="plane"
            size="20 20 0.1"
            rgba="0.65 0.65 0.65 1"
        />

        <!-- Human -->

        <body
            name="human"
            pos="0 -3 0.011"
            euler="0 0 -90"
        >

            <geom
                type="mesh"
                mesh="human"
                material="human_material"
                euler="90 0 0"
                contype="0"
                conaffinity="0"
            />

        </body>

        <!-- First-person camera -->

        <camera
            name="main_camera"
            pos="0 -8 1.45"
            euler="90 0 0"
            fovy="60"
        />

    </worldbody>

</mujoco>
"""