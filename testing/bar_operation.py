import mujoco
import numpy as np
import cv2
from YOLO.segmentation import Segmentation
from Visualisation.visualisation import Draw_Segmentation

#-----Segmentation---@
segmentor = Segmentation()

def run_operation(model, data, viewer, renderer):
    print("Operation connected to scene.")

    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")

    while viewer.is_running():
        mujoco.mj_step(model, data)

        # --------------------------------------------------
        # ALWAYS run wrist-camera segmentation, every step,
        # regardless of which camera the viewer is showing
        # --------------------------------------------------
        renderer.update_scene(data, camera='wrist_camera')
        wrist_frame = renderer.render()
        wrist_frame = np.asarray(wrist_frame)
        wrist_frame_bgr = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)
        
        boxes, labels, scores, mask, track_ids, class_names = segmentor.segment(frame=wrist_frame_bgr)

        # this is where your actual robot logic would use boxes/labels/mask, etc.
        # (localization, IK, grasp decisions) — runs every step no matter what.

        wrist_frame_bgr = Draw_Segmentation(wrist_frame_bgr, boxes, labels, scores, mask, track_ids, class_names,model,data)
        display_frame = cv2.cvtColor(wrist_frame_bgr, cv2.COLOR_BGR2RGB)

        # --------------------------------------------------
        # ONLY show the overlay when viewer is on wrist camera
        # --------------------------------------------------
        is_wrist = (
            viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
            and viewer.cam.fixedcamid == wrist_camera_id
        )

        if is_wrist:
            vp = viewer.viewport

            # resize the full render down to fit the available viewport — nothing gets cut off
            resized_frame = cv2.resize(display_frame, (vp.width, vp.height))

            viewport_rect = mujoco.MjrRect(vp.left, vp.bottom, vp.width, vp.height)
            viewer.set_images((viewport_rect, resized_frame))
        else:
            viewer.clear_images()

        viewer.sync()