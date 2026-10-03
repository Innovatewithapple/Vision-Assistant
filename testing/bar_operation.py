import mujoco
import numpy as np
import cv2
from YOLO.segmentation import Segmentation
from Visualisation.visualisation import Draw_Segmentation
from Visualisation.visualisation import disable_detection_and_get_bottles, approach_bottle, set_truth
from Calculation.panda_ik import PandaIK, drive_to_target

#-----Segmentation---@
segmentor = Segmentation()

# bar_operation.py (top of file, after imports)
def bottle_truth(model, data, name="bottle_1_geom"):
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    mid = model.geom_dataid[gid]
    adr, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
    verts = model.mesh_vert[adr:adr + n]
    R = data.geom_xmat[gid].reshape(3, 3)
    w = verts @ R.T + data.geom_xpos[gid]
    center_xy = (w[:, :2].min(0) + w[:, :2].max(0)) / 2
    print("body origin    :", data.xpos[model.geom_bodyid[gid]])
    print("base z / top z :", w[:, 2].min(), w[:, 2].max())
    print("axis center xy :", center_xy)
    print("radius         :", (w[:, :2].max(0) - w[:, :2].min(0)).max() / 2)
    print("height         :", w[:, 2].max() - w[:, 2].min())
    return center_xy, w[:, 2].min()

def run_operation(model, data, viewer, renderer):
    print("Operation connected to scene.")
    bottle_truth(model, data)    
    wrist_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")
    ik = PandaIK(model, data)

    phase = "scan"
    step_count = 0
    start_q = None
    approach_steps = 0
    APPROACH_STEPS = 1500     # about 3 s of simulated time (2 ms per step)
    arm_dofs = [model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
            for i in range(1, 8)]
    arm_qpos = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")]
            for i in range(1, 8)]
    SCAN_HOLD_STEPS = 300   # how long to sit at the scan pose before moving on

    # exact actuator values you gave
    SCAN_POSE_CTRL = np.array([2.9, -0.582, 0, 0, 0, 0.963, 0, 255])

    target_q = None
    gripper_ctrl = None

    while viewer.is_running():
        mujoco.mj_step(model, data)
        step_count += 1
        if step_count == 500:   
            true_xy, base_z = bottle_truth(model, data)  
            set_truth(true_xy) 
        renderer.update_scene(data, camera='wrist_camera')
        wrist_frame = renderer.render()
        wrist_frame = np.asarray(wrist_frame)
        wrist_frame_bgr = cv2.cvtColor(wrist_frame, cv2.COLOR_RGB2BGR)

        if phase == "scan":
            boxes, labels, scores, mask, track_ids, class_names = segmentor.segment(frame=wrist_frame_bgr)
            wrist_frame_bgr = Draw_Segmentation(wrist_frame_bgr, boxes, labels, scores, mask, track_ids, class_names, model, data)
        display_frame = cv2.cvtColor(wrist_frame_bgr, cv2.COLOR_BGR2RGB)

        is_wrist = (
            viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
            and viewer.cam.fixedcamid == wrist_camera_id
        )
        if is_wrist:
            vp = viewer.viewport
            resized_frame = cv2.resize(display_frame, (vp.width, vp.height))
            viewport_rect = mujoco.MjrRect(vp.left, vp.bottom, vp.width, vp.height)
            viewer.set_images((viewport_rect, resized_frame))
        else:
            viewer.clear_images()

        # # --------------------------------------------------
        # # PHASE LOGIC
        # # --------------------------------------------------
        # if phase == "scan":
        #     data.ctrl[:] = SCAN_POSE_CTRL
        #     joint_err = np.abs(data.qpos[arm_qpos] - SCAN_POSE_CTRL[:7])
        #     joint_err[3] = 0          # joint 4 sits at its limit, ignore it

        #     if joint_err.max() < 0.05:
        #         settle_counter += 1
        #     else:
        #         settle_counter = 0
        #     if step_count % 200 == 0:
        #         err = data.qpos[arm_qpos] - SCAN_POSE_CTRL[:7]
        #         print(f"t={data.time:.2f}s")
        #         print("  joint_err:", np.round(err, 3))
        #         print("  joint_vel:", np.round(data.qvel[arm_dofs], 2))

        #     if settle_counter >= SCAN_HOLD_STEPS:
        #         bottles = disable_detection_and_get_bottles()
        #         print(f"Scan complete. {len(bottles)} bottles registered.")

        #         if len(bottles) > 0:
        #             approach_q, approach_position = approach_bottle(
        #                 0,
        #                 model,
        #                 data,
        #                 ik
        #             )

        #             # ============================================================
        #             # DEBUG: CHECK WHAT IK ACTUALLY FOUND
        #             # ============================================================

        #             saved_qpos = data.qpos[:7].copy()

        #             # Temporarily apply IK solution
        #             data.qpos[:7] = approach_q
        #             mujoco.mj_forward(model, data)

        #             left_pos = data.xpos[ik.left_finger_id].copy()
        #             right_pos = data.xpos[ik.right_finger_id].copy()

        #             ik_gripper_center = (
        #                 left_pos + right_pos
        #             ) / 2.0

        #             ik_error = (
        #                 ik_gripper_center - approach_position
        #             )

        #             print("\n==============================")
        #             print("IK APPROACH FEASIBILITY")
        #             print("==============================")

        #             print("Requested target:")
        #             print(approach_position)

        #             print("\nIK finger center:")
        #             print(ik_gripper_center)

        #             print("\nIK position error:")
        #             print(ik_error)

        #             print("\nIK distance:")
        #             print(
        #                 np.linalg.norm(ik_error) * 1000,
        #                 "mm"
        #             )

        #             print("\nIK joint target:")
        #             print(approach_q)

        #             print("==============================")

        #             # IMPORTANT: restore the real robot state
        #             data.qpos[:7] = saved_qpos
        #             mujoco.mj_forward(model, data)

        #             # ============================================================
        #             # CONTINUE NORMAL APPROACH
        #             # ============================================================

        #             target_q = approach_q
        #             start_q = data.qpos[arm_qpos].copy()
        #             approach_steps = 0

        #             print("Approaching bottle_0 at:", approach_position)

        #             phase = "approach"

        #         else:
        #             print("No bottles found — staying idle.")
        #             phase = "idle"

        # elif phase == "approach":
        #     approach_steps += 1
        #     drive_to_target(model, data, target_q)      # straight to the target, like the scan

        #     if approach_steps == 500:                   # report once, after it has settled
        #         center = (data.xpos[ik.left_finger_id] + data.xpos[ik.right_finger_id]) / 2
        #         print("Gripper center:", np.round(center, 3))
        #         print("Target        :", np.round(approach_position, 3))
        #         print(f"Error: {np.linalg.norm(center - approach_position)*1000:.1f} mm")

        viewer.sync()