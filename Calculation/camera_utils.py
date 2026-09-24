import math
import mujoco

def get_camera_parameters(model, camera_name="main_camera", target_body_name="camera_target"):
    """
    Queries MuJoCo's memory model data directly to extract physical
    camera parameters dynamically for the pipeline estimator.
    """
    # 1. Look up scene IDs from compiled memory structures
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    
    if cam_id == -1:
        raise ValueError(f"Camera name '{camera_name}' not found in model XML configuration.")

    # 2. Extract position metrics directly from arrays
    cam_pos = model.cam_pos[cam_id]
    camera_height = float(cam_pos[2])
    fovy_deg = float(model.cam_fovy[cam_id])

    # 3. Dynamic tilt angle evaluation (For Target tracking cameras)
    if target_body_id != -1:
        target_pos = model.body_pos[target_body_id]
        delta_z = camera_height - target_pos[2]
        delta_y = abs(cam_pos[1] - target_pos[1])
        
        # Calculate pitch and store as a negative value for downward tilt
        calculated_tilt_deg = -math.degrees(math.atan(delta_z / delta_y))
    else:
        # Default fallback if no target body orientation is bound
        calculated_tilt_deg = 0.0

    return {
        "camera_height": camera_height,
        "fovy_degrees": fovy_deg,
        "tilt_angle_degrees": round(calculated_tilt_deg, 2)
    }
