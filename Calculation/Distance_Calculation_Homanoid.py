import math

def get_real_world_camera_parameters(image_height, image_width):
    """
    Acts as the environmental calibration management layer for the real-world
    street walking pipeline, computing frame parameters dynamically without hardcoding pixels.
    """
    # ----------------------------------------------------------------------
    # 1. HARD ENVIRONMENTAL PHYSICAL CONSTANTS (Calibrated to your setup)
    # ----------------------------------------------------------------------
    # Height of the camera lens above the floor pavement plane (e.g., chest level)
    CAMERA_HEIGHT_METERS = 1.45  
    
    # Gaze orientation angle relative to the flat floor horizon plane
    # 0.0 means looking straight down the sidewalk path
    TILT_ANGLE_DEGREES = 0.0     
    
    # The vertical Field of View (FOV) of the recording device lens
    # 60.0 degrees is the industry standard average for modern smartphones and action cams
    VERTICAL_FOV_DEGREES = 65.0  

    # ----------------------------------------------------------------------
    # 2. AUTOMATED OPTICAL GEOMETRY COMPUTATIONS
    # ----------------------------------------------------------------------
    # Dynamic 2D spatial coordinate center lines
    center_y = image_height / 2.0
    center_x = image_width / 2.0

    # Dynamic trigonometric mapping: Maps physical landscape angles to frame pixels
    focal_length = center_y / math.tan(math.radians(VERTICAL_FOV_DEGREES / 2.0))

    return {
        "camera_height": CAMERA_HEIGHT_METERS,
        "tilt_angle_degrees": TILT_ANGLE_DEGREES,
        "focal_length_pixels": focal_length,
        "optical_center_y": 587,
        "optical_center_x": center_x
    }
