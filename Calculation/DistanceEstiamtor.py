import math

class DistanceEstimator:
    def __init__(self,cam_height_meters,tilt_angle_degrees,focal_length_pixels,optical_center_y):
        self.H = cam_height_meters
        self.alpha_0 = tilt_angle_degrees
        self.f_y = focal_length_pixels
        self.c_y = optical_center_y

    def get_distance_to_base(self,bbox_bottom_y):
        """
        Calculate the distance to the base of an object (where it touches the ground)
        bbox_bottom_y: the box lower y coordinate of the bounding box (ymax)
        """

        # Calculate the angle of the pixel relative to the camera center
        alpha_i = math.atan((bbox_bottom_y - self.c_y) / self.f_y)

        # Calculate the total angle from the ground point
        total_angle = self.alpha_0 + alpha_i

        # Avoid division by 0 if looking at the horizon
        if total_angle <= 0:
            return float("inf")

        # Distance formula: Z = H / tan(total_angle)
        distance_meters = abs(self.H / math.tan(total_angle))
        return round(distance_meters,2)


"""
================================================================================
PIPELINE GEOMETRY NOTE: MONOCULAR DEPTH VS. DIAGONAL DISTANCE
================================================================================

1. WHAT THIS CODE MEASURES:
   - This script calculates PERPENDICULAR DEPTH DISTANCE (Z-axis forward gap).
   - It measures how far an object has advanced along the camera's forward heading.
   - It matches how flat camera sensors physically slice 3D space into 2D planes.

2. THE "LEFT/RIGHT" PHENOMENON (e.g., Person 2 at 7 meters forward, 2 meters left):
   - Real World & Simulation: If a person stands off to the side, their feet still 
     rest on the same horizontal line as someone standing directly front-and-center.
   - Sensor Output: Both people share the same pixel row (y2). The math naturally
     outputs the forward depth plane (7.00m), NOT the diagonal line (7.28m).
   - Human eyes think in diagonals, but flat camera sensors calculate in depth planes.

3. CALIBRATION SUMMARY:
   - Keep 'tilt_angle = 0.0' fixed unless the physical camera tilts up/down.
   - Apply 'y2 - 9' to cleanly cancel out bounding box outline thickness and shadows.
================================================================================
"""

"""
================================================================================
PIPELINE GEOMETRY NOTE: CAMERA ROTATION VS. SENSOR DEPTH
================================================================================

1. TURNING THE CAMERA DIRECTION (Pivoting from the same spot):
   - If the camera rotates left to look directly at Person 2, the flat depth 
     planes pivot alongside the camera lens.
   - Person 2 moves into the center of the frame, and the calculation updates 
     from the perpendicular depth (7.00m) to the true diagonal gap (7.28m).

2. IN A LIVE ROBOT PIPELINE:
   - When the camera turns, the physical distance between your feet and their feet 
     does not change, but the value calculated by a flat 2D sensor DOES change.
   - To keep calculations perfectly stable when panning, the pipeline must read 
     the camera's gyroscope/rotation angle to adjust the geometry grid.
================================================================================
"""