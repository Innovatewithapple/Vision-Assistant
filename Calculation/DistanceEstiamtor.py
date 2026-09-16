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