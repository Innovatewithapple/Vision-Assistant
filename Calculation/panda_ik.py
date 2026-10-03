import mujoco
import numpy as np
from requests.packages import target

class PandaIK:
    def __init__(self,model,data,hand_body_name='hand'):
        self.model = model
        self.data = data
        self.hand_body_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,hand_body_name)
        self.arm_dofs = 7
        self.link7_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "link7"
        )

        # Finger bodies
        self.left_finger_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "left_finger"
        )

        self.right_finger_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            "right_finger"
        )

        self.arm_dofs = 7

    def solve(self, target_position, iterations=2000):
        target_position = np.array(target_position, dtype=float)

        # --------------------------------------------------
        # Desired hand orientation
        # Only care about the hand Z axis pointing downward.
        # This means 0 degree tilt.
        # --------------------------------------------------
        desired_hand_z = np.array([0.0, 0.0, -1.0])

        # Start from current robot configuration
        q = self.data.qpos[:self.arm_dofs].copy()

        for iteration in range(iterations):

            # ----------------------------------------------
            # Put temporary IK configuration into MuJoCo
            # ----------------------------------------------
            self.data.qpos[:self.arm_dofs] = q
            mujoco.mj_forward(self.model, self.data)

            # ==================================================
            # 1. POSITION
            # ==================================================

            # Current finger positions
            left_finger_position = (
                self.data.xpos[self.left_finger_id].copy()
            )

            right_finger_position = (
                self.data.xpos[self.right_finger_id].copy()
            )

            # Finger-based gripper center
            current_position = (
                left_finger_position + right_finger_position
            ) / 2.0

            # Position error
            position_error = (
                target_position - current_position
            )

            # ----------------------------------------------
            # Left finger Jacobian
            # ----------------------------------------------
            jacobian_position_left = np.zeros(
                (3, self.model.nv)
            )

            jacobian_rotation_left = np.zeros(
                (3, self.model.nv)
            )

            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian_position_left,
                jacobian_rotation_left,
                left_finger_position,
                self.left_finger_id
            )

            # ----------------------------------------------
            # Right finger Jacobian
            # ----------------------------------------------
            jacobian_position_right = np.zeros(
                (3, self.model.nv)
            )

            jacobian_rotation_right = np.zeros(
                (3, self.model.nv)
            )

            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian_position_right,
                jacobian_rotation_right,
                right_finger_position,
                self.right_finger_id
            )

            # ----------------------------------------------
            # Finger-center position Jacobian
            # ----------------------------------------------
            jacobian_position = (
                jacobian_position_left
                + jacobian_position_right
            ) / 2.0

            J_position = (
                jacobian_position[:, :self.arm_dofs]
            )

            # ==================================================
            # 2. ORIENTATION / TILT
            # ==================================================

            # Current hand rotation matrix
            hand_rotation = self.data.xmat[
                self.hand_body_id
            ].reshape(3, 3)

            # Hand local Z axis expressed in world coordinates
            current_hand_z = hand_rotation[:, 2].copy()

            # Orientation error:
            # We want current_hand_z -> desired_hand_z
            orientation_error = (
                desired_hand_z - current_hand_z
            )

            # ----------------------------------------------
            # Hand rotational Jacobian
            # ----------------------------------------------
            hand_position = self.data.xpos[
                self.hand_body_id
            ].copy()

            jacobian_position_hand = np.zeros(
                (3, self.model.nv)
            )

            jacobian_rotation_hand = np.zeros(
                (3, self.model.nv)
            )

            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian_position_hand,
                jacobian_rotation_hand,
                hand_position,
                self.hand_body_id
            )

            # Only Panda arm joints
            J_rotation = (
                jacobian_rotation_hand[:, :self.arm_dofs]
            )

            # ----------------------------------------------
            # Convert angular Jacobian into a Jacobian
            # for the HAND Z AXIS.
            #
            # dz = omega x z
            # ----------------------------------------------
            z = current_hand_z

            skew_z = np.array([
                [0.0,   -z[2],  z[1]],
                [z[2],   0.0,  -z[0]],
                [-z[1], z[0],   0.0]
            ])

            J_hand_z = -skew_z @ J_rotation

            # ==================================================
            # 3. COMBINE POSITION + ORIENTATION
            # ==================================================

            # Position is in meters.
            # Orientation error is dimensionless.
            #
            # Increase this if orientation needs to have
            # stronger influence.
            position_weight = 1.0
            orientation_weight = 1.0

            J_combined = np.vstack([
                position_weight * J_position,
                orientation_weight * J_hand_z
            ])

            error_combined = np.concatenate([
                position_weight * position_error,
                orientation_weight * orientation_error
            ])

            # ==================================================
            # 4. DAMPED LEAST-SQUARES IK
            # ==================================================

            damping = 0.05

            dq = J_combined.T @ np.linalg.solve(
                J_combined @ J_combined.T
                + damping ** 2 * np.eye(6),
                error_combined
            )

            # Small IK step
            step_size = 0.05

            q += step_size * dq

            # ==================================================
            # 5. RESPECT PANDA JOINT LIMITS
            # ==================================================

            for joint_index in range(self.arm_dofs):

                lower = self.model.jnt_range[
                    joint_index, 0
                ]

                upper = self.model.jnt_range[
                    joint_index, 1
                ]

                q[joint_index] = np.clip(
                    q[joint_index],
                    lower,
                    upper
                )

            # ==================================================
            # 6. OPTIONAL CONVERGENCE CHECK
            # ==================================================

            position_error_norm = np.linalg.norm(
                position_error
            )

            tilt_error = np.linalg.norm(
                orientation_error
            )

            if (
                position_error_norm < 0.001
                and tilt_error < 0.00001
            ):
                break

        # ======================================================
        # KEEP FINAL IK CONFIGURATION
        # ======================================================

        self.data.qpos[:self.arm_dofs] = q
        mujoco.mj_forward(self.model, self.data)

        # ======================================================
        # FINAL POSITION CHECK
        # ======================================================

        final_left = self.data.xpos[
            self.left_finger_id
        ].copy()

        final_right = self.data.xpos[
            self.right_finger_id
        ].copy()

        final_position = (
            final_left + final_right
        ) / 2.0

        final_position_error = (
            target_position - final_position
        )

        final_position_error_norm = np.linalg.norm(
            final_position_error
        )

        # ======================================================
        # FINAL ORIENTATION CHECK
        # ======================================================

        final_hand_rotation = self.data.xmat[
            self.hand_body_id
        ].reshape(3, 3)

        final_hand_z = final_hand_rotation[:, 2]

        cos_angle = np.clip(
            np.dot(
                final_hand_z,
                desired_hand_z
            ),
            -1.0,
            1.0
        )

        final_tilt_angle = np.degrees(
            np.arccos(cos_angle)
        )

        # ======================================================
        # PRINT RESULT
        # ======================================================

        # print("\n==============================")
        # print("IK FINAL RESULT")
        # print("==============================")

        # print("Iterations used:", iteration + 1)

        # print("\nPOSITION")
        # print("Target position:", target_position)
        # print("Final gripper center:", final_position)
        # print("Final position error:",
        #     final_position_error)
        # print("Final position error norm:",
        #     final_position_error_norm)

        # print("\nORIENTATION")
        # print("Desired hand Z:", desired_hand_z)
        # print("Final hand Z:", final_hand_z)
        # print(
        #     f"Final tilt from vertical: "
        #     f"{final_tilt_angle:.6f} degrees"
        # )

        # print("\nSTATUS")
        # print(
        #     "Position converged:",
        #     final_position_error_norm < 0.001
        # )

        # print(
        #     "Orientation converged:",
        #     final_tilt_angle < 0.1
        # )

        # print("==============================")

        return q


import numpy as np

def drive_to_target(model, data, target_q, gripper_ctrl=None):
    """
    Call this once per simulation step. Pushes data.ctrl toward target_q
    for the 7 arm joints (with gravity compensation), and optionally
    sets the gripper's ctrl value too.

    target_q: array of 7 joint angles (from ik.solve())
    gripper_ctrl: optional scalar — gripper actuator value (e.g. 0=open, 255=closed).
                  Pass None to leave the gripper untouched this step.
    """
    kp = model.actuator_gainprm[:7, 0]

    corrected_ctrl = (
        data.qfrc_bias[:7] / kp
        + target_q
    )

    data.ctrl[:7] = corrected_ctrl

    if gripper_ctrl is not None:
        data.ctrl[7] = gripper_ctrl