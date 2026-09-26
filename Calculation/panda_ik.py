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

    def solve(self, target_position, iterations=1000):
        target_position = np.array(target_position,dtype=float)

        # Save the actual robot state
        original_qpos = self.data.qpos.copy()

        #--Start from the current robot configuration--!
        q = self.data.qpos[:self.arm_dofs].copy()

        for iteration in range(iterations):
            #--Put temporary configuration into MUJOCO--!
            self.data.qpos[:self.arm_dofs] = q

            mujoco.mj_forward(self.model,self.data)

            #--Current hand position--!
            current_position = self.data.xpos[self.hand_body_id].copy()

            #--Position Error--!
            error = target_position - current_position

            if (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}: "
                    f"error = {np.linalg.norm(error):.6f} m"
                )

            #--Close enough--!
            if np.linalg.norm(error) < 0.001:
                print(f"\n Reached close enough | error = {np.linalg.norm(error):.6f} m")
                break

            #--Position Jacobian--!
            jacobian_position = np.zeros((3,self.model.nv))

            #--Rotation Jacobian--!
            jacobian_rotation = np.zeros((3,self.model.nv))

            #--Now we calculate the affection by changing values--! (if i do this change how much it affect the joint or hand)
            mujoco.mj_jac(
                self.model,
                self.data,
                jacobian_position,
                jacobian_rotation,
                current_position,
                self.hand_body_id
            )
            #--Only Panda Arm Joint--!
            J = jacobian_position[:,:self.arm_dofs]

            #--Damped least-squares IK--!
            damping = 0.05

            dq = J.T @ np.linalg.solve(
                J @ J.T + damping ** 2 * np.eye(3), error
            )
            if iteration == 0:

                print("\n==============================")
                print("FIRST IK ITERATION")
                print("==============================")

                print("\nERROR:")
                print(error)

                print("\nJACOBIAN:")
                print(J)

                print("\nDQ:")
                print(dq)
            #--Small Step--!
            step_size = 0.05

            q += step_size * dq

            #--Respect panda joint limit--!
            for joint_index in range(self.arm_dofs):
                lower = self.model.jnt_range[joint_index,0]
                upper = self.model.jnt_range[joint_index,1]

                q[joint_index] = np.clip(q[joint_index],lower,upper)

        # #--Restore final configuration--!
        # # self.data.qpos[:self.arm_dofs] = q
        # self.data.qpos[:] = original_qpos
        # mujoco.mj_forward(self.model,self.data)

        # return q

        # --Keep final IK configuration--!
        self.data.qpos[:self.arm_dofs] = q
        mujoco.mj_forward(self.model, self.data)

        # --Verify actual final IK result--!
        final_position = self.data.xpos[self.hand_body_id].copy()
        final_error = target_position - final_position
        final_error_norm = np.linalg.norm(final_error)

        print("\n==============================")
        print("IK FINAL RESULT")
        print("==============================")
        print("Iterations used:", iteration + 1)
        print("Target position:", target_position)
        print("Final hand position:", final_position)
        print("Final error:", final_error)
        print("Final error norm:", final_error_norm)
        print("Tolerance:", 0.001)
        print("Converged:", final_error_norm < 0.001)
        print("==============================")

        return q

