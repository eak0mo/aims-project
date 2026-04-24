"""
# Operational Space Control

Using torque control as well as velocity control to map between the Operational (Task / End-Effector)
Space and the Joint Space

These controllers are written to be independent of whatever software is used to get the robot's
state / kinematics / dynamics. For instance, we need to know the following details at every timestep:

- Joint positions (and velocities, for torque control)
- End-effector position and orientation
- Dynamics matrices (for torque control), i.e. mass matrix, gravity vector, and Coriolis forces

These values can be obtained from simulation, a real robot, or other robot kinematics + dynamics packages.

For simplicity, we assume that the robot is operating in 3D and that the task is to control the
position and orientation of the end-effector (6D task). If a different task jacobian is desired,
this will require some slight modifications to the code (likely, to include a task selection matrix,
which depends on your preferred position/orientation representation).
"""

from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


@jax.tree_util.register_static
class PoseTaskTorqueController:
    """Operational Space Torque Controller for 6D position and orientation tasks

    Args:
        n_joints (int): Number of joints, e.g. 7 for 7-DOF robot
        kp_task (ArrayLike): Task-space proportional gain(s), shape (6,)
        kd_task (ArrayLike): Task-space derivative gain(s), shape (6,)
        kp_joint (ArrayLike): Joint-space proportional gain(s), shape (n_joints,)
        kd_joint (ArrayLike): Joint-space derivative gain(s), shape (n_joints,)
        tau_min (Optional[ArrayLike]): Minimum joint torques, shape (n_joints,)
        tau_max (Optional[ArrayLike]): Maximum joint torques, shape (n_joints,)
    """

    def __init__(
        self,
        n_joints: int,
        kp_task: ArrayLike,
        kd_task: ArrayLike,
        kp_joint: ArrayLike,
        kd_joint: ArrayLike,
        tau_min: Optional[ArrayLike],
        tau_max: Optional[ArrayLike],
    ):
        self.dim_space = 3  # 3D
        self.dim_task = 6  # Position and orientation in 3D
        assert isinstance(n_joints, int)
        self.n_joints = n_joints
        self.is_redundant = self.n_joints > self.dim_task
        self.kp_task = _format_gain(kp_task, self.dim_task)
        self.kd_task = _format_gain(kd_task, self.dim_task)
        self.kp_joint = _format_gain(kp_joint, self.n_joints)
        self.kd_joint = _format_gain(kd_joint, self.n_joints)
        self.tau_min = _format_limit(tau_min, self.n_joints, "lower")
        self.tau_max = _format_limit(tau_max, self.n_joints, "upper")

    def __call__(
        self,
        q: ArrayLike,
        qdot: ArrayLike,
        pos: ArrayLike,
        rot: ArrayLike,
        des_pos: ArrayLike,
        des_rot: ArrayLike,
        des_vel: ArrayLike,
        des_omega: ArrayLike,
        des_accel: ArrayLike,
        des_alpha: ArrayLike,
        des_q: ArrayLike,
        des_qdot: ArrayLike,
        J: ArrayLike,
        M: ArrayLike,
        M_inv: ArrayLike,
        g: ArrayLike,
        c: ArrayLike,
    ) -> Array:
        """Compute joint torques for operational space control

        Args:
            q (ArrayLike): Current joint positions, shape (n_joints,)
            qdot (ArrayLike): Current joint velocities, shape (n_joints,)
            pos (ArrayLike): Current end-effector position, shape (3,)
            rot (ArrayLike): Current end-effector rotation matrix, shape (3, 3)
            des_pos (ArrayLike): Desired end-effector position, shape (3,)
            des_rot (ArrayLike): Desired end-effector rotation matrix, shape (3, 3)
            des_vel (ArrayLike): Desired end-effector velocity, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_omega (ArrayLike): Desired end-effector angular velocity, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_accel (ArrayLike): Desired end-effector acceleration, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_alpha (ArrayLike): Desired end-effector angular acceleration, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_q (ArrayLike): Desired joint positions, shape (n_joints,).
                This is used as a nullspace posture task.
                If this is not required for the task, set it to the current joint positions (q)
            des_qdot (ArrayLike): Desired joint velocities, shape (n_joints,).
                This is used for a nullspace joint damping term.
                If this is not required for the task, set it to a zero vector
            J (ArrayLike): Basic Jacobian (mapping joint velocities to task velocities), shape (6, n_joints)
            M (ArrayLike): Mass matrix, shape (n_joints, n_joints)
            M_inv (ArrayLike): Inverse of the mass matrix, shape (n_joints, n_joints)
            g (ArrayLike): Gravity vector, shape (n_joints,)
            c (ArrayLike): Centrifugal/Coriolis vector, shape (n_joints,).
                This term is often neglected, and can be set to a zero vector if so.

        Returns:
            Array: Joint torques, shape (n_joints,)
        """
        # Check shapes
        assert q.shape == (self.n_joints,)
        assert qdot.shape == (self.n_joints,)
        assert pos.shape == (self.dim_space,)
        assert rot.shape == (self.dim_space, self.dim_space)
        assert des_pos.shape == (self.dim_space,)
        assert des_rot.shape == (self.dim_space, self.dim_space)
        assert des_vel.shape == (self.dim_space,)
        assert des_omega.shape == (self.dim_space,)
        assert des_accel.shape == (self.dim_space,)
        assert des_alpha.shape == (self.dim_space,)
        assert des_q.shape == (self.n_joints,)
        assert des_qdot.shape == (self.n_joints,)
        assert J.shape == (self.dim_task, self.n_joints)
        assert M.shape == (self.n_joints, self.n_joints)
        assert M_inv.shape == (self.n_joints, self.n_joints)
        assert g.shape == (self.n_joints,)
        assert c.shape == (self.n_joints,)

        # Compute twist
        twist = J @ qdot
        vel = twist[: self.dim_space]
        omega = twist[self.dim_space :]

        # Errors
        pos_error = pos - des_pos
        vel_error = vel - des_vel
        rot_error = orientation_error_3D(rot, des_rot)
        omega_error = omega - des_omega
        task_p_error = jnp.concatenate([pos_error, rot_error])
        task_d_error = jnp.concatenate([vel_error, omega_error])

        # Operational space matrices
        task_inertia_inv = J @ M_inv @ J.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ J.T @ task_inertia

        # Compute operational space task torques
        task_accel = (
            jnp.concatenate([des_accel, des_alpha])
            - self.kp_task * task_p_error
            - self.kd_task * task_d_error
        )
        task_wrench = task_inertia @ task_accel
        tau = J.T @ task_wrench

        # Add compensation for nonlinear effects
        tau += g + c

        if self.is_redundant:
            # Nullspace projection
            NT = jnp.eye(self.n_joints) - J.T @ J_bar.T
            # Add nullspace joint task
            q_error = q - des_q
            qdot_error = qdot - des_qdot
            joint_accel = -self.kp_joint * q_error - self.kd_joint * qdot_error
            secondary_joint_torques = M @ joint_accel
            tau += NT @ secondary_joint_torques

        # Clamp to torque limits
        return jnp.clip(tau, self.tau_min, self.tau_max)


@jax.tree_util.register_static
class PoseTaskVelocityController:
    """Operational Space Velocity Controller for 6D position and orientation tasks

    Args:
        n_joints (int): Number of joints, e.g. 7 for 7-DOF robot
        kp_task (ArrayLike): Task-space proportional gain(s), shape (6,)
        kp_joint (ArrayLike): Joint-space proportional gain(s), shape (n_joints,)
        qdot_min (Optional[ArrayLike]): Minimum joint velocities, shape (n_joints,)
        qdot_max (Optional[ArrayLike]): Maximum joint velocities, shape (n_joints,)
    """

    def __init__(
        self,
        n_joints: int,
        kp_task: ArrayLike,
        kp_joint: ArrayLike,
        qdot_min: Optional[ArrayLike],
        qdot_max: Optional[ArrayLike],
    ):
        self.dim_space = 3  # 3D
        self.dim_task = 6  # Position and orientation in 3D
        assert isinstance(n_joints, int)
        self.n_joints = n_joints
        self.is_redundant = self.n_joints > self.dim_task
        self.kp_task = _format_gain(kp_task, self.dim_task)
        self.kp_joint = _format_gain(kp_joint, self.n_joints)
        self.qdot_min = _format_limit(qdot_min, self.n_joints, "lower")
        self.qdot_max = _format_limit(qdot_max, self.n_joints, "upper")

    @jax.jit
    def __call__(
        self,
        q: ArrayLike,
        pos: ArrayLike,
        rot: ArrayLike,
        des_pos: ArrayLike,
        des_rot: ArrayLike,
        des_vel: ArrayLike,
        des_omega: ArrayLike,
        des_q: ArrayLike,
        J: ArrayLike,
        M_inv: Optional[ArrayLike] = None,
    ) -> Array:
        """Compute joint velocities for operational space control

        Args:
            q (ArrayLike): Current joint positions, shape (n_joints,)
            pos (ArrayLike): Current end-effector position, shape (3,)
            rot (ArrayLike): Current end-effector rotation matrix, shape (3, 3)
            des_pos (ArrayLike): Desired end-effector position, shape (3,)
            des_rot (ArrayLike): Desired end-effector rotation matrix, shape (3, 3)
            des_vel (ArrayLike): Desired end-effector velocity, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_omega (ArrayLike): Desired end-effector angular velocity, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_q (ArrayLike): Desired joint positions, shape (n_joints,).
                This is used as a nullspace posture task.
                If this is not required for the task, set it to the current joint positions (q)
            J (ArrayLike): Basic Jacobian (mapping joint velocities to task velocities), shape (6, n_joints)
            M_inv (Optional[ArrayLike]): Inverse of the mass matrix, shape (n_joints, n_joints).
                This is required to use a dynamically-consistent generalized inverse. If not provided,
                the pseudoinverse will be used.

        Returns:
            Array: Joint torques, shape (n_joints,)
        """
        # Check shapes
        assert q.shape == (self.n_joints,)
        assert pos.shape == (self.dim_space,)
        assert rot.shape == (self.dim_space, self.dim_space)
        assert des_pos.shape == (self.dim_space,)
        assert des_rot.shape == (self.dim_space, self.dim_space)
        assert des_vel.shape == (self.dim_space,)
        assert des_omega.shape == (self.dim_space,)
        assert des_q.shape == (self.n_joints,)
        assert J.shape == (self.dim_task, self.n_joints)

        # Errors
        pos_error = pos - des_pos
        rot_error = orientation_error_3D(rot, des_rot)
        task_p_error = jnp.concatenate([pos_error, rot_error])

        if M_inv is None:
            J_hash = jnp.linalg.pinv(J)  # "J pseudo"
        else:
            task_inertia_inv = J @ M_inv @ J.T
            task_inertia = jnp.linalg.inv(task_inertia_inv)
            J_hash = M_inv @ J.T @ task_inertia  # "J bar"

        # Compute task velocities
        task_vel = jnp.concatenate([des_vel, des_omega]) - self.kp_task * task_p_error
        # Map to joint velocities
        v = J_hash @ task_vel

        if self.is_redundant:
            # Nullspace projection
            N = jnp.eye(self.n_joints) - J_hash @ J
            # Add nullspace joint task
            q_error = q - des_q
            secondary_joint_vel = -self.kp_joint * q_error
            v += N @ secondary_joint_vel

        # Clamp to velocity limits
        return jnp.clip(v, self.qdot_min, self.qdot_max)


@jax.tree_util.register_static
class PositionTaskTorqueController:
    """Operational Space Torque Controller for 3D positional tasks

    Args:
        n_joints (int): Number of joints, e.g. 7 for 7-DOF robot
        kp_task (ArrayLike): Task-space proportional gain(s), shape (3,)
        kd_task (ArrayLike): Task-space derivative gain(s), shape (3,)
        kp_joint (ArrayLike): Joint-space proportional gain(s), shape (n_joints,)
        kd_joint (ArrayLike): Joint-space derivative gain(s), shape (n_joints,)
        tau_min (Optional[ArrayLike]): Minimum joint torques, shape (n_joints,)
        tau_max (Optional[ArrayLike]): Maximum joint torques, shape (n_joints,)
    """

    def __init__(
        self,
        n_joints: int,
        kp_task: ArrayLike,
        kd_task: ArrayLike,
        kp_joint: ArrayLike,
        kd_joint: ArrayLike,
        tau_min: Optional[ArrayLike],
        tau_max: Optional[ArrayLike],
    ):
        self.dim_space = 3  # 3D
        self.dim_task = 3  # Position in 3D
        assert isinstance(n_joints, int)
        self.n_joints = n_joints
        self.is_redundant = self.n_joints > self.dim_task
        self.kp_task = _format_gain(kp_task, self.dim_task)
        self.kd_task = _format_gain(kd_task, self.dim_task)
        self.kp_joint = _format_gain(kp_joint, self.n_joints)
        self.kd_joint = _format_gain(kd_joint, self.n_joints)
        self.tau_min = _format_limit(tau_min, self.n_joints, "lower")
        self.tau_max = _format_limit(tau_max, self.n_joints, "upper")

    @jax.jit
    def __call__(
        self,
        q: ArrayLike,
        qdot: ArrayLike,
        pos: ArrayLike,
        des_pos: ArrayLike,
        des_vel: ArrayLike,
        des_accel: ArrayLike,
        des_q: ArrayLike,
        des_qdot: ArrayLike,
        Jv: ArrayLike,
        M: ArrayLike,
        M_inv: ArrayLike,
        g: ArrayLike,
        c: ArrayLike,
    ) -> Array:
        """Compute joint torques for operational space control

        Args:
            q (ArrayLike): Current joint positions, shape (n_joints,)
            qdot (ArrayLike): Current joint velocities, shape (n_joints,)
            pos (ArrayLike): Current end-effector position, shape (3,)
            des_pos (ArrayLike): Desired end-effector position, shape (3,)
            des_vel (ArrayLike): Desired end-effector velocity, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_accel (ArrayLike): Desired end-effector acceleration, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_q (ArrayLike): Desired joint positions, shape (n_joints,).
                This is used as a nullspace posture task.
                If this is not required for the task, set it to a the current joint positions (q)
            des_qdot (ArrayLike): Desired joint velocities, shape (n_joints,).
                This is used for a nullspace joint damping term.
                If this is not required for the task, set it to a zero vector
            Jv (ArrayLike): Linear Jacobian (mapping joint velocities to task velocities), shape (3, n_joints)
            M (ArrayLike): Mass matrix, shape (n_joints, n_joints)
            M_inv (ArrayLike): Inverse of the mass matrix, shape (n_joints, n_joints)
            g (ArrayLike): Gravity vector, shape (n_joints,)
            c (ArrayLike): Centrifugal/Coriolis vector, shape (n_joints,).
                This term is often neglected, and can be set to a zero vector if so.

        Returns:
            Array: Joint torques, shape (n_joints,)
        """
        # Check shapes
        assert q.shape == (self.n_joints,)
        assert qdot.shape == (self.n_joints,)
        assert pos.shape == (self.dim_space,)
        assert des_pos.shape == (self.dim_space,)
        assert des_vel.shape == (self.dim_space,)
        assert des_accel.shape == (self.dim_space,)
        assert des_q.shape == (self.n_joints,)
        assert des_qdot.shape == (self.n_joints,)
        assert Jv.shape == (self.dim_task, self.n_joints)
        assert M.shape == (self.n_joints, self.n_joints)
        assert M_inv.shape == (self.n_joints, self.n_joints)
        assert g.shape == (self.n_joints,)
        assert c.shape == (self.n_joints,)

        # Compute twist
        vel = Jv @ qdot

        # Errors
        task_p_error = pos - des_pos
        task_d_error = vel - des_vel

        # Operational space matrices
        task_inertia_inv = Jv @ M_inv @ Jv.T
        task_inertia = jnp.linalg.inv(task_inertia_inv)
        J_bar = M_inv @ Jv.T @ task_inertia

        # Compute operational space task torques
        task_accel = (
            des_accel - self.kp_task * task_p_error - self.kd_task * task_d_error
        )
        task_force = task_inertia @ task_accel
        tau = Jv.T @ task_force

        # Add compensation for nonlinear effects
        tau += g + c

        if self.is_redundant:
            # Nullspace projection
            NT = jnp.eye(self.n_joints) - Jv.T @ J_bar.T
            # Add nullspace joint task
            q_error = q - des_q
            qdot_error = qdot - des_qdot
            joint_accel = -self.kp_joint * q_error - self.kd_joint * qdot_error
            secondary_joint_torques = M @ joint_accel
            tau += NT @ secondary_joint_torques

        # Clamp to torque limits
        return jnp.clip(tau, self.tau_min, self.tau_max)


@jax.tree_util.register_static
class PositionTaskVelocityController:
    """Operational Space Velocity Controller for 3D positional tasks

    Args:
        n_joints (int): Number of joints, e.g. 7 for 7-DOF robot
        kp_task (ArrayLike): Task-space proportional gain(s), shape (3,)
        kp_joint (ArrayLike): Joint-space proportional gain(s), shape (n_joints,)
        qdot_min (Optional[ArrayLike]): Minimum joint velocities, shape (n_joints,)
        qdot_max (Optional[ArrayLike]): Maximum joint velocities, shape (n_joints,)
    """

    def __init__(
        self,
        n_joints: int,
        kp_task: ArrayLike,
        kp_joint: ArrayLike,
        qdot_min: Optional[ArrayLike],
        qdot_max: Optional[ArrayLike],
    ):
        self.dim_space = 3  # 3D
        self.dim_task = 3  # Position in 3D
        assert isinstance(n_joints, int)
        self.n_joints = n_joints
        self.is_redundant = self.n_joints > self.dim_task
        self.kp_task = _format_gain(kp_task, self.dim_task)
        self.kp_joint = _format_gain(kp_joint, self.n_joints)
        self.qdot_min = _format_limit(qdot_min, self.n_joints, "lower")
        self.qdot_max = _format_limit(qdot_max, self.n_joints, "upper")

    @jax.jit
    def __call__(
        self,
        q: ArrayLike,
        pos: ArrayLike,
        des_pos: ArrayLike,
        des_vel: ArrayLike,
        des_q: ArrayLike,
        Jv: ArrayLike,
        M_inv: Optional[ArrayLike] = None,
    ) -> Array:
        """Compute joint velocities for operational space control

        Args:
            q (ArrayLike): Current joint positions, shape (n_joints,)
            pos (ArrayLike): Current end-effector position, shape (3,)
            des_pos (ArrayLike): Desired end-effector position, shape (3,)
            des_vel (ArrayLike): Desired end-effector velocity, shape (3,).
                If this is not required for the task, set it to a zero vector
            des_q (ArrayLike): Desired joint positions, shape (n_joints,).
                This is used as a nullspace posture task.
                If this is not required for the task, set it to the current joint positions (q)
            Jv (ArrayLike): Linear Jacobian (mapping joint velocities to task velocities), shape (3, n_joints)
            M_inv (Optional[ArrayLike]): Inverse of the mass matrix, shape (n_joints, n_joints).
                This is required to use a dynamically-consistent generalized inverse. If not provided,
                the pseudoinverse will be used.

        Returns:
            Array: Joint torques, shape (n_joints,)
        """
        # Check shapes
        assert q.shape == (self.n_joints,)
        assert pos.shape == (self.dim_space,)
        assert des_pos.shape == (self.dim_space,)
        assert des_vel.shape == (self.dim_space,)
        assert des_q.shape == (self.n_joints,)
        assert Jv.shape == (self.dim_task, self.n_joints)

        # Errors
        task_p_error = pos - des_pos

        if M_inv is None:
            J_hash = jnp.linalg.pinv(Jv)  # "J pseudo"
        else:
            task_inertia_inv = Jv @ M_inv @ Jv.T
            task_inertia = jnp.linalg.inv(task_inertia_inv)
            J_hash = M_inv @ Jv.T @ task_inertia  # "J bar"

        # Compute task velocities
        task_vel = des_vel - self.kp_task * task_p_error
        # Map to joint velocities
        v = J_hash @ task_vel

        if self.is_redundant:
            # Nullspace projection
            N = jnp.eye(self.n_joints) - J_hash @ Jv
            # Add nullspace joint task
            q_error = q - des_q
            secondary_joint_vel = -self.kp_joint * q_error
            v += N @ secondary_joint_vel

        # Clamp to velocity limits
        return jnp.clip(v, self.qdot_min, self.qdot_max)


# Helper functions


def orientation_error_3D(R_cur: ArrayLike, R_des: ArrayLike) -> ArrayLike:
    """Determine the angular error vector between two rotation matrices in 3D.

    Args:
        R_cur (ArrayLike): Current rotation matrix, shape (3, 3)
        R_des (ArrayLike): Desired rotation matrix, shape (3, 3)

    Returns:
        ArrayLike: Angular error, shape (3,)
    """
    return -0.5 * (
        jnp.cross(R_cur[:, 0], R_des[:, 0])
        + jnp.cross(R_cur[:, 1], R_des[:, 1])
        + jnp.cross(R_cur[:, 2], R_des[:, 2])
    )


def _format_gain(k: float, dim: int) -> ArrayLike:
    if isinstance(k, (float, int)):
        k = k * jnp.ones(dim)
    else:
        k = jnp.atleast_1d(k).flatten()
        assert k.shape == (dim,)
    return k


def _format_limit(arr: Optional[ArrayLike], dim: int, side: str) -> Optional[ArrayLike]:
    if arr is None:
        if side == "upper":
            arr = jnp.inf * jnp.ones(dim)
        elif side == "lower":
            arr = -jnp.inf * jnp.ones(dim)
        else:
            raise ValueError(f"Invalid side: {side}")
    else:
        arr = jnp.atleast_1d(arr).flatten()
        assert arr.shape == (dim,)
    return arr
