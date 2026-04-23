"""Testing the performance of OSCBF in highly-constrained settings

We consider a cluttered tabletop environment with many randomized obstacles,
each represented as a sphere. We then enforce collision avoidance with
all of the obstacles, and all of the collision bodies on the robot

There are likely "smarter" ways to filter out the collision pairs that are
least likely to cause a collision, but for now, this test just tries to see
how much we can scale up the collision avoidance while retaining real-time
performance.
"""

import argparse

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv, FrankaVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.core.controllers import PoseTaskTorqueController, PoseTaskVelocityController


np.random.seed(0)


@jax.tree_util.register_static
class CollisionsConfig(OSCBFTorqueConfig):
    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole body table avoidance
        h_table = (
            robot_collision_positions[:, 2] - self.z_min - robot_collision_radii.ravel()
        )

        return jnp.concatenate([h_collision, h_table])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


@jax.tree_util.register_static
class CollisionsVelocityConfig(OSCBFVelocityConfig):
    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole body table avoidance
        h_table = (
            robot_collision_positions[:, 2] - self.z_min - robot_collision_radii.ravel()
        )

        return jnp.concatenate([h_collision, h_table])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)
    # Set nullspace desired joint position
    nullspace_posture_goal = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
            0.0,
        ]
    )

    # Compute nominal control
    u_nom = osc_controller(
        q,
        qdot,
        pos=ee_tmat[:3, 3],
        rot=ee_tmat[:3, :3],
        des_pos=z_ee_des[:3],
        des_rot=jnp.reshape(z_ee_des[3:12], (3, 3)),
        des_vel=z_ee_des[12:15],
        des_omega=z_ee_des[15:18],
        des_accel=jnp.zeros(3),
        des_alpha=jnp.zeros(3),
        des_q=nullspace_posture_goal,
        des_qdot=jnp.zeros(robot.num_joints),
        J=J,
        M=M,
        M_inv=M_inv,
        g=g,
        c=c,
    )
    # Apply the CBF safety filter
    return cbf.safety_filter(z, u_nom)


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_velocity_control(
    robot: Manipulator,
    osc_controller: PoseTaskVelocityController,
    cbf: CBF,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    M_inv, J, ee_tmat = robot.dynamically_consistent_velocity_control_matrices(q)
    pos = ee_tmat[:3, 3]
    rot = ee_tmat[:3, :3]
    des_pos = z_ee_des[:3]
    des_rot = jnp.reshape(z_ee_des[3:12], (3, 3))
    des_vel = z_ee_des[12:15]
    des_omega = z_ee_des[15:18]
    # Set nullspace desired joint position
    des_q = jnp.array(
        [
            0.0,
            -jnp.pi / 6,
            0.0,
            -3 * jnp.pi / 4,
            0.0,
            5 * jnp.pi / 9,
            0.0,
        ]
    )
    u_nom = osc_controller(
        q, pos, rot, des_pos, des_rot, des_vel, des_omega, des_q, J, M_inv
    )
    return cbf.safety_filter(q, u_nom)


def main(control_method="torque", num_bodies=25):
    assert control_method in ["torque", "velocity"]

    robot = load_panda()
    z_min = 0.1

    max_num_bodies = 50

    # Sample a lot of collision bodies
    all_collision_pos = np.random.uniform(
        low=[0.2, -0.4, 0.1], high=[0.8, 0.4, 0.3], size=(max_num_bodies, 3)
    )
    all_collision_radii = np.random.uniform(low=0.01, high=0.1, size=(max_num_bodies,))
    # Only use a subset of them based on the desired quantity
    collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
    collision_radii = all_collision_radii[:num_bodies]
    collision_data = {"positions": collision_pos, "radii": collision_radii}

    torque_config = CollisionsConfig(robot, z_min, collision_pos, collision_radii)
    torque_cbf = CBF.from_config(torque_config)
    velocity_config = CollisionsVelocityConfig(
        robot, z_min, collision_pos, collision_radii
    )
    velocity_cbf = CBF.from_config(velocity_config)

    timestep = 1 / 240  #  1 / 1000
    bg_color = (1, 1, 1)
    if control_method == "torque":
        env = FrankaTorqueControlEnv(
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
            collision_data=collision_data,
            load_table=True,
        )
    else:
        env = FrankaVelocityControlEnv(
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
            collision_data=collision_data,
            load_table=True,
        )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.40,
        cameraYaw=104.40,
        cameraPitch=-37,
        cameraTargetPosition=(0.20, 0.07, -0.09),
    )

    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0
    osc_torque_controller = PoseTaskTorqueController(
        n_joints=robot.num_joints,
        kp_task=np.concatenate([kp_pos * np.ones(3), kp_rot * np.ones(3)]),
        kd_task=np.concatenate([kd_pos * np.ones(3), kd_rot * np.ones(3)]),
        kp_joint=kp_joint,
        kd_joint=kd_joint,
        # Note: torque limits will be enforced via the QP. We'll set them to None here
        # because we don't want to clip the values before the QP
        tau_min=None,
        tau_max=None,
    )

    osc_velocity_controller = PoseTaskVelocityController(
        n_joints=robot.num_joints,
        kp_task=np.array([kp_pos, kp_pos, kp_pos, kp_rot, kp_rot, kp_rot]),
        kp_joint=kp_joint,
        # Note: velocity limits will be enforced via the QP
        qdot_min=None,
        qdot_max=None,
    )

    @jax.jit
    def compute_torque_control_jit(z, z_ee_des):
        return compute_torque_control(
            robot, osc_torque_controller, torque_cbf, z, z_ee_des
        )

    @jax.jit
    def compute_velocity_control_jit(z, z_ee_des):
        return compute_velocity_control(
            robot, osc_velocity_controller, velocity_cbf, z, z_ee_des
        )

    if control_method == "torque":
        compute_control = compute_torque_control_jit
    elif control_method == "velocity":
        compute_control = compute_velocity_control_jit
    else:
        raise ValueError(f"Invalid control method: {control_method}")

    while True:
        q_qdot = env.get_joint_state()
        z_zdot_ee_des = env.get_desired_ee_state()
        tau = compute_control(q_qdot, z_zdot_ee_des)
        env.apply_control(tau)
        env.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run highly-constrained collision avoidance experiment."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        choices=["torque", "velocity"],
        default="torque",
        help="Control method to use (default: torque)",
    )
    parser.add_argument(
        "--num_bodies",
        type=int,
        default=25,
        help="Number of collision bodies to simulate (default: 25)",
    )
    args = parser.parse_args()
    main(control_method=args.control_method, num_bodies=args.num_bodies)
