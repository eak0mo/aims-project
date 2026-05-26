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
import pybullet
import sys

import numpy as np
import jax
import pandas as pd
import ast
import os
import jax.numpy as jnp
from jax.typing import ArrayLike

# for referencing the custom library and editied files in this folder
sys.path.append("././")

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv, FrankaVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.core.controllers import PoseTaskTorqueController, PoseTaskVelocityController
from oscbf.utils.trajectory import SinusoidalTaskTrajectory, WaypointTaskTrajectory
from oscbf.utils.visualization import create_box


from barriertransformer import barrier_generate as barrier
from barriertransformer import visualization as vis
from barriertransformer import metrics as met

name_date = "cus_cluttered_qwen3.5_35b_pnp_v2"
SHOW_PLOTS = False
SAVE_DATA = False
RECORD_VIDEO = False

exp_title = "Cluttered_Tabletop_Custom_qwen3.5_35b_pnp"
prompt_vers = "v2"

np.random.seed(0)


@jax.tree_util.register_static
class CollisionsConfig(OSCBFTorqueConfig):
    def __init__(
        self,
        robot: Manipulator,
        z_min: float,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        # adding CBF barriers and other details
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.z_min = z_min
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        self.pos_min = np.array(pos_min)
        self.pos_max = np.array(pos_max)
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # EE safe containment
        h_ee_safe_Set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        # joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # singularity avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

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

        # Whole-body safe containment
        robot_num_pts = robot_collision_positions.shape[0]
        h_whole_body_upper = (
            jnp.tile(self.whole_body_pos_max, (robot_num_pts, 1))
            - robot_collision_positions
            - robot_collision_radii
        ).ravel()
        h_whole_body_lower = (
            robot_collision_positions
            - jnp.tile(self.whole_body_pos_min, (robot_num_pts, 1))
            - robot_collision_radii
        ).ravel()

        # return jnp.concatenate([h_collision, h_table])
        return jnp.concatenate(
            [
                h_ee_safe_Set,
                h_joint_limits,
                h_singularity,
                h_collision,
                h_table,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

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
    u_saf = cbf.safety_filter(z, u_nom)
    return u_saf, u_nom


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


def main(control_method="torque", num_bodies=3):
    assert control_method in ["torque", "velocity"]

    robot = load_panda()
    z_min = 0.1

    max_num_bodies = 5
    ee_init_pos = (0.240, -0.000, 0.429)

    # Sample a lot of collision bodies
    all_collision_pos = np.random.uniform(
        low=[0.2, -0.4, 0.1], high=[0.8, 0.4, 0.3], size=(max_num_bodies, 3)
    )
    all_collision_radii = np.random.uniform(low=0.07, high=0.1, size=(max_num_bodies,))
    # Only use a subset of them based on the desired quantity
    collision_pos = np.atleast_2d(all_collision_pos[:num_bodies])
    collision_radii = all_collision_radii[:num_bodies]
    # print(collision_pos)
    # print(collision_radii)
    collision_data = {"positions": collision_pos, "radii": collision_radii}

    cus_col_pos = [0.4, 0.3, 0.55]
    cus_col_len = [0.2, 0.2, 0.2]

    # Combine sphere obstacles and the box obstacle (approximated as a sphere) for the prompt
    all_col_pos = collision_pos.tolist() + [cus_col_pos]
    all_col_rad = collision_radii.tolist() + [cus_col_len[0] / 2.0]

    sinusoid_init_pos = (0.34, 0.34, 0.34)
    amplitude = (0.01, 0, 0.16)
    frequency = (0.21, 0, 2.06)

    # waypoint/pick and drop traj
    waypoints = np.array(
        [
            [0.45, -0.5, 0.55],  # t=0.0s: Start above pick location
            [0.45, -0.5, 0.15],  # t=2.0s: Reach down to pick object
            [0.45, -0.5, 0.55],  # t=4.0s: Lift object back up
            [0.45, 0.50, 0.55],  # t=7.0s: Move horizontally above drop location
            [0.45, 0.50, 0.15],  # t=9.0s: Lower down to drop location
        ]
    )
    # Define the exact timestamp (in seconds) for each waypoint
    times = np.array([0.5, 2.0, 4.0, 7.0, 9.0])
    # Maintain a constant downward-facing end-effector orientation
    init_rot = np.array(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ]
    )

    prompt = barrier.create_prompt_col(
        ee_init_pos,
        sinusoid_init_pos,
        amplitude,
        frequency,
        all_col_pos,
        all_col_rad,
    )

    # ee_pos_min = np.array([0.15, -0.25, 0.25])
    # ee_pos_max = np.array([0.75, 0.25, 0.75])
    # wb_pos_min = np.array([-0.5, -0.5, 0.0])
    # wb_pos_max = np.array([0.75, 0.5, 1.0])

    # llm outputs
    # model = "llama3.1"
    # print(f"Generating Barrier from {model}")
    # ee_pos_min, ee_pos_max, wb_pos_min, wb_pos_max = barrier.generate_barrier(
    #     user_prompt=prompt
    # )
    # print("Barriers Generated: ee:", ee_pos_min, ee_pos_max)
    # print("Barriers Generated: whole body:", wb_pos_min, wb_pos_max)

    # tests for llm results
    # Dynamically locate the results folder relative to this script's path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(
        script_dir, "..", "..", "results", "llm_res", "pnp_qwen3.5_35b"
    )
    filename = "2026-05-23_Cluttered_Tabletop_Custom_qwen3.5_35b_v2_barriers.csv"
    filepath = os.path.normpath(os.path.join(folder, filename))

    # Read CSV
    df = pd.read_csv(filepath)

    # Convert list columns from strings to actual lists
    list_cols = ["EE_Min", "EE_Max", "WB_Min", "WB_Max"]
    for col in list_cols:
        df[col] = df[col].apply(ast.literal_eval)

    # Format and print results
    for _, row in df.iterrows():
        print(f"\nExperiment:     {row['Experiment']}")
        print(f"Prompt Version: {row['Prompt Version']}")

        # Format each list to 2 decimal places as a tuple
        ee_pos_min = tuple(round(v, 2) for v in row["EE_Min"])
        ee_pos_max = tuple(round(v, 2) for v in row["EE_Max"])
        wb_pos_min = tuple(round(v, 2) for v in row["WB_Min"])
        wb_pos_max = tuple(round(v, 2) for v in row["WB_Max"])

    print(f"pos_min: {ee_pos_min}, pos_max: {ee_pos_max}")
    print(f"wb_min: {wb_pos_min}, wb_max: {wb_pos_max}")

    torque_config = CollisionsConfig(
        robot,
        z_min,
        collision_pos,
        collision_radii,
        ee_pos_min,
        ee_pos_max,
        wb_pos_min,
        wb_pos_max,
    )
    torque_cbf = CBF.from_config(torque_config)
    # traj = SinusoidalTaskTrajectory(
    #     init_pos=sinusoid_init_pos,
    #     init_rot=np.array(
    #         [
    #             [1, 0, 0],
    #             [0, -1, 0],
    #             [0, 0, -1],
    #         ]
    #     ),
    #     amplitude=amplitude,
    #     angular_freq=frequency,
    #     phase=(0, 0, 0),
    # )
    traj = WaypointTaskTrajectory(waypoints=waypoints, times=times, init_rot=init_rot)

    # velocity configs
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
            xyz_min=torque_config.pos_min,
            xyz_max=torque_config.pos_max,
            wb_xyz_min=torque_config.whole_body_pos_min,
            wb_xyz_max=torque_config.whole_body_pos_max,
            traj=traj,
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

    # create a box obstacle
    # create_box(
    #     pos=cus_col_pos,  # Center position [x, y, z] in world frame
    #     orn=[0, 0, 0, 1],  # Orientation quaternion [x, y, z, w]
    #     mass=0.0,  # Setting mass=0 makes it a fixed/static object
    #     sidelengths=cus_col_len,  # Dimensions along [x, y, z] axes
    #     use_collision=True,  # True: Robot physically collides with it in PyBullet
    #     # False: Purely visual (ghost object)
    #     rgba=[0.867, 0.016, 0.016, 1],  # Color [R, G, B, Alpha]
    #     client=env.client,  # Target the active PyBullet client instance
    # )

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

    # old main loop
    # while True:
    #     q_qdot = env.get_joint_state()
    #     z_zdot_ee_des = env.get_desired_ee_state()
    #     tau = compute_control(q_qdot, z_zdot_ee_des)
    #     env.apply_control(tau)
    #     env.step()

    if RECORD_VIDEO:
        env.client.startStateLogging(
            env.client.STATE_LOGGING_VIDEO_MP4,
            f"results/new/custom_table/custom_{name_date}_video.mp4",
        )

    duration = 11.0
    # timestep = 1 / 1000
    n_timestep = int(duration / timestep)

    j_state = []
    j_state_des = []
    u_unsafe = []
    u_safe = []
    h_hist = []

    for i in range(n_timestep):
        q_qdot = env.get_joint_state()
        z_zdot_ee_des = env.get_desired_ee_state()
        tau, tau_unsafe = compute_control(q_qdot, z_zdot_ee_des)
        env.apply_control(tau)
        env.step()

        j_state.append(q_qdot)
        j_state_des.append(z_zdot_ee_des)
        u_safe.append(tau)
        u_unsafe.append(tau_unsafe)

        h_val = torque_config.h_2(q_qdot)
        h_hist.append(h_val)

        if i == 1:
            cameras, pixel_width, pixel_height = vis.get_camera_matrices()
            images = []
            for view, proj in cameras:
                width, height, rgb, depth, seg = env.client.getCameraImage(
                    width=pixel_width,
                    height=pixel_height,
                    viewMatrix=view,
                    projectionMatrix=proj,
                    renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,  # ER_TINY_RENDERER
                )
                images.append(rgb)

            vis.plot_views(
                images,
                pixel_width,
                pixel_height,
                show_plots=SHOW_PLOTS,
                name=f"results/new/custom_table/{name_date}_cam",
                folder="tabletop",
                save_image=SAVE_DATA,
            )

    ts = duration * np.arange(n_timestep)

    vis.plot_link_simulations(
        np.array(j_state),
        np.array(j_state_des),
        np.array(u_safe),
        ts,
        show_plots=SHOW_PLOTS,
        save_image=SAVE_DATA,
        name=f"results/new/custom_table/{name_date}_links",
    )

    # metrics
    q_pos = jnp.array(j_state)[:, : robot.num_joints]
    p_actual = jnp.array(jax.vmap(robot.ee_position)(q_pos))
    p_target = jnp.array(j_state_des)[:, :3]

    # Calculate Whole-Body joint spheres over the trajectory using vmap
    wb_spheres_data = jnp.array(jax.vmap(robot.link_collision_data)(q_pos))
    joint_spheres = wb_spheres_data[:, :, :3]
    joint_sphere_radii = np.array(wb_spheres_data[0, :, 3])

    sim_data = met.SimulationData(
        dt=timestep,
        time=ts,
        q_traj=q_pos,
        u_actual=jnp.array(u_safe),
        u_nominal=jnp.array(u_unsafe),
        p_actual=p_actual,
        p_target=p_target,
        pos_min=ee_pos_min,
        pos_max=ee_pos_max,
        wb_min=wb_pos_min,
        wb_max=wb_pos_max,
        h_val=jnp.array(h_hist),
        joint_spheres=joint_spheres,
        joint_sphere_radii=joint_sphere_radii,
        collision_spheres=collision_pos,
        collision_sphere_radii=collision_radii,
        experiment_title= exp_title,
        prompt_version= prompt_vers,
    )

    if SAVE_DATA:
        met.generate_report(sim_data, output_dir="results/new/custom_table")
        met.save_barriers_to_csv(sim_data, output_dir="results/new/custom_table")

    mean_tau = met.compute_mean_abs_torque(sim_data.u_actual)
    vis.plot_per_joint_torque(
        mean_tau,
        show_plots=SHOW_PLOTS,
        save_image=SAVE_DATA,
        name=f"results/new/custom_table/{name_date}_jtorque",
    )
    vis.plot_barrier_evolution(
        time=ts,
        h_val=sim_data.h_val,
        u_safe=sim_data.u_actual,
        u_unsafe=sim_data.u_nominal,
        show_plots=SHOW_PLOTS,
        save_image=SAVE_DATA,
        name=f"results/new/custom_table/{name_date}_hevolve",
    )


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
        default=3,
        help="Number of collision bodies to simulate (default: 25)",
    )
    args = parser.parse_args()
    main(control_method=args.control_method, num_bodies=args.num_bodies)
