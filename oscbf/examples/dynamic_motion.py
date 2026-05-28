"""Testing the performance of OSCBF during dynamic motions and input constraints

In general, we will command a rapid motion of the end-effector into the unsafe set,
and observe the controller's behavior under velocity control and torque control.

The reduced-order (velocity-control) OSCBF has no lower-level understanding of torque
limits, so the full-order (torque-control) OSCBF should perform better in this case.
"""

import pybullet
import argparse
import sys

from functools import partial

import numpy as np
import jax
import pandas as pd
import ast
import os
import jax.numpy as jnp
from jax.typing import ArrayLike
# import matplotlib.pyplot as plt

sys.path.append("././")
from barriertransformer.barrier_generate import (
    generate_barrier,
    create_prompt,
    create_prompt_old,
    create_prompt_col,
    create_prompt_pnp,
    load_barriers_from_csv,
)
from barriertransformer import visualization as vis
from barriertransformer import metrics as met
# from package.pack import test

from cbfpy import CBF
from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv, FrankaVelocityControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig, OSCBFVelocityConfig
from oscbf.utils.trajectory import SinusoidalTaskTrajectory, WaypointTaskTrajectory
from oscbf.core.controllers import (
    PoseTaskTorqueController,
    PoseTaskVelocityController,
)
from oscbf.utils.visualization import create_box

DATA_DIR = "results/new/dymo"
SHOW_IMAGES = True
name_date = "dynamic_motion_qwen3.5_35b_pnp_v2"
SAVE_DATA = False
PAUSE_FOR_PICTURES = False
RECORD_VIDEO = False
PICTURE_IDXS = [1000, 1250, 1600, 1900, 2200]

exp_title = "Dynamic_Motion_qwen3.5_35b_pnp"
prompt_vers = "v2"


@jax.tree_util.register_static
class EESafeSetTorqueConfig(OSCBFTorqueConfig):
    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        compensate_centrifugal_coriolis: bool,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.singularity_tol = 1e-3
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        super().__init__(
            robot, compensate_centrifugal_coriolis=compensate_centrifugal_coriolis
        )

    def h_2(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)
        # return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])
        h_ee_safe_set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        # added singularlity avoidance from multiple_safety_conditions.py
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])
        # print(f"hvals {h_singularity}, {h_ee_safe_set}")

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # return jnp.concatenate([h_ee_safe_set, h_joint_limits, h_singularity])
        return jnp.concatenate([h_ee_safe_set, h_singularity])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


@jax.tree_util.register_static
class EESafeSetVelocityConfig(OSCBFVelocityConfig):
    def __init__(self, robot: Manipulator, pos_min: ArrayLike, pos_max: ArrayLike):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        super().__init__(robot)

    def h_1(self, z, **kwargs):
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        return jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2, 3))
def compute_torque_control(
    robot: Manipulator,
    osc_controller: PoseTaskTorqueController,
    cbf: CBF,
    compensate_centrifugal_coriolis: bool,
    z: ArrayLike,
    z_ee_des: ArrayLike,
):
    q = z[: robot.num_joints]
    qdot = z[robot.num_joints :]
    M, M_inv, g, c, J, ee_tmat = robot.torque_control_matrices(q, qdot)

    if not compensate_centrifugal_coriolis:
        c = jnp.zeros(robot.num_joints)

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
    return cbf.safety_filter(z, u_nom), u_nom  # added unsafe output


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


def main(control_method="torque"):
    assert control_method in ["torque", "velocity"]

    robot = load_panda()
    # pos_min = (0.25, -0.25, 0.25)
    # pos_max = (0.65, 0.25, 0.65)

    # q0 = env.get_joint_state()[: robot.num_joints] doesn't work as it is defined later
    # ee_pos0 = np.array(robot.ee_position(q0))
    # print(
    #     f"Starting EE Position: [{ee_pos0[0]:.3f}, {ee_pos0[1]:.3f}, {ee_pos0[2]:.3f}]"
    # )

    ee_init_pos = (0.240, -0.000, 0.429)
    # sinusoid
    amplitude = (0, 0.14, 0)
    frequency = (0, 0.59, 0)
    sinusoid_init_pos = (0.37, 0.49, 0.45)

    # old values
    # amplitude = (0.25, 0, 0)
    # frequency = (5, 0, 0)
    # sinusoid_init_pos = (0.55, 0, 0.45)

    # sinusoid prompt
    # prompt = create_prompt(
    #     ee_init_pos, sinusoid_init_pos, amplitude, frequency
    # )
    # prompt = create_prompt_col(ee_init_pos, sinusoid_init_pos, amplitude, frequency)

    # prompt = create_prompt_old(
    #     ee_init_pos, sinusoid_init_pos, amplitude, frequency
    # )

    # pick and drop trajectory
    # waypoint/pick and drop traj
    way_point_init_post = (0.45, -0.5, 0.55)
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
    # pick and drop prompt
    # prompt = create_prompt_pnp(ee_init_pos, way_point_init_post, waypoints, times)
    # print(prompt)

    # integration with llama 3.1
    # model = "llama3.1"
    # print(f"Generating Barrier from {model}")
    # pos_min, pos_max, wb_min, wb_max = generate_barrier(
    #     user_prompt=prompt, sin_traj=False
    # )
    # print("Barriers Generated: ee:", pos_min, pos_max)
    # print("Barriers Generated: whole body:", wb_min, wb_max)

    # for pick and drop barrier from claude as reference
    # pos_min = (0.25, -0.7, -0.05)
    # pos_max = (0.65,  0.7,  0.75)
    # wb_min  = (-0.34, -0.70, -0.20)
    # wb_max  = ( 0.85,  0.70,  0.75)

    # tests for llm results
    # Dynamically locate the results folder relative to this script's path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(
        script_dir, "..", "..", "results", "llm_res", "pnp_qwen3.5_35b"
    )
    filename = "2026-05-23_Dynamic_Motion_qwen3.5_35b_v2_barriers.csv"

    pos_min, pos_max, wb_min, wb_max = load_barriers_from_csv(folder, filename)


    # NOTE: This term has a noticeable impact on the performance for this demo.
    # It's often neglected due to computational demands and model error
    compensate_centrifugal_coriolis = False

    torque_config = EESafeSetTorqueConfig(
        robot,
        pos_min,
        pos_max,
        compensate_centrifugal_coriolis=compensate_centrifugal_coriolis,
    )
    torque_cbf = CBF.from_config(torque_config)
    velocity_config = EESafeSetVelocityConfig(robot, pos_min, pos_max)
    velocity_cbf = CBF.from_config(velocity_config)
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
    timestep = 1 / 1000
    bg_color = (1, 1, 1)
    if control_method == "torque":
        env = FrankaTorqueControlEnv(
            torque_config.pos_min,
            torque_config.pos_max,
            traj=traj,
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )
    else:
        env = FrankaVelocityControlEnv(
            velocity_config.pos_min,
            velocity_config.pos_max,
            traj=traj,
            real_time=True,
            bg_color=bg_color,
            load_floor=False,
            timestep=timestep,
        )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=1.00,
        cameraYaw=12,
        cameraPitch=-2.6,
        cameraTargetPosition=(0.44, 0.16, 0.28),
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
            robot,
            osc_torque_controller,
            torque_cbf,
            compensate_centrifugal_coriolis,
            z,
            z_ee_des,
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

    # create a box obstacle
    # create_box(
    #     pos=[0.60, 0, 0.45],  # Center position [x, y, z] in world frame
    #     orn=[0, 0, 0, 1],  # Orientation quaternion [x, y, z, w]
    #     mass=0.0,  # Setting mass=0 makes it a fixed/static object
    #     sidelengths=[0.1, 0.1, 0.1],  # Dimensions along [x, y, z] axes
    #     use_collision=True,  # True: Robot physically collides with it in PyBullet
    #     # False: Purely visual (ghost object)
    #     rgba=[0.867, 0.016, 0.016, 1],  # Color [R, G, B, Alpha]
    #     client=env.client,  # Target the active PyBullet client instance
    # )

    if RECORD_VIDEO:
        # for saving a live recoding of the simulation from the environment.
        env.client.startStateLogging(
            env.client.STATE_LOGGING_VIDEO_MP4, f"results/new/dymo/{name_date}.mp4"
        )

    duration = 11.0
    num_timestep = int(duration / timestep)

    # while True:
    #     q_qdot = env.get_joint_state()
    #     z_zdot_ee_des = env.get_desired_ee_state()
    #     tau = compute_control(q_qdot, z_zdot_ee_des)
    #     env.apply_control(tau)
    #     env.step()
    q_hist = []
    q_des_hist = []
    u_safe_hist = []
    u_unsafe_hist = []
    h_hist = []

    for i in range(num_timestep):
        q_qdot = env.get_joint_state()
        z_zdot_ee_des = env.get_desired_ee_state()
        tau, u_unsafe = compute_control(q_qdot, z_zdot_ee_des)
        env.apply_control(tau)
        env.step()

        q_hist.append(q_qdot)
        q_des_hist.append(z_zdot_ee_des)
        u_safe_hist.append(tau)
        u_unsafe_hist.append(u_unsafe)

        if control_method == "torque":
            h_val = torque_cbf.h_2(z_zdot_ee_des)
        # elif control_method == "velocity":
        #     h_val = velocity_cbf.h_np(q_qdot, z_zdot_ee_des)
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
                show_plots=SHOW_IMAGES,
                name=f"results/new/dymo/{name_date}",
                folder="test_dynamotion_plots",
                save_image=SAVE_DATA,
            )
    print(h_val)

    ts = duration * np.arange(num_timestep)

    # print(np.array(h_val).shape)
    # print(h_hist[-1])

    vis.plot_link_simulations(
        np.array(q_hist),
        np.array(q_des_hist),
        np.array(u_safe_hist),
        ts,
        show_plots=SHOW_IMAGES,
        save_image=SAVE_DATA,
        name=f"results/new/dymo/{name_date}_links",
    )

    # # Converting lists to JAX arrays to speed up metric computation

    # Calculate End-Effector trajectory for MTE and SVR metrics
    q_pos = jnp.array(q_hist)[:, : robot.num_joints]
    p_actual = jnp.array(jax.vmap(robot.ee_position)(q_pos))
    p_target = jnp.array(q_des_hist)[:, :3]

    # Calculate Whole-Body joint spheres over the trajectory using vmap
    wb_spheres_data = jnp.array(jax.vmap(robot.link_collision_data)(q_pos))
    joint_spheres = wb_spheres_data[:, :, :3]
    joint_sphere_radii = np.array(wb_spheres_data[0, :, 3])

    sim_data = met.SimulationData(
        dt=timestep,
        time=ts,
        q_traj=q_pos,
        u_actual=jnp.array(u_safe_hist),
        u_nominal=jnp.array(u_unsafe_hist),
        p_actual=p_actual,
        p_target=p_target,
        pos_min=pos_min,
        pos_max=pos_max,
        wb_min=wb_min,
        wb_max=wb_max,
        h_val=jnp.array(h_hist),
        joint_spheres=joint_spheres,
        joint_sphere_radii=joint_sphere_radii,
        collision_spheres=None,
        collision_sphere_radii=None,
        experiment_title=exp_title,
        prompt_version=prompt_vers,
    )
    #
    # # 2. Generate CSV Report
    # # Calls all jitted functions and saves them to 'results/...'
    if SAVE_DATA:
        met.generate_report(sim_data, output_dir="results/new/dymo")
        met.save_barriers_to_csv(sim_data, output_dir="results/new/dymo")
    #
    # # 3. Generate Visualizations
    mean_tau = met.compute_mean_abs_torque(sim_data.u_actual)
    vis.plot_per_joint_torque(
        mean_tau,
        show_plots=SHOW_IMAGES,
        save_image=SAVE_DATA,
        name=f"results/new/dymo/{name_date}_jtorque",
    )
    #
    vis.plot_barrier_evolution(
        time=ts,
        h_val=sim_data.h_val,
        u_safe=sim_data.u_actual,
        u_unsafe=sim_data.u_nominal,
        show_plots=SHOW_IMAGES,
        save_image=SAVE_DATA,
        name=f"results/new/dymo/{name_date}_hevolve",
    )
    # # --- END METRICS INTEGRATION EXAMPLE ---
    # """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run end-effector safe-set containment experiment."
    )
    parser.add_argument(
        "--control_method",
        type=str,
        choices=["torque", "velocity"],
        default="torque",
        help="Control method to use (default: torque)",
    )
    args = parser.parse_args()
    main(control_method=args.control_method)
