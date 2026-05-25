"""Testing the performance of OSCBF under many different safety constraints, namely:

1. End-effector set containment
2. Joint limit avoidance
3. Singularity avoidance
4. Collision avoidance
5. Whole-body set containment

While there are many other safety constraints that we could also account for, this
should give a good view of the controller's performance under common situations
encountered in practice.
"""

import sys
import pybullet
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from cbfpy import CBF
import pandas as pd
import ast
import os

sys.path.append("././")
# importing custom library
from barriertransformer import barrier_generate as barrier
from barriertransformer import visualization as vis
from barriertransformer import metrics as met

from oscbf.core.manipulator import Manipulator, load_panda
from oscbf.core.manipulation_env import FrankaTorqueControlEnv
from oscbf.core.oscbf_configs import OSCBFTorqueConfig
from oscbf.utils.trajectory import SinusoidalTaskTrajectory, WaypointTaskTrajectory
from oscbf.core.controllers import PoseTaskTorqueController

RECORD_VIDEO = False
SAVE_DATA = False
SHOW_IMAGES = False
name_date = "mult_saf_pnp_qwen3.5_35b_v2"

exp_title = "Multiple_Safety_Conditions_pnp_qwen3.5_35b"
prompt_ver = "v2"


@jax.tree_util.register_static
class CombinedConfig(OSCBFTorqueConfig):
    def __init__(
        self,
        robot: Manipulator,
        pos_min: ArrayLike,
        pos_max: ArrayLike,
        collision_positions: ArrayLike,
        collision_radii: ArrayLike,
        whole_body_pos_min: ArrayLike,
        whole_body_pos_max: ArrayLike,
    ):
        self.pos_min = np.asarray(pos_min)
        self.pos_max = np.asarray(pos_max)
        self.q_min = robot.joint_lower_limits
        self.q_max = robot.joint_upper_limits
        self.singularity_tol = 1e-3
        self.collision_positions = np.atleast_2d(collision_positions)
        self.collision_radii = np.ravel(collision_radii)
        assert len(collision_positions) == len(collision_radii)
        self.num_collision_bodies = len(collision_positions)
        self.whole_body_pos_min = np.asarray(whole_body_pos_min)
        self.whole_body_pos_max = np.asarray(whole_body_pos_max)
        super().__init__(robot)

    def h_2(self, z, **kwargs):
        # Extract values
        q = z[: self.num_joints]
        ee_pos = self.robot.ee_position(q)
        q_min = jnp.asarray(self.q_min)
        q_max = jnp.asarray(self.q_max)

        # EE Set Containment
        h_ee_safe_set = jnp.concatenate([self.pos_max - ee_pos, ee_pos - self.pos_min])

        # Joint Limit Avoidance
        h_joint_limits = jnp.concatenate([q_max - q, q - q_min])

        # Singularity Avoidance
        sigmas = jax.lax.linalg.svd(self.robot.ee_jacobian(q), compute_uv=False)
        h_singularity = jnp.array([jnp.prod(sigmas) - self.singularity_tol])

        # Collision Avoidance
        robot_collision_pos_rad = self.robot.link_collision_data(q)
        robot_collision_positions = robot_collision_pos_rad[:, :3]
        robot_collision_radii = robot_collision_pos_rad[:, 3, None]
        robot_num_pts = robot_collision_positions.shape[0]
        center_deltas = (
            robot_collision_positions[:, None, :] - self.collision_positions[None, :, :]
        ).reshape(-1, 3)
        radii_sums = (
            robot_collision_radii[:, None] + self.collision_radii[None, :]
        ).reshape(-1)
        h_collision = jnp.linalg.norm(center_deltas, axis=1) - radii_sums

        # Whole-body Set Containment
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

        return jnp.concatenate(
            [
                h_ee_safe_set,
                h_joint_limits,
                h_singularity,
                h_collision,
                h_whole_body_upper,
                h_whole_body_lower,
            ]
        )

    def alpha(self, h):
        return 10.0 * h

    def alpha_2(self, h_2):
        return 10.0 * h_2


# @partial(jax.jit, static_argnums=(0, 1, 2))
def compute_control(
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
    tau = cbf.safety_filter(z, u_nom)
    return tau, u_nom


def main():
    robot = load_panda()
    # ee_pos_min = np.array([0.15, -0.25, 0.25])
    # ee_pos_max = np.array([0.75, 0.25, 0.75])
    # wb_pos_min = np.array([-0.5, -0.5, 0.0])
    # wb_pos_max = np.array([0.75, 0.5, 1.0])

    ee_init_pos = (0.240, -0.000, 0.429)

    collision_pos = np.array([[0.5, 0.5, 0.5]])
    collision_radii = np.array([0.3])

    sinusoid_init_pos = (0.39, 0.37, 0.33)
    amplitude = (0, 0, 0.29)
    frequency = (0, 0, 4.18)

    # prompt = barrier.create_prompt_col_old(
    #     ([0, 0, 0]),
    #     ([0.240, -0.000, 0.429]),
    #     sinusoid_init_pos,
    #     amplitude,
    #     frequency,
    #     collision_pos.tolist(),
    #     collision_radii,
    # )

    prompt = barrier.create_prompt_col(
        ee_init_pos,
        sinusoid_init_pos,
        amplitude,
        frequency,
        collision_pos,
        collision_radii,
    )

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
    # model = "llama3.1:latest"
    # print(f"Generating Barrier from {model}")
    # ee_pos_min, ee_pos_max, wb_pos_min, wb_pos_max = barrier.generate_barrier(
    #     user_prompt=prompt
    # )
    # print("Barriers Generated: ee:", ee_pos_min, ee_pos_max)
    # print("Barriers Generated: whole body:", wb_pos_min, wb_pos_max)

    # tests for llm results
    # Dynamically locate the results folder relative to this script's path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(script_dir, "..", "..", "results", "llm_res", "pnp_qwen3.5_35b")
    filename = "2026-05-23_Multiple_Safety_Conditions_qwen3.5_35b_v2_barriers.csv"
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

    collision_data = {"positions": collision_pos, "radii": collision_radii}
    config = CombinedConfig(
        robot,
        ee_pos_min,
        ee_pos_max,
        collision_pos,
        collision_radii,
        wb_pos_min,
        wb_pos_max,
    )
    cbf = CBF.from_config(config)
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
    env = FrankaTorqueControlEnv(
        config.pos_min,
        config.pos_max,
        collision_data=collision_data,
        wb_xyz_min=wb_pos_min,
        wb_xyz_max=wb_pos_max,
        traj=traj,
        load_floor=False,
        bg_color=(1, 1, 1),
        real_time=True,
    )

    env.client.resetDebugVisualizerCamera(
        cameraDistance=2,
        cameraPitch=-27.80,
        cameraYaw=36.80,
        cameraTargetPosition=(0.08, 0.49, -0.04),
    )

    kp_pos = 50.0
    kp_rot = 20.0
    kd_pos = 20.0
    kd_rot = 10.0
    kp_joint = 10.0
    kd_joint = 5.0
    osc_controller = PoseTaskTorqueController(
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

    @jax.jit
    def compute_control_jit(z, z_des):
        return compute_control(robot, osc_controller, cbf, z, z_des)

    

    # while True:
    #     joint_state = env.get_joint_state()
    #     ee_state_des = env.get_desired_ee_state()
    #     tau = compute_control_jit(joint_state, ee_state_des)
    #     env.apply_control(tau)
    #     env.step()

    if RECORD_VIDEO:
        # for saving the video in the env
        env.client.startStateLogging(
            env.client.STATE_LOGGING_VIDEO_MP4, f"results/new/mulsafe/{name_date}_video.mp4"
        )

    duration = 11.0
    timestep = 1 / 1000
    n_timestep = int(duration / timestep)

    j_state = []
    j_state_des = []
    u_safe = []
    u_unsafe = []
    h_hist = []

    for i in range(n_timestep):
        joint_state = env.get_joint_state()
        ee_state_des = env.get_desired_ee_state()
        tau, u_nom = compute_control_jit(joint_state, ee_state_des)
        env.apply_control(tau)
        env.step()

        j_state.append(joint_state)
        j_state_des.append(ee_state_des)
        u_safe.append(tau)
        u_unsafe.append(u_nom)

        # Calculate h_val using config.h_2
        h_val = config.h_2(joint_state)
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
                name=f"results/new/mulsafe/{name_date}",
                folder="test_dynamotion_plots",
                save_image=SAVE_DATA,
            )

    ts = duration * np.arange(n_timestep)
    vis.plot_link_simulations(
        np.array(j_state),
        np.array(j_state_des),
        np.array(u_safe),
        ts,
        show_plots=SHOW_IMAGES,
        save_image=SAVE_DATA,
        name=f"results/new/mulsafe/{name_date}_links",
    )

    # --- METRICS INTEGRATION ---
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
        prompt_version=prompt_ver,
    )

    if SAVE_DATA:
        met.generate_report(sim_data, output_dir="results/new/mulsafe")
        met.save_barriers_to_csv(sim_data, output_dir="results/new/mulsafe")

    mean_tau = met.compute_mean_abs_torque(sim_data.u_actual)
    vis.plot_per_joint_torque(
        mean_tau,
        show_plots=SHOW_IMAGES,
        save_image=SAVE_DATA,
        name=f"results/new/mulsafe/{name_date}_jtorque",
    )
    vis.plot_barrier_evolution(
        time=ts,
        h_val=sim_data.h_val,
        u_safe=sim_data.u_actual,
        u_unsafe=sim_data.u_nominal,
        show_plots=SHOW_IMAGES,
        save_image=SAVE_DATA,
        name=f"results/new/mulsafe/{name_date}_hevolve",
    )


if __name__ == "__main__":
    main()
