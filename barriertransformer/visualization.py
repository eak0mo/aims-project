import pybullet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import jax
import os
from oscbf.core.manipulator import load_panda


def get_camera_matrices(
    target=(0.44, 0.16, 0.28),
    distance=1.7,
    pitch=-27.8,
    roll=0,
    fov=60,
    pixel_width=1080,
    pixel_height=720,
    near=0.01,
    far=10.0,
    up_axis=2,
):
    aspect = pixel_width / pixel_height
    """
    Returns a list of (view_matrix, projection_matrix) tuples
    for 3 cameras spaced 120 degrees apart around the robot.
    """
    yaw_angles = [45, 110, 210]
    matrices = []

    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=fov,
        aspect=aspect,
        nearVal=near,
        farVal=far,
    )

    for yaw in yaw_angles:
        view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=list(target),
            distance=distance,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            upAxisIndex=up_axis,
        )
        matrices.append((view_matrix, projection_matrix))

    # --- top-down camera ---
    top_down_height = 1.3  # adjust to your robot's max height + margin
    view_matrix_top = pybullet.computeViewMatrix(
        cameraEyePosition=[target[0], target[1], top_down_height],
        cameraTargetPosition=list(target),
        cameraUpVector=[0, 1, 0],  # Y-axis as up so image aligns along Y
    )
    matrices.append((view_matrix_top, projection_matrix))

    return matrices, pixel_width, pixel_height


def plot_views(
    images,
    pixel_width,
    pixel_height,
    show_plots=False,
    name=None,
    save_image=False,
    folder: str = "test_dynamotion_plots",
):
    set_style()
    camera_labels = ["Camera 1", "Camera 2", "Camera 3", "Top-down"]
    mosaic = [["Camera 1", "Camera 2"], ["Camera 3", "Top-down"]]

    fig, axes = plt.subplot_mosaic(mosaic, figsize=(5, 4), dpi=300)

    for label, img in zip(camera_labels, images):
        np_img = np.reshape(img, (pixel_height, pixel_width, 4))
        axes[label].imshow(np_img)
        axes[label].set_title(label, fontsize=8)
        axes[label].axis("off")

    if save_image:
        # path = os.path.join(folder, name)
        plt.savefig(name + ".pdf")
        # plt.savefig(path)

    if show_plots:
        plt.show()

    return


def set_style():
    my_pal = [
        "000000",
        "29AF8C",
        "97BE49",
        "3D9CCC",
        "7C60C6",
        "D58C2E",
        "C9492C",
        "44546A",
    ]
    my_pal = [f"#{c}" for c in my_pal]
    sns.reset_defaults()  # useful when adjusting style a lot
    sns.set_theme(
        context="paper",
        style="ticks",
        # palette="Set2",
        palette=my_pal,
        rc={
            "pdf.fonttype": 42,  # embed font in output
            "svg.fonttype": "none",  # embed font in output
            "figure.facecolor": "white",
            "figure.dpi": 100,
            "axes.facecolor": "None",
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.right": False,
            "axes.spines.top": False,
        },
    )


def plot_link_simulations(
    q_hist,
    q_des_hist,
    u_safe_hist,
    ts,
    names=None,
    show_plots=False,
    save_image=False,
    name="simulation_plot",
):
    """
    Plots 3 different plots: the current and desired EE position, the safe command,
    and the difference between task and the end-effector position.
    """

    set_style()
    q_hist = np.asarray(q_hist)
    q_des_hist = np.asarray(q_des_hist)
    u_safe_hist = np.asarray(u_safe_hist)
    ts = np.asarray(ts)

    # 1. Compute actual EE position
    robot = load_panda()

    # We vmap over the joint positions (first num_joints elements of q_hist)
    q_pos = q_hist[:, : robot.num_joints]

    # Compute EE positions using forward kinematics
    ee_pos_hist = jax.vmap(robot.ee_position)(q_pos)

    # EE desired is in q_des_hist (N, 18), first 3 are positions
    ee_des_pos_hist = q_des_hist[:, :3]

    # EE task error (difference between desired task and actual EE position)
    ee_error = ee_des_pos_hist - ee_pos_hist

    # Create 3 subplots using subplot_mosaic
    mosaic = [["pos"], ["cmd"], ["err"]]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(10, 12), sharex=True)

    # Plot 1: EE Position
    ax_pos = axes["pos"]
    ax_pos.plot(ts, ee_pos_hist[:, 0], label="Current X", color="r", linestyle="-")
    ax_pos.plot(ts, ee_pos_hist[:, 1], label="Current Y", color="g", linestyle="-")
    ax_pos.plot(ts, ee_pos_hist[:, 2], label="Current Z", color="b", linestyle="-")
    ax_pos.plot(
        ts,
        ee_des_pos_hist[:, 0],
        label="Desired X",
        color="r",
        linestyle="--",
        alpha=0.7,
    )
    ax_pos.plot(
        ts,
        ee_des_pos_hist[:, 1],
        label="Desired Y",
        color="g",
        linestyle="--",
        alpha=0.7,
    )
    ax_pos.plot(
        ts,
        ee_des_pos_hist[:, 2],
        label="Desired Z",
        color="b",
        linestyle="--",
        alpha=0.7,
    )
    ax_pos.set_ylabel("Position (m)")
    ax_pos.set_title("Current and Desired EE Position")
    ax_pos.legend( ncol=2)

    # Plot 2: Safe Command
    ax_cmd = axes["cmd"]
    num_links = u_safe_hist.shape[1]
    for i in range(num_links):
        label = names[i] if names is not None else f"Link {i + 1}"
        ax_cmd.plot(ts, u_safe_hist[:, i], label=label)
    ax_cmd.set_ylabel("Control Command")
    ax_cmd.set_title("Safe Control Commands")
    ax_cmd.legend( ncol=4)

    # Plot 3: Error
    ax_err = axes["err"]
    ax_err.plot(ts, ee_error[:, 0], label="Error X", color="r")
    ax_err.plot(ts, ee_error[:, 1], label="Error Y", color="g")
    ax_err.plot(ts, ee_error[:, 2], label="Error Z", color="b")
    error_norm = np.linalg.norm(ee_error, axis=1)
    ax_err.plot(ts, error_norm, label="Error Norm", color="k", linestyle=":")
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position Error (m)")
    ax_err.set_title("Difference Between Task and EE Position")
    ax_err.legend()

    # plt.tight_layout()

    if save_image:
        plt.savefig(name + ".pdf")

    if show_plots:
        plt.show()

    return fig, axes
