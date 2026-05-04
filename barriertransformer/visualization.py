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
        for i, img in enumerate(images):
            np_img = np.reshape(img, (pixel_height, pixel_width, 4))
            fig_temp, ax_temp = plt.subplots(figsize=(5, 4), dpi=300)
            ax_temp.imshow(np_img)
            ax_temp.axis("off")
            fig_temp.savefig(f"{name}_{i+1}.pdf", bbox_inches="tight", pad_inches=0)
            plt.close(fig_temp)

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

    def _plot_pos(ax):
        ax.plot(ts, ee_pos_hist[:, 0], label="Current X", color="r", linestyle="-")
        ax.plot(ts, ee_pos_hist[:, 1], label="Current Y", color="g", linestyle="-")
        ax.plot(ts, ee_pos_hist[:, 2], label="Current Z", color="b", linestyle="-")
        ax.plot(
            ts,
            ee_des_pos_hist[:, 0],
            label="Desired X",
            color="r",
            linestyle="--",
            alpha=0.7,
        )
        ax.plot(
            ts,
            ee_des_pos_hist[:, 1],
            label="Desired Y",
            color="g",
            linestyle="--",
            alpha=0.7,
        )
        ax.plot(
            ts,
            ee_des_pos_hist[:, 2],
            label="Desired Z",
            color="b",
            linestyle="--",
            alpha=0.7,
        )
        ax.set_ylabel("Position (m)")
        ax.set_title("Current and Desired EE Position")
        ax.legend(ncol=2)

    def _plot_cmd(ax):
        num_links = u_safe_hist.shape[1]
        for i in range(num_links):
            label = names[i] if names is not None else f"Link {i + 1}"
            ax.plot(ts, u_safe_hist[:, i], label=label)
        ax.set_ylabel("Control Command")
        ax.set_title("Safe Control Commands")
        ax.legend(ncol=4)

    def _plot_err(ax):
        ax.plot(ts, ee_error[:, 0], label="Error X", color="r")
        ax.plot(ts, ee_error[:, 1], label="Error Y", color="g")
        ax.plot(ts, ee_error[:, 2], label="Error Z", color="b")
        error_norm = np.linalg.norm(ee_error, axis=1)
        ax.plot(ts, error_norm, label="Error Norm", color="k", linestyle=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Difference Between Task and EE Position")
        ax.legend()

    # Plot 1: EE Position
    _plot_pos(axes["pos"])

    # Plot 2: Safe Command
    _plot_cmd(axes["cmd"])

    # Plot 3: Error
    _plot_err(axes["err"])

    # plt.tight_layout()

    if save_image:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        _plot_pos(ax1)
        fig1.savefig(f"{name}_1.pdf", bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, 4))
        _plot_cmd(ax2)
        fig2.savefig(f"{name}_2.pdf", bbox_inches="tight")
        plt.close(fig2)

        fig3, ax3 = plt.subplots(figsize=(10, 4))
        _plot_err(ax3)
        fig3.savefig(f"{name}_3.pdf", bbox_inches="tight")
        plt.close(fig3)

        plt.savefig(name + ".pdf")

    if show_plots:
        plt.show()

    return fig, axes
