import pybullet
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_camera_matrices(
    target=(0.44, 0.16, 0.28),
    distance=1,
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
    images, pixel_width, pixel_height, show_plots=False, name=None, save_image=False
):
    camera_labels = ["Camera 1", "Camera 2", "Camera 3", "Top-down"]
    mosaic = [["Camera 1", "Camera 2"], ["Camera 3", "Top-down"]]

    fig, axes = plt.subplot_mosaic(mosaic, figsize=(5, 4), dpi=300)

    for label, img in zip(camera_labels, images):
        np_img = np.reshape(img, (pixel_height, pixel_width, 4))
        axes[label].imshow(np_img)
        axes[label].set_title(label, fontsize=8)
        axes[label].axis("off")

    if save_image:
        plt.savefig(name + ".pdf")

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
