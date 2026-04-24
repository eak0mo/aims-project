"""Pybullet debug visualizer helper functions"""

from typing import Optional

import numpy as np
import numpy.typing as npt
import pybullet
from pybullet_utils.bullet_client import BulletClient


def visualize_3D_box(
    box: npt.ArrayLike,
    padding: Optional[npt.ArrayLike] = None,
    rgba: npt.ArrayLike = (1, 0, 0, 0.5),
    client: Optional[BulletClient] = None,
) -> int:
    """Visualize a box in Pybullet

    Args:
        box (npt.ArrayLike): Lower and upper xyz limits of the axis-aligned box, shape (2, 3)
        padding (Optional[npt.ArrayLike]): If expanding (or contracting) the boxes by a certain amount, include the
            (x, y, z) padding distances here (shape (3,)). Defaults to None.
        rgba (npt.ArrayLike): Color of the box (RGB + alpha), shape (4,). Defaults to (1, 0, 0, 0.5).
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: Pybullet ID of the box
    """
    lower, upper = box
    if padding is not None:
        lower -= padding
        upper += padding
    return create_box(
        pos=(lower + (upper - lower) / 2),  # Midpoint
        orn=(0, 0, 0, 1),
        mass=0,
        sidelengths=(upper - lower),
        use_collision=False,
        rgba=rgba,
        client=client,
    )


def create_box(
    pos: npt.ArrayLike,
    orn: npt.ArrayLike,
    mass: float,
    sidelengths: npt.ArrayLike,
    use_collision: bool,
    rgba: npt.ArrayLike = (1, 1, 1, 1),
    client: Optional[BulletClient] = None,
) -> int:
    """Creates a rigid box in the Pybullet simulation

    Args:
        pos (npt.ArrayLike): Position of the box in world frame, shape (3)
        orn (npt.ArrayLike): Orientation (XYZW quaternion) of the box in world frame, shape (4,)
        mass (float): Mass of the box. If set to 0, the box is fixed in space
        sidelengths (npt.ArrayLike): Sidelengths of the box along the local XYZ axes, shape (3,)
        use_collision (bool): Whether or not collision is enabled for the box
        rgba (npt.ArrayLike, optional): Color of the box, with each RGBA value being in [0, 1].
            Defaults to (1, 1, 1, 1) (white)
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: ID of the box in Pybullet
    """
    client: pybullet = pybullet if client is None else client
    if len(sidelengths) != 3:
        raise ValueError("Must provide the dimensions of the three sides of the box")
    half_extents = np.asarray(sidelengths) / 2
    visual_id = client.createVisualShape(
        pybullet.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=rgba,
    )
    if use_collision:
        collision_id = client.createCollisionShape(
            pybullet.GEOM_BOX,
            halfExtents=half_extents,
        )
    else:
        collision_id = -1
    box_id = client.createMultiBody(
        baseMass=mass,
        basePosition=pos,
        baseOrientation=orn,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )
    return box_id


def visualize_3D_sphere(
    center: npt.ArrayLike,
    radius: float,
    rgba: npt.ArrayLike = (1, 0, 0, 0.5),
    client: Optional[BulletClient] = None,
) -> int:
    """Visualize a sphere in Pybullet

    Args:
        center (npt.ArrayLike): Center of the sphere, shape (3,)
        radius (float): Radius of the sphere
        rgba (npt.ArrayLike): Color of the sphere (RGB + alpha), shape (4,). Defaults to (0, 1, 0, 0.5).
        client (BulletClient, optional): If connecting to multiple physics servers, include the client
            (the class instance, not just the ID) here. Defaults to None (use default connected client)

    Returns:
        int: Pybullet ID of the sphere
    """
    client: pybullet = pybullet if client is None else client
    visual_id = client.createVisualShape(
        pybullet.GEOM_SPHERE,
        radius=radius,
        rgbaColor=rgba,
    )
    collision_id = -1
    sphere_id = client.createMultiBody(
        baseMass=0,
        basePosition=center,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
    )
    return sphere_id
