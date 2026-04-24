URDF is originally from https://github.com/erwincoumans/pybullet_robots/tree/master/data/franka_panda

Modifications include:
- Removing the `package://` linking to meshes
- Set a fixed joint axis to `0, 0, 1` instead of `0, 0, 0` (which can have issues with certain parsers)
- Locked the gripper joints (to ignore in the kinematic chain)
- Removed "safety controller" entries
