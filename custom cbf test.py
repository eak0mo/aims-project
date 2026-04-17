import jax.numpy as jnp
from cbfpy import CBF, CBFConfig
import pybullet
import time
import pybullet_data
# need an environment to get states


# pybullet env details
#choose connection method: GUI, DIRECT, SHARED_MEMORY
pybullet.connect(pybullet.GUI)
pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

pybullet.loadURDF(("plane.urdf"), 0, 0, 0)
#load URDF, given a relative or absolute file+path
# obj = pybullet.loadURDF(("r2d2.urdf"),[0,0,0.5])

# load a ball as a target 
ball = pybullet.loadURDF(("sphere2.urdf"), [3, 3, 10])

posX = 0
posY = 3
posZ = 2
kuka = pybullet.loadURDF(("kuka_iiwa/model.urdf"), posX,
                         posY, posZ)

#query the number of joints of the object
numJoints = pybullet.getNumJoints(kuka)

print(numJoints)

#set the gravity acceleration
pybullet.setGravity(0, 0, -9.8)

#step the simulation for 5 seconds
t_end = time.time() + 5
while time.time() < t_end:
  pybullet.stepSimulation()
  posAndOrn = pybullet.getBasePositionAndOrientation(kuka)
#   print(posAndOrn)

print("finished")
#remove all objects
pybullet.resetSimulation()

#disconnect from the physics server
pybullet.disconnect()



# Create a config class for your problem inheriting from the CBFConfig class
class MyCBFConfig(CBFConfig):
    def __init__(self):
        super().__init__(
            # Define the state and control dimensions
            n = 2, # [x, x_dot]
            m = 1, # [F_x]
            # Define control limits (if desired)
            u_min = None,
            u_max = None,
        )

    # Define the control-affine dynamics functions `f` and `g` for your system
    def f(self, z):
        A = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        return A @ z

    def g(self, u):
        mass = 1.0
        B = jnp.array([[0.0], [1.0 / mass]])
        return B @ u

    # Define the barrier function `h`
    # The *relative degree* of this system is 2, so, we'll use the h_2 method
    def h_2(self, z):
        x_min = 2.0
        x = z[0]
        return jnp.array([x - x_min])

config = MyCBFConfig()
cbf = CBF.from_config(config)

# Pseudocode
while True:
    z = get_state()
    z_des = get_desired_state()
    u_nom = nominal_controller(z, z_des)
    u = cbf.safety_filter(z, u_nom)
    apply_control(u)
    step()