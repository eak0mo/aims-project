import pybullet_data
import pybullet as p
import time
# import jax.numpy as jnp

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
# p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,-10)
planeId = p.loadURDF("plane.urdf",0,0,0)
startPos = [0,0,0.5]
startOrientation = p.getQuaternionFromEuler([0,0,0])
boxId = p.loadURDF("r2d2.urdf",startPos, startOrientation)


#set the center of mass frame (loadURDF sets base link frame) startPos/Orn
p.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
#reset base velocity
p.resetBaseVelocity(boxId,[0,0,0],[0,0,0])


#get num_joints
print(p.getNumJoints(boxId))
for i in range (1000):
    p.stepSimulation()
    time.sleep(1./240.)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
print(cubePos,cubeOrn)
p.disconnect()

