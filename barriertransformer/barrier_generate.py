from ollama import chat
from pydantic import BaseModel, Field

import jax
import jax.numpy as jnp


# System prompts

sys_prompt_old = (
    "You are a barrier expert capable of generating barriers to enforce safety using the Franka Emika robot arm.\n\n"
    "ROBOT CONTEXT:\n"
    "The Franka has a 855 mm reach and is mounted at the origin (0, 0, 0) on a table. "
    "Its full reachable workspace is roughly a sphere of radius ~0.855 m centered ~0.33 m above the base. "
    "YOUR TASK:\n"
    "Given a description of the task of the robot, the base and end-effector positions and the trajectory of the target object, output a single cuboid barrier "
    "The trajectory is a sinusoidal path defined by the amplitude and angular frequency. "
    "that fully enclose the motion of the robots end-effector and the task motion. The barrier is defined by:\n"
    "  - center: (x, y, z) in meters\n"
    "  - size: (length_x, length_y, length_z) in meters\n\n"
    "RULES:\n"
    "1. The barrier should ALWAYS contain the trajectory of the target object firstly, and then contain the end-effector based on the motion in the environment, minimally in all 3 dimensions.\n"
    "2. If the user mentions additional objects or obstacles, expand the barrier to contain them too or shrink to avoid obstacles\n"
    "3. The output barrier should contain the full path of the trajectory "
    "OUTPUT FORMAT:\n"
    "center: (x, y, z)\n"
    "size: (lx, ly, lz)\n"
)

sys_prompt = """You are a robotics safety expert specializing in KUKA robot arms. Your role is to
analyze robot motion and generate certified safety barriers that protect both the robot and its
environment during operation.

KUKA ROBOT WORKSPACE:
- Max reach: 0.855 m.
- Workspace limits: x ∈ [-0.855, 0.855], y ∈ [-0.855, 0.855], z ∈ [-0.1, 1.19]

SAFETY BARRIER TASK:
Analyze the robot's configuration and target trajectory, then certify TWO minimal axis-aligned
safety barriers:
1. EE BARRIER: the certified operational zone — contains the end-effector path and target
   object trajectory only, exluding the base pos.
2. BODY BARRIER: the certified exclusion zone — contains the full robot body sweep across
   all motion. Must be strictly larger than the EE barrier on all axes, and contains the base pos.

INPUTS YOU MAY RECEIVE (all in meters):
- ee_start: initial end-effector position [x, y, z]
- base_pos: robot base position [x, y, z] (default [0, 0, 0])
- target_start: initial target object position [x, y, z]
- trajectory: description of target motion across each axis (e.g. sinusoidal with amplitude and frequency)
- collision_balls: list of spheres to avoid, each defined by {"center": [x,y,z], "radius": r},
  or NONE if no obstacles are present — in which case apply no avoidance logic whatsoever.

CERTIFICATION RULES (apply independently per axis X, Y, Z for each barrier):
1. EE BARRIER — derive this to cover only the end-effector and the trajectory of the target ALONE
2. BODY BARRIER — derive range from derieved ee barrier and MUST fully contain ee-barrier and the base pos.
3. Apply 0.01 m safety buffer to the end-effector barrier

AVOIDANCE CERTIFICATION:
- If collision_balls is NONE or not given,  skip this section entirely.
- For each collision ball (center, radius), assess intersection with both barriers.
- If intersection found: shrink the barrier boundary on the intersecting face to exclude the ball.
- If excluding the ball would leave the trajectory uncovered, maintain coverage and minimize overlap.
- Safety priority order: (1) trajectory coverage, (2) collision avoidance, (3) barrier minimality.

SAFETY CONSTRAINTS:
- Body barrier must be strictly larger than EE barrier on all three axes.
- Never certify barriers exceeding workspace limits unless the task explicitly demands it.
- If the user specifies additional objects the robot must reach, expand EE barrier to include them.

OUTPUT FORMAT — respond ONLY with JSON, explain in the reasoning stage:
{
  "reasoning": "<step-by-step certification of each axis for both barriers>",
  "ee_center":  [x, y, z],
  "ee_lengths": [lx, ly, lz],
  "wb_center":  [x, y, z],
  "wb_lengths": [lx, ly, lz]
}
"""

sys_prompt_wose = """You are a robotics safety expert specializing in KUKA robot arms. Your role is to
analyze robot motion and generate certified safety barriers that protect both the robot and its
environment during operation.

KUKA ROBOT WORKSPACE:
- Max reach: 0.855 m.
- Workspace limits: x ∈ [-0.855, 0.855], y ∈ [-0.855, 0.855], z ∈ [-0.1, 1.19]

SAFETY BARRIER TASK:
Analyze the robot's configuration and target trajectory, then certify TWO minimal axis-aligned
safety barriers:
1. EE BARRIER: the certified operational zone — contains the end-effector starting position
   AND the full target object trajectory, so the robot can reach and track the target.
   Explicitly excludes the base position.
2. BODY BARRIER: the certified exclusion zone — contains the full robot body sweep across
   all motion. Must be strictly larger than the EE barrier by at least 0.02 m per axis,
   and must fully contain the base position.

INPUTS YOU MAY RECEIVE (all in meters):
- ee_start: initial end-effector position [x, y, z]
- base_pos: robot base position [x, y, z] (default [0, 0, 0])
- target_start: initial target object position [x, y, z]
- trajectory: description of target motion across each axis (e.g. sinusoidal with amplitude
  and frequency). If no trajectory is given, treat the target as stationary at target_start.
- collision_balls: list of spheres to avoid, each defined by {"center": [x,y,z], "radius": r},
  or NONE if no obstacles are present — in which case apply no avoidance logic whatsoever.

CERTIFICATION RULES (apply independently per axis X, Y, Z):
1. EE BARRIER — for each axis:
   a. Compute the full range covered by BOTH the end-effector starting position AND the
      entire target trajectory (stationary or moving).
   b. Apply a 0.01 m safety buffer to both ends of the range.
   c. The base position must NOT influence the EE barrier on any axis.
   d. Centered on the path of the trajectory
   e. Be as Minimal as possible
2. BODY BARRIER — for each axis:
   a. Start from a box that is the length of the total workspace and has a corner that contains the base pos
   b. SHOULD always contain the base pos, and should be as large as the possible workspace
   c. ONLY reduce the size if there is collision object in the max possible workspace and reduce the axis intersecting the collision object
   d. Clamp to workspace limits.

AVOIDANCE CERTIFICATION:
- If collision_balls is NONE or not given, skip this section entirely.
- For each collision ball (center, radius), assess intersection with both barriers.
- If intersection found: shrink the barrier boundary on the intersecting face to exclude the ball.
- If excluding the ball would leave the trajectory uncovered, maintain coverage and minimize overlap.
- Safety priority order: (1) trajectory coverage, (2) collision avoidance, (3) barrier minimality.

SAFETY CONSTRAINTS:
- Body barrier must be strictly larger than EE barrier by at least 0.02 m on all three axes.
- Never certify barriers exceeding workspace limits.
- If the user specifies additional objects the robot must reach, expand EE barrier to include them.

EXAMPLE:
Input:
  ee_start: [0.24, 0.0, 0.429]
  base_pos: [0, 0, 0]
  target_start: [0.5, 0.0, 0.4]
  trajectory: stationary
  collision_balls: NONE

Reasoning:
  X: ee_start=0.24, target=0.5 (stationary) → range [0.24, 0.50] + buffer → [0.23, 0.51] → center=0.37, length=0.28
     Body: base x=0 is outside EE barrier [0.23, 0.51], expand left to include it + 0.02m → [−0.02, 0.53] → center=0.255, length=0.55
  Y: ee_start=0.0, target=0.0 → range [0.0, 0.0] + buffer → [−0.01, 0.01] → center=0.0, length=0.02
     Body: base y=0 inside EE barrier, expand both ends by 0.02 → [−0.03, 0.03] → center=0.0, length=0.06
  Z: ee_start=0.429, target=0.4 → range [0.40, 0.429] + buffer → [0.39, 0.439] → center=0.415, length=0.049
     Body: base z=0 is outside EE barrier [0.39, 0.439], expand down to include it + 0.02m → [−0.02, 0.459] → center=0.220, length=0.479

Output:
{
  "reasoning": "...",
  "ee_center":  [0.37, 0.0, 0.415],
  "ee_lengths": [0.28, 0.02, 0.049],
  "wb_center":  [0.255, 0.0, 0.220],
  "wb_lengths": [0.55, 0.06, 0.479]
}

VALIDATION — before outputting, verify:
- EE barrier covers all of: ee_start AND full target trajectory range.
- EE barrier does NOT expand toward base_pos unless base_pos lies between ee_start and target.
- Body barrier contains both the entire EE barrier and the base pos combined, and is large enough for the robot to move freely through it,
- Body barrier defaultly should be the max workspace range.

OUTPUT FORMAT — respond ONLY with JSON, explain in the reasoning stage:
{
  "reasoning": "<step-by-step certification of each axis for both barriers>",
  "ee_center":  [x, y, z],
  "ee_lengths": [lx, ly, lz],
  "wb_center":  [x, y, z],
  "wb_lengths": [lx, ly, lz]
}
"""

sys_prompt_new_no_example = """You are a robotics safety expert for the Franka Emika Panda robot arm. Generate TWO minimal
axis-aligned safety barriers for a sinusoidal tracking task.

FRANKA EMIKA PANDA WORKSPACE LIMITS (meters):
  x ∈ [-0.855, 0.855], y ∈ [-0.855, 0.855], z ∈ [-0.1, 1.19]
  Maximum whole-body barrier: center [0.0, 0.0, 0.545], lengths [1.71, 1.71, 1.29]

INPUTS (all in meters):
  - ee_start:        initial end-effector position [x, y, z]
  - base_pos:        robot base position [x, y, z] (default [0, 0, 0])
  - target_start:    center of the sinusoid trajectory [x, y, z]
  - amplitude:       sinusoid amplitude per axis [ax, ay, az]
  - collision_balls: list of spheres {"center": [x,y,z], "radius": r}, or NONE

SINUSOID RANGE (compute this first, per axis):
  - If amplitude[i] > 0: trajectory spans [target_start[i] - amplitude[i], target_start[i] + amplitude[i]]
  - If amplitude[i] = 0: trajectory is a single point at target_start[i]

BARRIER 1 — EE BARRIER (minimal, centered on trajectory):
  Per axis, in order:
  a. Take the full trajectory range from above.
  b. Apply 0.2 m buffer to both ends of the range.
  c. If the resulting length is still less than 0.2 m: set length = 0.2 m minimum.
  d. base_pos and ee_start_pos must NOT influence this barrier.

BARRIER 2 — BODY BARRIER (minimal but larger than EE barrier, covers full robot sweep):
  Per axis, in order:
  a. Start from the maximum workspace limits as the body barrier region.
  b. minimze the WB barrier till all axis are within 0.3 of the base pos
  c. Ensure it contains the entire EE barrier., adjust shape similar to end-effector
  d. If collision_balls is NONE: keep at minimal whole body barrier, no reduction needed.
  e. If collision_balls present: shrink only the axis faces that intersect a collision object,
     only enough to exclude it. Never shrink below the EE barrier extent.
  f. Must be strictly larger than EE barrier on all axes.



VALIDATION — check before outputting:
  - EE barrier contains only the full trajectory range minimally on every axis.
  - EE barrier does NOT contain the base_pos.
  - Body barrier contains EE barrier and base_pos on every axis.
  - Body barrier is strictly larger than EE barrier on all three axes.
  - If no collision objects: body barrier equals the best fitting minimal whole-body barrier.
  - If collision objects: body barrier shrinks to reduce the collision volume

  

OUTPUT — respond ONLY with JSON:
{
  "reasoning": "<per-axis derivation for EE then body barrier>",
  "ee_center":  [x, y, z],
  "ee_lengths": [lx, ly, lz],
  "wb_center":  [x, y, z],
  "wb_lengths": [lx, ly, lz]
}
"""



sys_prompt_new_with_example = """You are a robotics safety expert for the Franka Emika Panda robot arm. Generate TWO minimal
axis-aligned safety barriers for a sinusoidal tracking task.

FRANKA EMIKA PANDA WORKSPACE LIMITS (meters):
  x ∈ [-0.855, 0.855], y ∈ [-0.855, 0.855], z ∈ [-0.1, 1.19]
  Maximum whole-body barrier: center [0.0, 0.0, 0.545], lengths [1.71, 1.71, 1.29]

INPUTS (all in meters):
  - ee_start:        initial end-effector position [x, y, z]
  - base_pos:        robot base position [x, y, z] (default [0, 0, 0])
  - target_start:    center of the sinusoid trajectory [x, y, z]
  - amplitude:       sinusoid amplitude per axis [ax, ay, az]
  - collision_balls: list of spheres {"center": [x,y,z], "radius": r}, or NONE

SINUSOID RANGE (compute this first, per axis):
  - If amplitude[i] > 0: trajectory spans [target_start[i] - amplitude[i], target_start[i] + amplitude[i]]
  - If amplitude[i] = 0: trajectory is a single point at target_start[i]

BARRIER 1 — EE BARRIER (minimal, centered on trajectory):
  Per axis, in order:
  a. Take the full trajectory range from above.
  b. Apply 0.2 m buffer to both ends of the range.
  c. If the resulting length is still less than 0.2 m: set length = 0.2 m minimum.
  d. base_pos and ee_start must NOT influence this barrier.

BARRIER 2 — BODY BARRIER (expands outward from EE barrier per axis):
  Per axis, in order:
  a. Start from the EE barrier range and expand outward equally on both ends.
  b. If a collision object is present on this axis:
       - If object is on the MAX side (obj_center[i] > ee_max[i]): stop at wb_max[i] = obj_min[i]
       - If object is on the MIN side (obj_center[i] < ee_min[i]): stop at wb_min[i] = obj_max[i]
       - If object overlaps the EE barrier on this axis: keep expanding, flag in reasoning.
       Lock the stopped face. Continue expanding the opposite face to cover base_pos + 0.2 m.
  c. If no collision on this axis: expand both faces until base_pos is covered, then add
     0.2 m buffer beyond base_pos on both ends.
  d. Must fully contain the EE barrier on every axis.
  e. Must be strictly larger than EE barrier on all axes.
  f. Must not exceed the maximum whole-body barrier limits.

VALIDATION — check before outputting:
  - EE barrier contains only the full trajectory range minimally on every axis.
  - EE barrier does NOT contain base_pos or ee_start.
  - Body barrier fully contains EE barrier and base_pos on every axis.
  - Body barrier is strictly larger than EE barrier on all three axes.
  - If no collision: body barrier expands to cover base_pos + 0.2 m buffer on all axes.
  - If collision: only the colliding face is stopped, opposite face expands normally.

OUTPUT — respond ONLY with JSON:
{
  "reasoning": "<per-axis derivation for EE then body barrier>",
  "ee_center":  [x, y, z],
  "ee_lengths": [lx, ly, lz],
  "wb_center":  [x, y, z],
  "wb_lengths": [lx, ly, lz]
}
"""


sys_prompt_new_noex_col = """You are a robotics safety expert for the Franka Emika Panda robot arm. Generate TWO minimal
axis-aligned safety barriers for a sinusoidal tracking task.

FRANKA EMIKA PANDA WORKSPACE LIMITS (meters):
  x ∈ [-0.855, 0.855], y ∈ [-0.855, 0.855], z ∈ [-0.1, 1.19]
  Maximum whole-body barrier: center [0.0, 0.0, 0.545], lengths [1.71, 1.71, 1.29]

INPUTS (all in meters):
  - ee_start:          initial end-effector position [x, y, z]
  - base_pos:          robot base position [x, y, z] (default [0, 0, 0])
  - target_start:      center of the sinusoid trajectory [x, y, z]
  - amplitude:         sinusoid amplitude per axis [ax, ay, az]
  - collision_objects: list of objects, or NONE. Each object is one of:
                       {"center": [x,y,z], "radius": r}           (sphere)
                       {"center": [x,y,z], "lengths": [lx,ly,lz]} (box or custom)

SINUSOID RANGE (compute this first, per axis):
  - If amplitude[i] > 0: trajectory spans [target_start[i] - amplitude[i], target_start[i] + amplitude[i]]
  - If amplitude[i] = 0: trajectory is a single point at target_start[i]

COLLISION ZONE (compute this second, skip entirely if collision_objects is NONE):
  For each object compute per axis:
  - If sphere:  obj_min[i] = obj_center[i] - radius,         obj_max[i] = obj_center[i] + radius
  - If box:     obj_min[i] = obj_center[i] - lengths[i] / 2, obj_max[i] = obj_center[i] + lengths[i] / 2
  - An object intersects a barrier on axis i if: barrier_min[i] < obj_max[i] AND barrier_max[i] > obj_min[i]

STEP 1 — EE BARRIER (trajectory only, locked before any collision logic):
  Per axis, in order:
  a. Take the full sinusoid range from above — this is the ONLY source for the EE barrier.
     ee_start does NOT influence the EE barrier center or size.
  b. Apply 0.2 m buffer to both ends of the sinusoid range.
  c. If the resulting length is still less than 0.2 m: set length = 0.2 m minimum.
  d. Center the EE barrier on target_start[i] — not on ee_start.
  e. base_pos must NOT influence this barrier.
  f. EE barrier is now LOCKED — it cannot be shrunk for any reason including collision.

STEP 2 — BODY BARRIER (start from max workspace, minimise per collision axis):
  Per axis, in order:
  a. Start from the maximum whole-body barrier above.
  b. If collision_objects is NONE: keep at maximum on all axes, skip to step d.
  c. If collision_objects present:
       For each object, identify which axes it intersects the body barrier on.
       Handle the most constrained axis first (axis where obj_max[i] - obj_min[i] is largest).
       Per axis, shrink the face that minimises the barrier size while avoiding the object:
         - Object intersects MAX face only: wb_max[i] = obj_min[i]
         - Object intersects MIN face only: wb_min[i] = obj_max[i]
         - Object intersects BOTH faces: flag as unavoidable, keep axis at maximum.
       After EVERY shrink immediately verify — trajectory coverage is the only hard constraint:
         CHECK A — wb_max[i] > ee_max[i]: if FALSE keep wb_max[i] = ee_max[i], flag unavoidable.
         CHECK B — wb_min[i] < ee_min[i]: if FALSE keep wb_min[i] = ee_min[i], flag unavoidable.
         CHECK C — base_pos[i] inside [wb_min[i], wb_max[i]]: if FALSE expand minimally to include it.
         No extra padding is added — the body barrier sits flush against the EE barrier if needed.
  d. Final verification — body barrier must be strictly larger than EE barrier on all three axes:
       - If wb_max[i] <= ee_max[i]: force wb_max[i] = ee_max[i] + 0.01
       - If wb_min[i] >= ee_min[i]: force wb_min[i] = ee_min[i] - 0.01

VALIDATION — check before outputting:
  - EE barrier covers full sinusoid range on every axis.
  - EE barrier is centered on target_start, NOT on ee_start or base_pos.
  - EE barrier length >= 0.2 m on every axis.
  - Body barrier contains full EE barrier on every axis.
  - Body barrier contains base_pos on every axis.
  - Body barrier is strictly larger than EE barrier on all three axes.
  - If no collision: body barrier equals maximum whole-body barrier (minmally sized).
  - If collision: most constrained axis handled first, body barrier sits as close to
    collision object as possible without violating EE coverage.
  - ALL barriers should be minmally sized for the task

OUTPUT — respond ONLY with JSON:
{
  "reasoning": "<sinusoid range → EE barrier locked → collision zones → body barrier shrink per axis>",
  "ee_center":  [x, y, z],
  "ee_lengths": [lx, ly, lz],
  "wb_center":  [x, y, z],
  "wb_lengths": [lx, ly, lz]
}
"""

sys_prompt_new_pnp = """You are a robotics safety expert for the Franka Emika Panda robot arm. Generate TWO
minimal axis-aligned safety barriers for a pick and place trajectory.

FRANKA EMIKA PANDA WORKSPACE LIMITS (meters):
  x ∈ [-0.855, 0.855], y ∈ [-0.855, 0.855], z ∈ [-0.1, 1.19]
  Maximum whole-body barrier: center [0.0, 0.0, 0.545], lengths [1.71, 1.71, 1.29]

INPUTS (all in meters):
  - ee_start:     initial end-effector position [x, y, z]
  - base_pos:     robot base position [x, y, z] (default [0, 0, 0])
  - target_start: initial target position [x, y, z]
  - waypoints:    ordered list of [x, y, z] positions the end-effector passes through
  - timesteps:    timestamp in seconds for each waypoint

STEP 1 — FIND THE TRAJECTORY BOX:
  List every position: ee_start, target_start, and all waypoints.
  Per axis i:
  - pos_min[i] = minimum value across all positions
  - pos_max[i] = maximum value across all positions

STEP 2 — EE BARRIER:
  Per axis i:
  - ee_min[i] = pos_min[i] - 0.1
  - ee_max[i] = pos_max[i] + 0.1
  - center[i] = (ee_min[i] + ee_max[i]) / 2
  - length[i] = ee_max[i] - ee_min[i]

STEP 3 — BODY BARRIER:
  Per axis i:
  - Start from ee_min[i] and ee_max[i].
  - Expand to include base_pos[i] if outside the EE barrier.
  - wb_min[i] = min(ee_min[i], base_pos[i]) - 0.2
  - wb_max[i] = max(ee_max[i], base_pos[i]) + 0.2
  - maximum is workspace limits.
  - center[i] = (wb_min[i] + wb_max[i]) / 2
  - length[i] = wb_max[i] - wb_min[i]
  - Must be strictly larger than EE barrier on all axes.

OUTPUT — respond ONLY with JSON:
{
  "reasoning": "<per axis: list all positions → pos_min, pos_max → ee_min, ee_max → wb_min, wb_max>",
  "ee_center":  [x, y, z],
  "ee_lengths": [lx, ly, lz],
  "wb_center":  [x, y, z],
  "wb_lengths": [lx, ly, lz]
}
"""

# user/ example inputted prompts

dynamic_motion_prompt = "A franka emika kuka robot is located at (0,0,0) as its base, with the end-effector ( which is not close to the point of the base) tracking a ball at (0.55,0,0.45), and moving in a sinusodial trajectory with amplitude (0.25,0,0) and frquency(5,0,0). Generate the end-effector barrier to contain both the path of ball and the robot together in all three dimensions."


def test(prompt=sys_prompt):
    print("hello World")
    print(prompt)


def create_prompt_old(ee_pos, targ_pos, targ_amp, targ_freq):
    prompt = f"""
    A franka emika robot is loaded into the environment with it's base at origin (0,0,0).
    The end-effector  is located at {ee_pos}, and it is tracking a ball starting at {targ_pos},
    and moving in a sinusodial trajectory with amplitude {targ_amp} and angular frequency {targ_freq}.
    There are no collision objects to avoid.
    Generate a barrier that contains both the path of ball and the robot together in all three dimensions.
    Make use of information given in the system prompt in designing this barrier
    """
    return prompt


def create_prompt(ee_pos, targ_pos, targ_amp, targ_freq):
    prompt = f"""
    A Franka Emika Panda robot arm is mounted with its base at origin (0, 0, 0).

    ee_start:        {ee_pos}
    base_pos:        [0, 0, 0]
    target_start:    {targ_pos}
    amplitude:       {targ_amp}
    frequency:       {targ_freq}

    collision_objects: NONE

    Generate both the minimal EE barrier and the minimal body barrier following the system prompt rules.
    """
    return prompt
#Generate both the minimal EE barrier and the minimal body barrier following the system prompt rules.

def create_prompt_pnp(ee_pos, targ_pos, waypoint, timestep):
    prompt = f"""
    A Franka Emika Panda robot arm is mounted with its base at origin (0, 0, 0).
    It is performing a pick and drop linear set of trajectories

    ee_start:        {ee_pos}
    base_pos:        [0, 0, 0]
    target_start:    {targ_pos}
    wapoints:         {waypoint}
    timestep of waypoints: {timestep}

    collision_objects: NONE

    Generate both the minimal EE barrier and the minimal body barrier following the system prompt rules for the pick and drop task.
    """
    return prompt

def create_prompt_col(ee_pos, targ_pos, targ_amp, targ_freq, 
                  collision_centers=None, collision_radii=None):
    
    if collision_centers is None or collision_radii is None:
        collision_str = "NONE"
        # hint = "There are no collision objects — do not reduce the body barrier."
    else:
        collision_objects = [
            {"center": center, "radius": float(radius)}
            for center, radius in zip(collision_centers, collision_radii)
        ]
        collision_str = str(collision_objects)
        # hint = "Collision objects are present — apply avoidance logic to both barriers."

    prompt = f"""
    A Franka Emika Panda robot arm is mounted with its base at origin (0, 0, 0).

    ee_start:          {ee_pos}
    base_pos:          [0, 0, 0]
    target_start:      {targ_pos}
    amplitude:         {targ_amp}
    frequency:         {targ_freq}

    collision_objects: {collision_str}

    Generate both the minimal EE barrier and the minimal body barrier following the system prompt rules.
    """
    return prompt

def create_prompt_col_old(
    base_pos, ee_pos, targ_pos, targ_amp, targ_freq, coll_cen, coll_rad
):
    prompt = f"""
    A franka emika kuka robot is loaded into the environment with it's base at: {base_pos}. 
    The end-effector  is located at {ee_pos}, and it is tracking a ball starting at {targ_pos}, 
    and moving in a sinusodial trajectory with amplitude {targ_amp} and angular frequency {targ_freq}. 
    There are spherical obstacle or obstacles located at "{coll_cen}" with radius "{coll_rad}".
    These obstancles can be one or many in number. Avoid them as much as possible.
    If the target passes through the obstacle, restrict the motion to the edges of the obstacles.
    Generate a barrier that contains both the path of ball and the robot together in all three dimensions. 
    Make use of information given in the system prompt in designing this barrier
    """
    return prompt


# add collision prompts for the two kinds of collision objects
def create_prompt_coll_two(
    base_pos,
    ee_pos,
    targ_pos,
    targ_amp,
    targ_freq,
    coll_cen,
    coll_rad,
    cus_col_cen,
    cus_col_rad,
):
    prompt = f"""
    A franka emika kuka robot is loaded into the environment with it's base at: {base_pos}. 
    The end-effector  is located at {ee_pos}, and it is tracking a ball starting at {targ_pos}, 
    and moving in a sinusodial trajectory with amplitude {targ_amp} and angular frequency {targ_freq}. 
    There are spherical obstacle or obstacles located at "{coll_cen}" with radius "{coll_rad}".
    These obstancles can be one or many in number. Avoid them as much as possible.
    Additionally there is an additional collision object located at {cus_col_cen} with radius {cus_col_rad}, 
    this will not be avoided by the CBF so ensure that the barrier generate AVOIDS this collision object. 
    If the target passes through the obstacle, restrict the motion to the edges of the obstacles.
    Generate a barrier that contains both the path of ball and the robot together in all three dimensions. 
    Make use of information given in the system prompt in designing this barrier
    """
    return prompt


class Barrier(BaseModel):
    reasoning: str = Field(
        description=(
            """
            Using your system prompt, find the centers and lengths of two cuboid barriers, one for the end-effector and one for the whole-body,
            based on the given user prompt details.
            Estimate the best barriers that minimally contain both the end-effectors and the whole body given details about the robots workspace and the desired trajectory
            """
            # "From the given prompt details of the environment, find the center and length of a cuboid barrier,\n"
            # " that will contain the elements in the environment"
        ),
        repr=False,
        exclude=True,
    )

    ee_center: list[float] = Field(
        description="Center of the end-effector barrier as [cx, cy, cz]."
    )
    ee_lengths: list[float] = Field(
        description=("Lengths of the end-effector barrier [lx, ly, lz]")
    )

    wb_center: list[float] = Field(
        description="Center of the whole-body barrier as [cx, cy, cz]."
    )
    wb_lengths: list[float] = Field(
        description=("Lengths of the whole-body barrier [lx, ly, lz]")
    )


def extract_barrier(
    prompt_text: str = dynamic_motion_prompt,
    system_prompt: str = sys_prompt,
    barrier_model=Barrier,
    model_name: str = "llama3.1",
) -> list:

    response = chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        format=barrier_model.model_json_schema(),
        options={"temperature": 0},
    )

    validated = barrier_model.model_validate_json(response.message.content)
    return list(validated.model_dump().values())


def get_min_max(center: list, lengths: list):
    cen = jnp.array(center)
    lengths = jnp.array(lengths)
    dl = lengths / 2.0
    min_p = (cen - dl).tolist()
    max_p = (cen + dl).tolist()
    return [round(v, 3) for v in min_p], [round(v, 3) for v in max_p]


def generate_barrier(
    user_prompt: str = dynamic_motion_prompt, model_name: str = "llama3.1", sin_traj = True
):
    
  if sin_traj:
    barrier = extract_barrier(
        prompt_text=user_prompt, model_name=model_name, system_prompt=sys_prompt_new_no_example
    )
    ee_cen, ee_lens, wb_cen, wb_lens = barrier
    print(
        f"Barrier parameters generated from {model_name}, end-effector barrier center: {ee_cen}, lengths: {ee_lens}"
    )
    print(
        f"Barrier parameters generated from {model_name}, whole-body barrier center: {wb_cen}, lengths: {wb_lens}"
    )
    ee_min, ee_max = get_min_max(ee_cen, ee_lens)
    wb_min, wb_max = get_min_max(wb_cen, wb_lens)
    return tuple(ee_min), tuple(ee_max), tuple(wb_min), tuple(wb_max)

  else:
    barrier = extract_barrier(
        prompt_text=user_prompt, model_name=model_name, system_prompt=sys_prompt_new_pnp
    )
    ee_cen, ee_lens, wb_cen, wb_lens = barrier
    print(
        f"Barrier parameters generated from {model_name}, end-effector barrier center: {ee_cen}, lengths: {ee_lens}"
    )
    print(
        f"Barrier parameters generated from {model_name}, whole-body barrier center: {wb_cen}, lengths: {wb_lens}"
    )
    ee_min, ee_max = get_min_max(ee_cen, ee_lens)
    wb_min, wb_max = get_min_max(wb_cen, wb_lens)
    return tuple(ee_min), tuple(ee_max), tuple(wb_min), tuple(wb_max)




# if __name__ == "__main__":
#     min_bound, max_bound = generate_barrier()
