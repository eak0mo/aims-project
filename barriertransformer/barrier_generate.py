from ollama import chat
from pydantic import BaseModel, Field

import jax
import jax.numpy as jnp


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

dynamic_motion_prompt = "A franka emika kuka robot is located at (0,0,0) as its base, with the end-effector ( which is not close to the point of the base) tracking a ball at (0.55,0,0.45), and moving in a sinusodial trajectory with amplitude (0.25,0,0) and frquency(5,0,0). Generate the end-effector barrier to contain both the path of ball and the robot together in all three dimensions."


def test(prompt=sys_prompt):
    print("hello World")
    print(prompt)


def create_prompt(base_pos, ee_pos, targ_pos, targ_amp, targ_freq):
    prompt = f"""
    A franka emika kuka robot is loaded into the environment with it's base at: {base_pos}. 
    The end-effector  is located at {ee_pos}, and it is tracking a ball starting at {targ_pos}, 
    and moving in a sinusodial trajectory with amplitude {targ_amp} and angular frequency {targ_freq}. 
    There are no collision objects to avoid.
    Generate a barrier that contains both the path of ball and the robot together in all three dimensions. 
    Make use of information given in the system prompt in designing this barrier
    """
    return prompt


def create_prompt_col(
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
    user_prompt: str = dynamic_motion_prompt, model_name: str = "llama3.1"
):
    barrier = extract_barrier(prompt_text=user_prompt, model_name=model_name)
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
