from ollama import chat
from pydantic import BaseModel, Field

import jax
import jax.numpy as jnp


class Barrier(BaseModel):
    reasoning: str = Field(
        description=(
            "From the given prompt details of the environment, find the center and length of a cuboid barrier,\n"
            " that will contain the elements in the environment"
        ),
        repr=False,
        exclude=True,
    )

    center: list[float] = Field(description="Centre of the shape as [cx, cy, cz].")
    lengths: list[float] = Field(
        description=("CUBOID — full side lengths as [sx, sy, sz]")
    )


sys_prompt = (
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

dynamic_motion_prompt = "A franka emika kuka robot is located at (0,0,0) as its base, with the end-effector ( which is not close to the point of the base) tracking a ball at (0.55,0,0.45), and moving in a sinusodial trajectory with amplitude (0.25,0,0) and frquency(5,0,0). Generate a barrier to contain both the path of ball and the robot together in all three dimensions and the robot has workspace above 0"


def test(prompt=sys_prompt):
    print("hello World")
    print(prompt)


def create_prompt(base_pos, ee_pos, targ_pos, targ_amp, targ_freq):
    prompt = f"""
    A franka emika kuka robot is loaded into the environment with it's base at: {base_pos}. 
    The end-effector  is located at {ee_pos}, and it is tracking a ball starting at {targ_pos}, and moving in a sinusodial trajectory with amplitude {targ_amp} and angular frequency {targ_freq}. Generate a barrier that contains both the path of ball and the robot together in all three dimensions. 
    Make use of the system prompt in designing this barrier
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
    print(
        f"Barrier parameters generated from {model_name}, center: {barrier[0]}, lengths: {barrier[1]}"
    )
    min, max = get_min_max(barrier[0], barrier[1])
    return tuple(min), tuple(max)


# if __name__ == "__main__":
#     min_bound, max_bound = generate_barrier()
