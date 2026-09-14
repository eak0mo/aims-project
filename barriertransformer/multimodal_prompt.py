import ast
import os

import jax
import jax.numpy as jnp
import pandas as pd
from ollama import chat
from pydantic import BaseModel, Field


dynamic_motion_prompt = ""
sys_prompt = "You're given an image with a robot in the scene, the base of the robot is at [0,0,0] and the end-effector is at [0.2,0.2,0],follow the commands in the user prompts to return the desired output"
prompt_text = "given the attached image of a pybullet simulation with the franka panda emika robot, what are the coordinates of the end-effector (green cuboid) you can see in the image, estimate the coordinates based on the robots description?"

img_path = ["results/im/test.png"]


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
            {"role": "user", "content": prompt_text, "images": img_path},
        ],
        format=barrier_model.model_json_schema(),
        options={"temperature": 0},
    )

    validated = barrier_model.model_validate_json(response.message.content)
    return list(validated.model_dump().values())


def test_image_prompt(model_name, system_prompt, prompt_text):

    response = chat(
        model=model_name,
        messages=[
            # {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text, "images": img_path},
        ],
        # format=barrier_model.model_json_schema(),
        # options={"temperature": 0},
    )

    return response.message.content


output = test_image_prompt(
    model_name="gemma4:e2b", system_prompt=sys_prompt, prompt_text=prompt_text
)
print(output)
