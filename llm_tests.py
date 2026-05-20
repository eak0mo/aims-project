from barriertransformer import barrier_generate as bar

from barriertransformer import visualization as viz

from barriertransformer import metrics as met

import jax.numpy as jnp
import argparse

import numpy as np
import os
import sys


def run_tests(model="llama3.1:70b"):

    # Examples from codebase: dynamic motion, multiple safety condition, cluttered tabletop, cluttered tabletop custom
    dynamic_motion = {
        "experiment_title": "Dynamic_Motion_res_19_05",
        "ee_start": [0.24, 0.0, 0.429],
        "target_start": [0.37, 0.49, 0.45],
        "amplitude": [0.0, 0.14, 0.0],
        "frequency": [0.0, 0.59, 0.0],
        "collision_centers": None,
        "collision_radii": None,
        "output_dir": "results",
    }

    # Multiple Safety Conditions
    multiple_safety = {
        "experiment_title": "Multiple_Safety_Conditions_14_05",
        "ee_start": [0.24, 0.0, 0.429],
        "target_start": [0.4, 0.0, 0.35],
        "amplitude": [0.0, 0.25, 0.0],
        "frequency": [0.0, 5.0, 0.0],
        "collision_centers": [[0.5, 0.5, 0.5]],
        "collision_radii": [0.3],
        "output_dir": "results",
    }

    # Cluttered Tabletop
    np.random.seed(0)

    max_num_bodies = 50
    all_collision_pos = jnp.array(
        np.random.uniform(
            low=[0.2, -0.4, 0.1], high=[0.8, 0.4, 0.3], size=(max_num_bodies, 3)
        )
    )
    all_collision_radii = jnp.array(
        np.random.uniform(low=0.01, high=0.1, size=(max_num_bodies,))
    )
    cluttered_collision_pos = jnp.atleast_2d(all_collision_pos[:25]).tolist()

    cluttered_collision_radii = all_collision_radii[:25].tolist()

    cluttered_tabletop = {
        "experiment_title": "Cluttered_Tabletop",
        "ee_start": [0.24, 0.0, 0.429],
        "target_start": [0.4, 0.0, 0.35],
        "amplitude": [0.0, 0.25, -0.15],
        "frequency": [0.0, 5.0, 2.5],
        "collision_centers": cluttered_collision_pos,
        "collision_radii": cluttered_collision_radii,
        "output_dir": "results",
    }

    # Cluttered Tabletop Custom
    np.random.seed(0)

    max_num_bodies_custom = 5
    all_collision_pos_custom = jnp.array(
        np.random.uniform(
            low=[0.2, -0.4, 0.1], high=[0.8, 0.4, 0.3], size=(max_num_bodies_custom, 3)
        )
    )
    all_collision_radii_custom = jnp.array(
        np.random.uniform(low=0.07, high=0.1, size=(max_num_bodies_custom,))
    )
    cluttered_custom_collision_pos = jnp.atleast_2d(
        all_collision_pos_custom[:3]
    ).tolist()

    cluttered_custom_collision_radii = all_collision_radii_custom[:3].tolist()

    cluttered_tabletop_custom = {
        "experiment_title": "Cluttered_Tabletop_Custom",
        "ee_start": [0.24, 0.0, 0.429],
        "target_start": [0.4, 0.0, 0.35],
        "amplitude": [0.0, 0.25, -0.15],
        "frequency": [0.0, 5.0, 2.5],
        "collision_centers": cluttered_custom_collision_pos,
        "collision_radii": cluttered_custom_collision_radii,
        "output_dir": "results",
    }

    all_jobs = [
        dynamic_motion,
        multiple_safety,
        cluttered_tabletop_custom,
        cluttered_tabletop,
    ]

    for job in all_jobs:
        print(f"\n==================================================")
        print(f"Running Experiment: {job['experiment_title']}")
        print(f"==================================================")

        # Generate legacy prompt with one-shot example (wose)
        if job["collision_centers"] is None or len(job["collision_centers"]) == 0:
            prompt = bar.create_prompt_old(
                ee_pos=job["ee_start"],
                targ_pos=job["target_start"],
                targ_amp=job["amplitude"],
                targ_freq=job["frequency"],
            )
        else:
            prompt = bar.create_prompt_col_old(
                base_pos=[0, 0, 0],
                ee_pos=job["ee_start"],
                targ_pos=job["target_start"],
                targ_amp=job["amplitude"],
                targ_freq=job["frequency"],
                coll_cen=job["collision_centers"],
                coll_rad=job["collision_radii"],
            )

        print(f"\n  -> Prompt Version: wose")

        try:
            print("    Generating barriers with model...")
            ee_min, ee_max, wb_min, wb_max = bar.generate_barrier_old(
                user_prompt=prompt, model_name=model, ver01=False
            )

            print(f"    Generated barriers successfully:")
            print(f"      EE Min/Max: {ee_min} / {ee_max}")
            print(f"      WB Min/Max: {wb_min} / {wb_max}")

            # Save to CSV
            sim_data = met.SimulationData(
                dt=0.001,
                time=jnp.array([]),
                q_traj=jnp.array([]),
                u_actual=jnp.array([]),
                u_nominal=jnp.array([]),
                pos_min=jnp.array(ee_min),
                pos_max=jnp.array(ee_max),
                wb_min=jnp.array(wb_min),
                wb_max=jnp.array(wb_max),
                experiment_title=job["experiment_title"],
                prompt_version="wose",
            )

            met.save_barriers_to_csv(
                sim_data, output_dir=os.path.join(job["output_dir"], "llm_res")
            )

        except Exception as e:
            print(f"    Error during barrier generation: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM barrier tests.")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:70b",
        help="Ollama model name to use (default: llama3.1:70b)",
    )
    args = parser.parse_args()
    run_tests(model=args.model)
