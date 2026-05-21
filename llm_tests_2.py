from barriertransformer import barrier_generate as bar
from barriertransformer import metrics as met
import jax.numpy as jnp
import argparse
import numpy as np
import os
import sys

def run_tests(model="llama3.1:70b"):
    model_name = model.replace(":", "_")

    # Custom collision box represented as a sphere
    cus_col_pos = [0.4, 0.3, 0.55]
    cus_col_rad = 0.1

    # 1. Dynamic Motion (no collisions)
    dynamic_motion = {
        "experiment_title": f"Dynamic_Motion_{model_name}",
        "ee_start": [0.24, 0.0, 0.429],
        "target_start": [0.37, 0.49, 0.45],
        "amplitude": [0.0, 0.14, 0.0],
        "frequency": [0.0, 0.59, 0.0],
        "collision_centers": None,
        "collision_radii": None,
        "output_dir": "results",
    }

    # 2. Multiple Safety Conditions (original + custom box)
    multiple_safety = {
        "experiment_title": f"Multiple_Safety_Conditions_{model_name}",
        "ee_start": [0.24, 0.0, 0.429],
        "target_start": [0.4, 0.0, 0.35],
        "amplitude": [0.0, 0.25, 0.0],
        "frequency": [0.0, 5.0, 0.0],
        "collision_centers": [[0.5, 0.5, 0.5], cus_col_pos],
        "collision_radii": [0.3, cus_col_rad],
        "output_dir": "results",
    }

    # 3. Cluttered Tabletop Custom (original + custom box)
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
        "experiment_title": f"Cluttered_Tabletop_Custom_{model_name}",
        "ee_start": [0.24, 0.0, 0.429],
        "target_start": [0.4, 0.0, 0.35],
        "amplitude": [0.0, 0.25, -0.15],
        "frequency": [0.0, 5.0, 2.5],
        "collision_centers": cluttered_custom_collision_pos + [cus_col_pos],
        "collision_radii": cluttered_custom_collision_radii + [cus_col_rad],
        "output_dir": "results",
    }

    # 4. Cluttered Tabletop (original + custom box)
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
        "experiment_title": f"Cluttered_Tabletop_{model_name}",
        "ee_start": [0.24, 0.0, 0.429],
        "target_start": [0.4, 0.0, 0.35],
        "amplitude": [0.0, 0.25, -0.15],
        "frequency": [0.0, 5.0, 2.5],
        "collision_centers": cluttered_collision_pos + [cus_col_pos],
        "collision_radii": cluttered_collision_radii + [cus_col_rad],
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

        # Generate new sinusoid prompt
        prompt = bar.create_prompt_col(
            ee_pos=job["ee_start"],
            targ_pos=job["target_start"],
            targ_amp=job["amplitude"],
            targ_freq=job["frequency"],
            collision_centers=job["collision_centers"],
            collision_radii=job["collision_radii"],
        )

        p_ver = "v2"
        print(f"\n  -> Prompt Version: {p_ver}")

        try:
            print(f"    Generating barriers with model: {model}")
            ee_min, ee_max, wb_min, wb_max = bar.generate_barrier(
                user_prompt=prompt, model_name=model, sin_traj=True
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
                prompt_version=p_ver,
            )

            output_path = os.path.join(job["output_dir"], "llm_res", model_name)
            os.makedirs(output_path, exist_ok=True)

            met.save_barriers_to_csv(sim_data, output_dir=output_path)

        except Exception as e:
            print(f"    Error during barrier generation under {p_ver}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM barrier tests version 2.")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:70b",
        help="Ollama model name to use (default: llama3.1:70b)",
    )
    args = parser.parse_args()
    run_tests(model=args.model)
