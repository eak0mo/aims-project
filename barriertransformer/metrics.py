from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import os
import csv
from functools import partial

@dataclass
class SimulationData:
    dt: float
    time: np.ndarray                 
    
    # Kinematics & Control
    q_traj: np.ndarray               
    u_actual: np.ndarray             
    u_nominal: np.ndarray            
    p_actual: np.ndarray = None      # End-effector actual positions (N, 3)
    p_target: np.ndarray = None      # End-effector target positions (N, 3)
    
    # Barrier Data (Safe Set)
    pos_min: np.ndarray = None        # End-effector barrier min bounds
    pos_max: np.ndarray = None        # End-effector barrier max bounds
    wb_min: np.ndarray = None         # Whole-body barrier min bounds
    wb_max: np.ndarray = None         # Whole-body barrier max bounds
    h_val: np.ndarray = None         
    
    # Joint Spheres (bubbles wrapping the robot's own joints/links for Whole-Body SVR)
    joint_spheres: np.ndarray = None # Shape: (T, num_spheres, 3)
    joint_sphere_radii: np.ndarray = None      # Shape: (num_spheres,)
    
    # Collision Spheres (external environment obstacles evaluated against safe set bounds for CIA)
    collision_spheres: np.ndarray = None # Shape: (num_obstacles, 3) or (T, num_obstacles, 3)
    collision_sphere_radii: np.ndarray = None  # Shape: (num_obstacles,)
    
    # Metadata for CSV
    experiment_title: str = "default_experiment"
    prompt_version: str = "v1"
    date: str = datetime.now().strftime("%Y-%m-%d")


@partial(jax.jit, static_argnames=["steps_10s"])
def compute_mte(p_actual: jnp.ndarray, p_target: jnp.ndarray, steps_10s: int):
    """Mean Tracking Error (MTE): Calculates the average and std dev L2 norm between actual and target end-effector positions over all time and the last 10 seconds."""
    errors = jnp.linalg.norm(p_actual - p_target, axis=1)
    mean_all, std_all = jnp.mean(errors), jnp.std(errors)
    
    # Slicing for the last 10 seconds using a compile-time static slice length
    errors_last_10s = errors[-steps_10s:]
    mean_last_10s, std_last_10s = jnp.mean(errors_last_10s), jnp.std(errors_last_10s)
    
    return mean_all, std_all, mean_last_10s, std_last_10s


@jax.jit
def compute_svr_ee(p_actual: jnp.ndarray, box_min: jnp.ndarray, box_max: jnp.ndarray):
    """Safety Violation Rate (SVR) EE: Percentage of time steps where the end-effector position exits the safe barrier bounds."""
    outside_min = p_actual < box_min
    outside_max = p_actual > box_max
    violation_per_timestep = jnp.any(outside_min | outside_max, axis=1)
    return (jnp.sum(violation_per_timestep) / len(violation_per_timestep)) * 100.0


@jax.jit
def compute_svr_wb(joint_spheres: jnp.ndarray, joint_sphere_radii: jnp.ndarray, box_min: jnp.ndarray, box_max: jnp.ndarray):
    """Safety Violation Rate (SVR) WB: Percentage of time steps where any whole-body joint sphere exits the safe container bounds."""
    outside_min = (joint_spheres - joint_sphere_radii[:, None]) < box_min
    outside_max = (joint_spheres + joint_sphere_radii[:, None]) > box_max
    
    violation_per_timestep = jnp.any(outside_min | outside_max, axis=(1, 2))
    return (jnp.sum(violation_per_timestep) / len(violation_per_timestep)) * 100.0


@jax.jit
def compute_cia(collision_spheres: jnp.ndarray, collision_sphere_radii: jnp.ndarray, box_min: jnp.ndarray, box_max: jnp.ndarray):
    """Collision Intersection Area (CIA): Cumulative and peak intersection volume between environmental collision spheres and safe set bounds."""
    clamped = jnp.clip(collision_spheres, box_min, box_max)
    distances = jnp.linalg.norm(collision_spheres - clamped, axis=-1) 
    
    H = jnp.maximum(0, collision_sphere_radii - distances) 
    intersection_volumes = (jnp.pi * (H**2) / 3) * (3 * collision_sphere_radii - H)
    
    # Handle static vs dynamic collision spheres gracefully
    if intersection_volumes.ndim == 1:
        total_vol = jnp.sum(intersection_volumes)
        return total_vol, total_vol
    else:
        total_volume_per_timestep = jnp.sum(intersection_volumes, axis=1)
        return jnp.sum(total_volume_per_timestep), jnp.max(total_volume_per_timestep)


@jax.jit
def compute_bar(box_min: jnp.ndarray, box_max: jnp.ndarray):
    """Barrier Volume (BAR): Computes the spatial volume of a safe set given its bounds."""
    return jnp.prod(box_max - box_min)


@jax.jit
def compute_bact(u_actual: jnp.ndarray, u_nominal: jnp.ndarray, dt: float, epsilon: float = 1e-4):
    """Barrier Activation Rate (BAct): Measures how often and for how long the CBF active filter overrides the nominal control input."""
    diff = jnp.linalg.norm(u_actual - u_nominal, axis=1)
    is_active = diff > epsilon
    activation_rate = (jnp.sum(is_active) / len(u_actual)) * 100.0
    return activation_rate, jnp.sum(is_active) * dt


@jax.jit
def compute_mean_abs_torque(tau: jnp.ndarray):
    """Per-joint Mean Absolute Torque: Calculates the average absolute torque applied per joint."""
    return jnp.mean(jnp.abs(tau), axis=0)


@jax.jit
def compute_tce(u: jnp.ndarray, dt: float):
    """Total Control Effort (TCE): Integral of control effort squared over the trajectory."""
    u_squared_norm = jnp.sum(u**2, axis=1)
    return jnp.sum(u_squared_norm) * dt


def generate_report(data: SimulationData, output_dir: str = "results"):
    """Generates the metrics report and saves it to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{data.date}_{data.experiment_title}_{data.prompt_version}_results.csv"
    filepath = os.path.join(output_dir, filename)
    
    # SVR for End-Effector Barrier evaluates if the commanded/unsafe trajectory (p_target) exits the safe barrier
    if data.p_target is not None and data.pos_min is not None and data.pos_max is not None:
        svr_ee = float(compute_svr_ee(jnp.array(data.p_target), jnp.array(data.pos_min), jnp.array(data.pos_max)))
    else:
        svr_ee = 0.0
        
    # CIA for End-Effector Barrier evaluates external collision objects against the EE barrier
    if data.collision_spheres is not None and data.collision_sphere_radii is not None and data.pos_min is not None and data.pos_max is not None:
        cia_ee_cumul, cia_ee_peak = compute_cia(jnp.array(data.collision_spheres), jnp.array(data.collision_sphere_radii), jnp.array(data.pos_min), jnp.array(data.pos_max))
        cia_ee_cumul, cia_ee_peak = float(cia_ee_cumul), float(cia_ee_peak)
    else:
        cia_ee_cumul, cia_ee_peak = 0.0, 0.0

    # SVR and CIA for Whole-Body Barrier
    has_wb = data.wb_min is not None and data.wb_max is not None
    if has_wb:
        if data.joint_spheres is not None and data.joint_sphere_radii is not None:
            svr_wb = float(compute_svr_wb(jnp.array(data.joint_spheres), jnp.array(data.joint_sphere_radii), jnp.array(data.wb_min), jnp.array(data.wb_max)))
        else:
            svr_wb = None
            
        if data.collision_spheres is not None and data.collision_sphere_radii is not None:
            cia_wb_cumul, cia_wb_peak = compute_cia(jnp.array(data.collision_spheres), jnp.array(data.collision_sphere_radii), jnp.array(data.wb_min), jnp.array(data.wb_max))
            cia_wb_cumul, cia_wb_peak = float(cia_wb_cumul), float(cia_wb_peak)
        else:
            cia_wb_cumul, cia_wb_peak = None, None
    else:
        svr_wb, cia_wb_cumul, cia_wb_peak = None, None, None

    # Volume (BAR)
    ee_vol = float(compute_bar(jnp.array(data.pos_min), jnp.array(data.pos_max))) if data.pos_min is not None and data.pos_max is not None else 0.0
    wb_vol = float(compute_bar(jnp.array(data.wb_min), jnp.array(data.wb_max))) if has_wb else None
    
    # Control Metrics
    bact_rate, bact_dur = compute_bact(jnp.array(data.u_actual), jnp.array(data.u_nominal), data.dt)
    bact_rate, bact_dur = float(bact_rate), float(bact_dur)
    tce_val = float(compute_tce(jnp.array(data.u_actual), data.dt))

    # Mean Tracking Error (MTE)
    if data.p_actual is not None and data.p_target is not None:
        steps_10s = int(10.0 / data.dt)
        steps_10s = min(steps_10s, len(data.p_actual))
        mte_mean_all, mte_std_all, mte_mean_last_10s, mte_std_last_10s = compute_mte(jnp.array(data.p_actual), jnp.array(data.p_target), steps_10s=steps_10s)
        mte_mean_all, mte_std_all = float(mte_mean_all), float(mte_std_all)
        mte_mean_last_10s, mte_std_last_10s = float(mte_mean_last_10s), float(mte_std_last_10s)
    else:
        mte_mean_all, mte_std_all, mte_mean_last_10s, mte_std_last_10s = 0.0, 0.0, 0.0, 0.0

    results_row = {
        "Experiment": data.experiment_title,
        "Prompt Version": data.prompt_version,
        "MTE_mean_all": mte_mean_all,
        "MTE_std_all": mte_std_all,
        "MTE_mean_last_10s": mte_mean_last_10s,
        "MTE_std_last_10s": mte_std_last_10s,
        "SVR_EE_%": svr_ee,
        "CIA_EE_cumul_vol": cia_ee_cumul,
        "CIA_EE_peak_vol": cia_ee_peak,
        "SVR_WB_%": svr_wb,
        "CIA_WB_cumul_vol": cia_wb_cumul,
        "CIA_WB_peak_vol": cia_wb_peak,
        "BAR_EE_vol": ee_vol,
        "BAR_WB_vol": wb_vol,
        "BAct_%": bact_rate,
        "BAct_duration_s": bact_dur,
        "TCE": tce_val
    }
    
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_row)
        
    print(f"Metrics saved to {filepath}")
    return results_row


def save_barriers_to_csv(data: SimulationData, output_dir: str = "results"):
    """Saves the generated barriers (EE and WB bounds) to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{data.date}_{data.experiment_title}_{data.prompt_version}_barriers.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Format the barrier bounds as lists for the CSV, or "None" if missing
    ee_min_str = str(data.pos_min.tolist()) if hasattr(data.pos_min, "tolist") else str(data.pos_min)
    ee_max_str = str(data.pos_max.tolist()) if hasattr(data.pos_max, "tolist") else str(data.pos_max)
    wb_min_str = str(data.wb_min.tolist()) if hasattr(data.wb_min, "tolist") else str(data.wb_min)
    wb_max_str = str(data.wb_max.tolist()) if hasattr(data.wb_max, "tolist") else str(data.wb_max)

    row = {
        "Experiment": data.experiment_title,
        "Prompt Version": data.prompt_version,
        "EE_Min": ee_min_str,
        "EE_Max": ee_max_str,
        "WB_Min": wb_min_str,
        "WB_Max": wb_max_str,
    }
    
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        
    print(f"Barriers saved to {filepath}")
    return row
