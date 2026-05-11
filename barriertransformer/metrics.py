from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np
from datetime import datetime
import os
import csv

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
    
    # Robot Collision Spheres
    robot_spheres: np.ndarray = None 
    sphere_radii: np.ndarray = None  
    
    # Metadata for CSV
    experiment_title: str = "default_experiment"
    prompt_version: str = "v1"
    date: str = datetime.now().strftime("%Y-%m-%d")


@jax.jit
def compute_mte(p_actual: jnp.ndarray, p_target: jnp.ndarray):
    """Mean Tracking Error (MTE): Calculates the average and std dev L2 norm between actual and target end-effector positions."""
    errors = jnp.linalg.norm(p_actual - p_target, axis=1)
    return jnp.mean(errors), jnp.std(errors)


@jax.jit
def compute_svr(robot_spheres: jnp.ndarray, radii: jnp.ndarray, box_min: jnp.ndarray, box_max: jnp.ndarray):
    """Safety Violation Rate (SVR): Percentage of time steps where any robot sphere exits the safe barrier."""
    outside_min = (robot_spheres - radii[:, None]) < box_min
    outside_max = (robot_spheres + radii[:, None]) > box_max
    
    violation_per_timestep = jnp.any(outside_min | outside_max, axis=(1, 2))
    svr_percentage = (jnp.sum(violation_per_timestep) / len(violation_per_timestep)) * 100.0
    return svr_percentage


@jax.jit
def compute_cia(robot_spheres: jnp.ndarray, radii: jnp.ndarray, box_min: jnp.ndarray, box_max: jnp.ndarray):
    """Collision Intersection Area (CIA): Cumulative and peak intersection volume between robot spheres and safe set bounds."""
    clamped = jnp.clip(robot_spheres, box_min, box_max)
    distances = jnp.linalg.norm(robot_spheres - clamped, axis=-1) 
    
    H = jnp.maximum(0, radii - distances) 
    intersection_volumes = (jnp.pi * (H**2) / 3) * (3 * radii - H)
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
    
    # SVR and CIA for End-Effector Barrier
    if data.robot_spheres is not None and data.sphere_radii is not None and data.pos_min is not None and data.pos_max is not None:
        svr_ee = float(compute_svr(jnp.array(data.robot_spheres), jnp.array(data.sphere_radii), jnp.array(data.pos_min), jnp.array(data.pos_max)))
        cia_ee_cumul, cia_ee_peak = compute_cia(jnp.array(data.robot_spheres), jnp.array(data.sphere_radii), jnp.array(data.pos_min), jnp.array(data.pos_max))
        cia_ee_cumul, cia_ee_peak = float(cia_ee_cumul), float(cia_ee_peak)
    else:
        svr_ee, cia_ee_cumul, cia_ee_peak = 0.0, 0.0, 0.0

    # SVR and CIA for Whole-Body Barrier
    if data.robot_spheres is not None and data.sphere_radii is not None and data.wb_min is not None and data.wb_max is not None:
        svr_wb = float(compute_svr(jnp.array(data.robot_spheres), jnp.array(data.sphere_radii), jnp.array(data.wb_min), jnp.array(data.wb_max)))
        cia_wb_cumul, cia_wb_peak = compute_cia(jnp.array(data.robot_spheres), jnp.array(data.sphere_radii), jnp.array(data.wb_min), jnp.array(data.wb_max))
        cia_wb_cumul, cia_wb_peak = float(cia_wb_cumul), float(cia_wb_peak)
    else:
        svr_wb, cia_wb_cumul, cia_wb_peak = 0.0, 0.0, 0.0

    # Volume (BAR)
    ee_vol = float(compute_bar(jnp.array(data.pos_min), jnp.array(data.pos_max))) if data.pos_min is not None and data.pos_max is not None else 0.0
    wb_vol = float(compute_bar(jnp.array(data.wb_min), jnp.array(data.wb_max))) if data.wb_min is not None and data.wb_max is not None else 0.0
    
    # Control Metrics
    bact_rate, bact_dur = compute_bact(jnp.array(data.u_actual), jnp.array(data.u_nominal), data.dt)
    bact_rate, bact_dur = float(bact_rate), float(bact_dur)
    tce_val = float(compute_tce(jnp.array(data.u_actual), data.dt))

    # Mean Tracking Error (MTE)
    if data.p_actual is not None and data.p_target is not None:
        mte_mean, mte_std = compute_mte(jnp.array(data.p_actual), jnp.array(data.p_target))
        mte_mean, mte_std = float(mte_mean), float(mte_std)
    else:
        mte_mean, mte_std = 0.0, 0.0

    results_row = {
        "Experiment": data.experiment_title,
        "Prompt Version": data.prompt_version,
        "MTE_mean": mte_mean,
        "MTE_std": mte_std,
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
