# scripts/generate_data.py

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from typing import Dict, Tuple

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from systems.lip import LIPSystem
from systems.five_link import FiveLinkSystem, FiveLinkParams
from systems.rollout import rollout_system, make_sine_input_function


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_box(low: torch.Tensor, high: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Uniformly sample from a box [low, high].

    Args:
        low:  (d,)
        high: (d,)
        num_samples: int

    Returns:
        samples: (num_samples, d)
    """
    if low.shape != high.shape:
        raise ValueError(f"low.shape {low.shape} must equal high.shape {high.shape}.")
    d = low.shape[0]
    rand = torch.rand(num_samples, d, dtype=low.dtype, device=low.device)
    return low.unsqueeze(0) + (high - low).unsqueeze(0) * rand


def sample_sine_amplitudes(
    num_trajectories: int,
    input_dim: int,
    num_frequencies: int,
    amp_low: float,
    amp_high: float,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Sample amplitudes for sinusoidal inputs.

    Returns:
        amps: (num_trajectories, input_dim, num_frequencies)
    """
    amps = amp_low + (amp_high - amp_low) * torch.rand(
        num_trajectories, input_dim, num_frequencies, device=device, dtype=dtype
    )
    return amps


def generate_lip_library(
    system: LIPSystem,
    num_trajectories: int,
    dt: float,
    num_steps: int,
    z0_low: torch.Tensor,
    z0_high: torch.Tensor,
    frequencies: torch.Tensor,
    amp_low: float,
    amp_high: float,
    integrator: str = "rk4",
) -> Dict[str, torch.Tensor]:
    """
    Generate a library of LIP trajectories.

    Returns dict with:
        z0:   (Nz, 2)
        amps: (Nz, 1, K)
        t:    (T+1,)
        x:    (Nz, T+1, 2)   # here x means z-state trajectory
        y:    (Nz, T+1, 2)
        u:    (Nz, T, 1)     # here u means v-input trajectory
    """
    dtype = z0_low.dtype
    device = z0_low.device

    z0_all = sample_box(z0_low, z0_high, num_trajectories)  # (Nz, 2)
    amps_all = sample_sine_amplitudes(
        num_trajectories=num_trajectories,
        input_dim=1,
        num_frequencies=frequencies.shape[0],
        amp_low=amp_low,
        amp_high=amp_high,
        device=device,
        dtype=dtype,
    )

    x_trajs = []
    y_trajs = []
    u_trajs = []
    t_ref = None

    for k in range(num_trajectories):
        z0 = z0_all[k]                  # (2,)
        amps = amps_all[k, 0, :]        # (K,)
        input_fn = make_sine_input_function(amps, frequencies)

        out = rollout_system(
            system=system,
            x0=z0,
            input_fn=input_fn,
            dt=dt,
            num_steps=num_steps,
            integrator=integrator,
        )

        if t_ref is None:
            t_ref = out["t"]

        x_trajs.append(out["x"])        # (T+1, 2)
        y_trajs.append(out["y"])        # (T+1, 2)
        u_trajs.append(out["u"])        # (T, 1)

    return {
        "z0": z0_all,                   # (Nz, 2)
        "amps": amps_all,               # (Nz, 1, K)
        "t": t_ref,                     # (T+1,)
        "x": torch.stack(x_trajs, dim=0),   # (Nz, T+1, 2)
        "y": torch.stack(y_trajs, dim=0),   # (Nz, T+1, 2)
        "u": torch.stack(u_trajs, dim=0),   # (Nz, T, 1)
    }


def generate_five_link_library(
    system: FiveLinkSystem,
    num_trajectories: int,
    dt: float,
    num_steps: int,
    x0_low: torch.Tensor,
    x0_high: torch.Tensor,
    frequencies: torch.Tensor,
    amp_low: float,
    amp_high: float,
    integrator: str = "rk4",
) -> Dict[str, torch.Tensor]:
    """
    Generate a library of five-link trajectories.

    Returns dict with:
        x0:   (Nx, 10)
        amps: (Nx, 5, K)
        t:    (T+1,)
        x:    (Nx, T+1, 10)
        y:    (Nx, T+1, 2)
        u:    (Nx, T, 5)
    """
    dtype = x0_low.dtype
    device = x0_low.device

    x0_all = sample_box(x0_low, x0_high, num_trajectories)  # (Nx, 10)
    amps_all = sample_sine_amplitudes(
        num_trajectories=num_trajectories,
        input_dim=5,
        num_frequencies=frequencies.shape[0],
        amp_low=amp_low,
        amp_high=amp_high,
        device=device,
        dtype=dtype,
    )

    x_trajs = []
    y_trajs = []
    u_trajs = []
    t_ref = None

    for k in range(num_trajectories):
        x0 = x0_all[k]                # (10,)
        amps = amps_all[k]            # (5, K)
        input_fn = make_sine_input_function(amps, frequencies)

        out = rollout_system(
            system=system,
            x0=x0,
            input_fn=input_fn,
            dt=dt,
            num_steps=num_steps,
            integrator=integrator,
        )

        if t_ref is None:
            t_ref = out["t"]

        x_trajs.append(out["x"])      # (T+1, 10)
        y_trajs.append(out["y"])      # (T+1, 2)
        u_trajs.append(out["u"])      # (T, 5)

    return {
        "x0": x0_all,                    # (Nx, 10)
        "amps": amps_all,                # (Nx, 5, K)
        "t": t_ref,                      # (T+1,)
        "x": torch.stack(x_trajs, dim=0),    # (Nx, T+1, 10)
        "y": torch.stack(y_trajs, dim=0),    # (Nx, T+1, 2)
        "u": torch.stack(u_trajs, dim=0),    # (Nx, T, 5)
    }


def flatten_lip_library(lib: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Flatten LIP trajectory library into point samples.

    Input:
        x: (Nz, T+1, 2)
        y: (Nz, T+1, 2)
        u: (Nz, T, 1)

    Since decay loss uses v_i at each state point and the input exists only on T steps,
    we align samples with the first T states:
        z_i = z(t_k), v_i = v(t_k), y2_i = y2(t_k), k=0,...,T-1

    Returns:
        z:  (Nz*T, 2)
        v:  (Nz*T, 1)
        y2: (Nz*T, 2)
    """
    z = lib["x"][:, :-1, :].reshape(-1, 2)
    v = lib["u"].reshape(-1, 1)
    y2 = lib["y"][:, :-1, :].reshape(-1, 2)

    return {
        "z": z,
        "v": v,
        "y2": y2,
    }


def flatten_five_link_library(lib: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Flatten five-link trajectory library into point samples aligned with first T states.

    Returns:
        x:  (Nx*T, 10)
        y1: (Nx*T, 2)
        u:  (Nx*T, 5)
    """
    x = lib["x"][:, :-1, :].reshape(-1, 10)
    y1 = lib["y"][:, :-1, :].reshape(-1, 2)
    u = lib["u"].reshape(-1, 5)

    return {
        "x": x,
        "y1": y1,
        "u": u,
    }


def pair_samples_randomly(
    lip_points: Dict[str, torch.Tensor],
    five_points: Dict[str, torch.Tensor],
    num_pairs: int,
) -> Dict[str, torch.Tensor]:
    """
    Randomly pair LIP points with five-link points.

    Returns:
        x:  (N, 10)
        z:  (N, 2)
        u:  (N, 5)
        v:  (N, 1)
        y1: (N, 2)
        y2: (N, 2)
    """
    num_lip = lip_points["z"].shape[0]
    num_five = five_points["x"].shape[0]

    idx_lip = torch.randint(0, num_lip, (num_pairs,))
    idx_five = torch.randint(0, num_five, (num_pairs,))

    z = lip_points["z"][idx_lip]
    v = lip_points["v"][idx_lip]
    y2 = lip_points["y2"][idx_lip]

    x = five_points["x"][idx_five]
    u = five_points["u"][idx_five]
    y1 = five_points["y1"][idx_five]

    return {
        "x": x,
        "z": z,
        "u": u,
        "v": v,
        "y1": y1,
        "y2": y2,
    }


def pair_samples_by_output_proximity(
    lip_points: Dict[str, torch.Tensor],
    five_points: Dict[str, torch.Tensor],
    num_pairs: int,
    eps_pair: float,
) -> Dict[str, torch.Tensor]:
    """
    Pair samples only when ||y1 - y2|| <= eps_pair.

    This is a simple rejection-sampling version.
    If not enough close pairs are found, it returns however many were found.
    """
    z_all = lip_points["z"]
    v_all = lip_points["v"]
    y2_all = lip_points["y2"]

    x_all = five_points["x"]
    u_all = five_points["u"]
    y1_all = five_points["y1"]

    num_lip = z_all.shape[0]
    num_five = x_all.shape[0]

    xs, zs, us, vs, y1s, y2s = [], [], [], [], [], []

    max_trials = 20 * num_pairs
    trials = 0

    while len(xs) < num_pairs and trials < max_trials:
        i = torch.randint(0, num_lip, (1,)).item()
        j = torch.randint(0, num_five, (1,)).item()

        if torch.norm(y1_all[j] - y2_all[i], p=2).item() <= eps_pair:
            xs.append(x_all[j])
            zs.append(z_all[i])
            us.append(u_all[j])
            vs.append(v_all[i])
            y1s.append(y1_all[j])
            y2s.append(y2_all[i])

        trials += 1

    if len(xs) == 0:
        raise RuntimeError(
            "No paired samples found under the current eps_pair. "
            "Try a larger threshold or use random pairing."
        )

    return {
        "x": torch.stack(xs, dim=0),
        "z": torch.stack(zs, dim=0),
        "u": torch.stack(us, dim=0),
        "v": torch.stack(vs, dim=0),
        "y1": torch.stack(y1s, dim=0),
        "y2": torch.stack(y2s, dim=0),
    }


def default_config() -> Dict:

    return{
    "seed": 42,
    "device": "cpu",
    "dtype": "float32",
    "save_dir": os.path.join("data", "processed"),
    "save_name": "train_dataset.pt",

    "integrator": "euler",   # 改这里
    "dt": 0.02,              # 稍微大一点
    "num_steps": 50,          # 先极小
    "num_lip_trajectories": 10,
    "num_five_link_trajectories": 10,
    "num_pairs": 1000,

    "pair_mode": "close",
    "eps_pair": 0.5,

    "z0_low": [-0.3, -0.5],
    "z0_high": [0.3, 0.5],

    "x0_low": [-0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2],
    "x0_high": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],

    "lip_frequencies": [0.5, 1.0],
    "five_link_frequencies": [0.5, 1.0],

    "lip_amp_low": -0.1,
    "lip_amp_high": 0.1,
    "five_amp_low": -0.1,
    "five_amp_high": 0.1,

    "five_link_params": {
        "lengths": (0.5, 0.5, 0.4, 0.4, 0.3),
        "com_lengths": (0.25, 0.25, 0.2, 0.2, 0.15),
        "masses": (2.0, 2.0, 1.5, 1.5, 1.0),
        "inertias": (0.03, 0.03, 0.02, 0.02, 0.01),
        "damping": (0.05, 0.05, 0.03, 0.03, 0.02),
        "gravity": 9.81,
        "B": None,
    },
}


def build_systems(cfg: Dict, device: str, dtype: torch.dtype):
    lip = LIPSystem(g=9.81, h=1.0, device=device, dtype=dtype)

    fp = cfg["five_link_params"]
    five_params = FiveLinkParams(
        lengths=tuple(fp["lengths"]),
        com_lengths=tuple(fp["com_lengths"]),
        masses=tuple(fp["masses"]),
        inertias=tuple(fp["inertias"]),
        damping=tuple(fp["damping"]),
        gravity=fp["gravity"],
        B=fp["B"],
    )
    five_link = FiveLinkSystem(params=five_params, device=device, dtype=dtype)

    return lip, five_link


def main():
    cfg = default_config()
    set_seed(cfg["seed"])

    device = cfg["device"]
    dtype = torch.float32 if cfg["dtype"] == "float32" else torch.float64

    os.makedirs(cfg["save_dir"], exist_ok=True)
    save_path = os.path.join(cfg["save_dir"], cfg["save_name"])

    lip, five_link = build_systems(cfg, device=device, dtype=dtype)

    z0_low = torch.tensor(cfg["z0_low"], dtype=dtype, device=device)
    z0_high = torch.tensor(cfg["z0_high"], dtype=dtype, device=device)

    x0_low = torch.tensor(cfg["x0_low"], dtype=dtype, device=device)
    x0_high = torch.tensor(cfg["x0_high"], dtype=dtype, device=device)

    lip_freqs = torch.tensor(cfg["lip_frequencies"], dtype=dtype, device=device)
    five_freqs = torch.tensor(cfg["five_link_frequencies"], dtype=dtype, device=device)

    print("Generating LIP trajectory library...")
    lip_lib = generate_lip_library(
        system=lip,
        num_trajectories=cfg["num_lip_trajectories"],
        dt=cfg["dt"],
        num_steps=cfg["num_steps"],
        z0_low=z0_low,
        z0_high=z0_high,
        frequencies=lip_freqs,
        amp_low=cfg["lip_amp_low"],
        amp_high=cfg["lip_amp_high"],
        integrator=cfg["integrator"],
    )

    print("Generating five-link trajectory library...")
    five_lib = generate_five_link_library(
        system=five_link,
        num_trajectories=cfg["num_five_link_trajectories"],
        dt=cfg["dt"],
        num_steps=cfg["num_steps"],
        x0_low=x0_low,
        x0_high=x0_high,
        frequencies=five_freqs,
        amp_low=cfg["five_amp_low"],
        amp_high=cfg["five_amp_high"],
        integrator=cfg["integrator"],
    )

    print("Flattening trajectory libraries...")
    lip_points = flatten_lip_library(lip_lib)
    five_points = flatten_five_link_library(five_lib)

    print("Pairing samples...")
    if cfg["pair_mode"] == "random":
        dataset = pair_samples_randomly(
            lip_points=lip_points,
            five_points=five_points,
            num_pairs=cfg["num_pairs"],
        )
    elif cfg["pair_mode"] == "close":
        dataset = pair_samples_by_output_proximity(
            lip_points=lip_points,
            five_points=five_points,
            num_pairs=cfg["num_pairs"],
            eps_pair=cfg["eps_pair"],
        )
    else:
        raise ValueError(f"Unknown pair_mode: {cfg['pair_mode']}")

    payload = {
        "config": cfg,
        "lip_library": lip_lib,
        "five_link_library": five_lib,
        "lip_points": lip_points,
        "five_link_points": five_points,
        "dataset": dataset,
    }

    torch.save(payload, save_path)

    print("\nSaved dataset to:", save_path)
    print("Dataset summary:")
    print("  x shape :", dataset["x"].shape)
    print("  z shape :", dataset["z"].shape)
    print("  u shape :", dataset["u"].shape)
    print("  v shape :", dataset["v"].shape)
    print("  y1 shape:", dataset["y1"].shape)
    print("  y2 shape:", dataset["y2"].shape)


if __name__ == "__main__":
    main()
