from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.interface import InterfaceNet
from models.simulation_function import CNetwork, SimulationFunction
from scripts.train import build_systems
from systems.rollout import euler_step, rk4_step


def sine_v(t: torch.Tensor, amplitude: float, frequency: float) -> torch.Tensor:
    """Scalar LIP input v(t) = amplitude * sin(frequency * t)."""
    return amplitude * torch.sin(frequency * t).view(1)


def sine_steady_state_z0(lip, amplitude: float, frequency: float, dtype: torch.dtype, device: str) -> torch.Tensor:
    """
    Initial condition for the bounded particular solution of
        p_ddot = omega^2 * (p - A sin(alpha t)).

    For p(t) = B sin(alpha t), B = omega^2 A / (omega^2 + alpha^2).
    At t=0, z0 = [p(0), p_dot(0)] = [0, alpha * B].
    """
    omega_sq = lip.omega ** 2
    gain = omega_sq * amplitude / (omega_sq + frequency ** 2)
    return torch.tensor([0.0, frequency * gain], dtype=dtype, device=device)


def load_checkpoint(path: str, device: str) -> Dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def make_models(checkpoint: Dict, device: str, dtype: torch.dtype):
    config = checkpoint.get("config", {})
    lip, five_link = build_systems(device=device, dtype=dtype)

    interface_net = InterfaceNet().to(device)
    c_net = CNetwork().to(device)
    sim_fn = SimulationFunction(
        c_net,
        five_link,
        lip,
        m=config.get("m", 0.01),
    ).to(device)

    interface_net.load_state_dict(checkpoint["interface_state_dict"])
    sim_fn.load_state_dict(checkpoint["sim_fn_state_dict"])

    interface_net.eval()
    sim_fn.eval()

    return lip, five_link, interface_net, sim_fn


def closed_loop_rollout(
    lip,
    five_link,
    interface_net: InterfaceNet,
    sim_fn: SimulationFunction,
    x0: torch.Tensor,
    z0: torch.Tensor,
    dt: float,
    num_steps: int,
    v_amplitude: float,
    v_frequency: float,
    integrator: str,
    r: float,
    max_abs_z: float | None,
) -> Dict[str, torch.Tensor]:
    if integrator not in ("euler", "rk4"):
        raise ValueError(f"Unsupported integrator: {integrator}")

    step_fn = euler_step if integrator == "euler" else rk4_step
    device = x0.device
    dtype = x0.dtype

    x = x0.view(1, -1)
    z = z0.view(1, -1)

    t_hist = []
    x_hist = []
    z_hist = []
    y1_hist = []
    y2_hist = []
    V_hist = []
    Vdot_hist = []
    violation_hist = []
    err_hist = []
    v_hist = []
    u_hist = []

    stopped_early = False
    stop_reason = ""

    for k in range(num_steps + 1):
        t = torch.tensor(k * dt, dtype=dtype, device=device)
        v = sine_v(t, v_amplitude, v_frequency).to(device=device, dtype=dtype).view(1, 1)

        y1 = five_link.output(x)
        y2 = lip.output(z)
        V = sim_fn(x, z)
        err = torch.linalg.norm(y1 - y2, dim=-1)

        x_req = x.clone().detach().requires_grad_(True)
        z_req = z.clone().detach().requires_grad_(True)
        V_req = sim_fn(x_req, z_req)
        u_req = interface_net(x_req, z_req, v)
        fx_req = five_link.dynamics(x_req, u_req)
        fz_req = lip.dynamics(z_req, v)
        dV_dx = torch.autograd.grad(V_req.sum(), x_req, create_graph=False, retain_graph=True)[0]
        dV_dz = torch.autograd.grad(V_req.sum(), z_req, create_graph=False, retain_graph=True)[0]
        Vdot = (dV_dx * fx_req).sum(dim=-1) + (dV_dz * fz_req).sum(dim=-1)
        violation = Vdot + r * V_req.detach()

        t_hist.append(t.detach())
        x_hist.append(x.squeeze(0).detach())
        z_hist.append(z.squeeze(0).detach())
        y1_hist.append(y1.squeeze(0).detach())
        y2_hist.append(y2.squeeze(0).detach())
        V_hist.append(V.squeeze(0).detach())
        Vdot_hist.append(Vdot.squeeze(0).detach())
        violation_hist.append(violation.squeeze(0).detach())
        err_hist.append(err.squeeze(0).detach())

        if k == num_steps:
            break
        if max_abs_z is not None and z.abs().max().item() > max_abs_z:
            stopped_early = True
            stop_reason = f"|z| exceeded {max_abs_z}"
            break

        u = interface_net(x, z, v)

        v_hist.append(v.squeeze(0).detach())
        u_hist.append(u.squeeze(0).detach())

        z = step_fn(lip, z, v, dt).detach()
        x = step_fn(five_link, x, u, dt).detach()

    return {
        "t": torch.stack(t_hist, dim=0),
        "x": torch.stack(x_hist, dim=0),
        "z": torch.stack(z_hist, dim=0),
        "y1": torch.stack(y1_hist, dim=0),
        "y2": torch.stack(y2_hist, dim=0),
        "V": torch.stack(V_hist, dim=0),
        "Vdot": torch.stack(Vdot_hist, dim=0),
        "violation": torch.stack(violation_hist, dim=0),
        "output_error": torch.stack(err_hist, dim=0),
        "v": torch.stack(v_hist, dim=0),
        "u": torch.stack(u_hist, dim=0),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }


def save_csv(path: str, rollout: Dict[str, torch.Tensor]) -> None:
    t = rollout["t"].cpu()
    z = rollout["z"].cpu()
    x = rollout["x"].cpu()
    y1 = rollout["y1"].cpu()
    y2 = rollout["y2"].cpu()
    V = rollout["V"].cpu()
    Vdot = rollout["Vdot"].cpu()
    violation = rollout["violation"].cpu()
    err = rollout["output_error"].cpu()
    v = rollout["v"].cpu()
    u = rollout["u"].cpu()

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "z0",
                "z1",
                "y1_0",
                "y1_1",
                "y2_0",
                "y2_1",
                "V",
                "Vdot",
                "violation",
                "output_error",
                "v",
                *[f"x{i}" for i in range(x.shape[-1])],
                *[f"u{i}" for i in range(u.shape[-1])],
            ]
        )

        for k in range(t.shape[0]):
            v_k = v[k, 0].item() if k < v.shape[0] else ""
            u_k = u[k].tolist() if k < u.shape[0] else [""] * u.shape[-1]
            writer.writerow(
                [
                    t[k].item(),
                    z[k, 0].item(),
                    z[k, 1].item(),
                    y1[k, 0].item(),
                    y1[k, 1].item(),
                    y2[k, 0].item(),
                    y2[k, 1].item(),
                    V[k].item(),
                    Vdot[k].item(),
                    violation[k].item(),
                    err[k].item(),
                    v_k,
                    *x[k].tolist(),
                    *u_k,
                ]
            )


def maybe_save_plots(output_dir: str, rollout: Dict[str, torch.Tensor], show_plots: bool = False) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        return []

    t = rollout["t"].cpu()
    y1 = rollout["y1"].cpu()
    y2 = rollout["y2"].cpu()
    Vdot = rollout["Vdot"].cpu()
    position_error = torch.abs(y1[:, 0] - y2[:, 0])
    velocity_error = torch.abs(y1[:, 1] - y2[:, 1])
    saved_paths = []

    def save_current_figure(filename: str) -> None:
        path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(path, dpi=200)
        saved_paths.append(path)
        if not show_plots:
            plt.close()

    plt.figure()
    plt.plot(t, y1[:, 0], label="y1 position")
    plt.plot(t, y2[:, 0], label="y2 position")
    plt.plot(t, position_error, label="|y1 position - y2 position|")
    plt.xlabel("t")
    plt.legend()
    save_current_figure("closed_loop_outputs_position.png")

    plt.figure()
    plt.plot(t, y1[:, 1], label="y1 velocity")
    plt.plot(t, y2[:, 1], label="y2 velocity")
    plt.plot(t, velocity_error, label="|y1 velocity - y2 velocity|")
    plt.xlabel("t")
    plt.legend()
    save_current_figure("closed_loop_outputs_velocity.png")

    plt.figure()
    plt.plot(t, Vdot, label="Vdot")
    plt.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    plt.xlabel("t")
    plt.legend()
    save_current_figure("closed_loop_Vdot.png")

    if show_plots:
        plt.show()

    return saved_paths


def parse_args() -> argparse.Namespace:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser(
        description="Evaluate the trained simulation function and interface on a closed-loop rollout."
    )
    parser.add_argument("--checkpoint", default=os.path.join(project_root, "checkpoints", "latest.pt"))
    parser.add_argument("--output-dir", default=os.path.join(project_root, "results", "closed_loop"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--integrator", choices=["euler", "rk4"], default="euler")
    parser.add_argument("--v-amplitude", type=float, default=0.1)
    parser.add_argument("--v-frequency", type=float, default=0.2)
    parser.add_argument("--r", type=float, default=None, help="Decay rate. Defaults to checkpoint config['r'].")
    parser.add_argument(
        "--max-abs-z",
        type=float,
        default=5.0,
        help="Stop the rollout once any absolute z component exceeds this value. Use a negative value to disable.",
    )
    parser.add_argument("--z0", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument(
        "--steady-state-z0",
        action="store_true",
        help="Use the bounded sinusoidal LIP reference initial condition instead of --z0.",
    )
    parser.add_argument("--x0", type=float, nargs=10, default=[0.0] * 10)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-plots", action="store_true", help="Open matplotlib windows after saving plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, device=args.device)
    lip, five_link, interface_net, sim_fn = make_models(checkpoint, device=args.device, dtype=dtype)
    checkpoint_config = checkpoint.get("config", {})
    r = checkpoint_config.get("r", 0.001) if args.r is None else args.r
    max_abs_z = None if args.max_abs_z < 0.0 else args.max_abs_z
    horizon = args.dt * args.num_steps
    if horizon > 1.0:
        print(
            "Warning: this rollout horizon is longer than the current data-generation default "
            f"of about 1.0 s. LIP dynamics are open-loop unstable, so z(t) may leave the "
            f"training region quickly. Current horizon: {horizon:.3f} s."
        )

    x0 = torch.tensor(args.x0, dtype=dtype, device=args.device)
    if args.steady_state_z0:
        z0 = sine_steady_state_z0(
            lip=lip,
            amplitude=args.v_amplitude,
            frequency=args.v_frequency,
            dtype=dtype,
            device=args.device,
        )
        print(f"Using bounded sinusoidal LIP z0: {z0.detach().cpu().tolist()}")
    else:
        z0 = torch.tensor(args.z0, dtype=dtype, device=args.device)

    rollout = closed_loop_rollout(
        lip=lip,
        five_link=five_link,
        interface_net=interface_net,
        sim_fn=sim_fn,
        x0=x0,
        z0=z0,
        dt=args.dt,
        num_steps=args.num_steps,
        v_amplitude=args.v_amplitude,
        v_frequency=args.v_frequency,
        integrator=args.integrator,
        r=r,
        max_abs_z=max_abs_z,
    )

    pt_path = os.path.join(args.output_dir, "closed_loop_rollout.pt")
    csv_path = os.path.join(args.output_dir, "closed_loop_rollout.csv")

    torch.save(
        {
            "args": vars(args),
            "r": r,
            "max_abs_z": max_abs_z,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "rollout": rollout,
        },
        pt_path,
    )
    save_csv(csv_path, rollout)

    saved_plot_paths = []
    if not args.no_plots:
        saved_plot_paths = maybe_save_plots(args.output_dir, rollout, show_plots=args.show_plots)

    err = rollout["output_error"]
    V = rollout["V"]
    violation = rollout["violation"]
    z = rollout["z"]
    v = rollout["v"]
    print("Closed-loop evaluation finished.")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  epoch: {checkpoint.get('epoch')}")
    print(f"  saved pt: {pt_path}")
    print(f"  saved csv: {csv_path}")
    if saved_plot_paths:
        print("  saved plots:")
        for path in saved_plot_paths:
            print(f"    {path}")
    print(f"  mean ||y1-y2||: {err.mean().item():.6e}")
    print(f"  max  ||y1-y2||: {err.max().item():.6e}")
    print(f"  V initial: {V[0].item():.6e}")
    print(f"  V final:   {V[-1].item():.6e}")
    print(f"  mean Vdot+rV: {violation.mean().item():.6e}")
    print(f"  max  Vdot+rV: {violation.max().item():.6e}")
    print(f"  z0 range: [{z[:, 0].min().item():.6e}, {z[:, 0].max().item():.6e}]")
    print(f"  z1 range: [{z[:, 1].min().item():.6e}, {z[:, 1].max().item():.6e}]")
    if v.numel() > 0:
        print(f"  v range:  [{v[:, 0].min().item():.6e}, {v[:, 0].max().item():.6e}]")
    if rollout["stopped_early"]:
        print(f"  stopped early: {rollout['stop_reason']}")


if __name__ == "__main__":
    main()
