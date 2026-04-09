# systems/rollout.py

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch


Tensor = torch.Tensor


def _ensure_2d(x: Tensor, last_dim: int) -> tuple[Tensor, bool]:
    """
    Ensure x has shape (B, last_dim). If input is 1D, add batch dimension.
    Returns:
        x_2d, squeezed
    """
    if x.shape[-1] != last_dim:
        raise ValueError(f"Expected last dimension {last_dim}, got shape {x.shape}.")
    squeezed = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        squeezed = True
    return x, squeezed


def euler_step(system, x: Tensor, u: Tensor, dt: float) -> Tensor:
    """
    One Euler integration step:
        x_{k+1} = x_k + dt * f(x_k, u_k)
    """
    dx = system.dynamics(x, u)
    return x + dt * dx


def rk4_step(system, x: Tensor, u: Tensor, dt: float) -> Tensor:
    """
    One RK4 integration step with zero-order-hold input u over the step.
    """
    k1 = system.dynamics(x, u)
    k2 = system.dynamics(x + 0.5 * dt * k1, u)
    k3 = system.dynamics(x + 0.5 * dt * k2, u)
    k4 = system.dynamics(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def rollout_system(
    system,
    x0: Tensor,
    input_fn: Callable[[Tensor], Tensor],
    dt: float,
    num_steps: int,
    integrator: str = "rk4",
    clamp_state_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    Roll out a single trajectory or a batch of trajectories.

    Args:
        system:
            Must provide:
                dynamics(x, u)
                output(x)
            with batch-compatible PyTorch tensors.
        x0:
            Initial state, shape (state_dim,) or (B, state_dim).
        input_fn:
            Function of time t returning input u(t).
            Should return shape (input_dim,) or (B, input_dim).
        dt:
            Time step.
        num_steps:
            Number of integration steps.
        integrator:
            "euler" or "rk4".
        clamp_state_fn:
            Optional function applied after every step:
                x_next = clamp_state_fn(x_next)

    Returns:
        dict with keys:
            "t":      (T,)
            "x":      (B, T, state_dim) or (T, state_dim) if single
            "y":      (B, T, output_dim) or (T, output_dim) if single
            "u":      (B, T, input_dim) or (T, input_dim) if single
        where T = num_steps + 1 for x,y and T = num_steps for u.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")

    state_dim = x0.shape[-1]
    x, squeezed = _ensure_2d(x0, state_dim)
    device = x.device
    dtype = x.dtype
    batch_size = x.shape[0]

    if integrator not in ("euler", "rk4"):
        raise ValueError(f"Unsupported integrator '{integrator}'. Use 'euler' or 'rk4'.")

    step_fn = euler_step if integrator == "euler" else rk4_step

    x_hist = [x]
    y_hist = [system.output(x)]
    u_hist = []
    t_hist = [torch.tensor(0.0, dtype=dtype, device=device)]

    for k in range(num_steps):
        t_k = torch.tensor(k * dt, dtype=dtype, device=device)

        u_k = input_fn(t_k)
        if not isinstance(u_k, torch.Tensor):
            u_k = torch.tensor(u_k, dtype=dtype, device=device)
        else:
            u_k = u_k.to(device=device, dtype=dtype)

        if u_k.dim() == 1:
            u_k = u_k.unsqueeze(0)

        if u_k.shape[0] == 1 and batch_size > 1:
            u_k = u_k.expand(batch_size, -1)

        if u_k.shape[0] != batch_size:
            raise ValueError(
                f"Input batch size mismatch: got u_k.shape[0]={u_k.shape[0]}, expected {batch_size}."
            )

        x_next = step_fn(system, x, u_k, dt)

        if clamp_state_fn is not None:
            x_next = clamp_state_fn(x_next)

        y_next = system.output(x_next)

        u_hist.append(u_k)
        x_hist.append(x_next)
        y_hist.append(y_next)
        t_hist.append(torch.tensor((k + 1) * dt, dtype=dtype, device=device))

        x = x_next

    x_traj = torch.stack(x_hist, dim=1)  # (B, T+1, state_dim)
    y_traj = torch.stack(y_hist, dim=1)  # (B, T+1, output_dim)
    u_traj = torch.stack(u_hist, dim=1)  # (B, T, input_dim)
    t_traj = torch.stack(t_hist, dim=0)  # (T+1,)

    if squeezed:
        x_traj = x_traj.squeeze(0)
        y_traj = y_traj.squeeze(0)
        u_traj = u_traj.squeeze(0)

    return {
        "t": t_traj,
        "x": x_traj,
        "y": y_traj,
        "u": u_traj,
    }


def rollout_from_input_trajectory(
    system,
    x0: Tensor,
    u_traj: Tensor,
    dt: float,
    integrator: str = "rk4",
    clamp_state_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> Dict[str, Tensor]:
    """
    Roll out using a precomputed discrete-time input trajectory.

    Args:
        system:
            Must provide dynamics(x, u), output(x).
        x0:
            Initial state, shape (state_dim,) or (B, state_dim).
        u_traj:
            Input trajectory, shape (T, input_dim) or (B, T, input_dim).
        dt:
            Time step.
        integrator:
            "euler" or "rk4".
        clamp_state_fn:
            Optional projection/clamp on state.

    Returns:
        Same format as rollout_system.
    """
    if u_traj.dim() not in (2, 3):
        raise ValueError(
            f"u_traj must have shape (T, input_dim) or (B, T, input_dim), got {u_traj.shape}."
        )

    state_dim = x0.shape[-1]
    x, squeezed_x = _ensure_2d(x0, state_dim)
    device = x.device
    dtype = x.dtype
    batch_size = x.shape[0]

    if u_traj.dim() == 2:
        u_traj = u_traj.unsqueeze(0)  # (1, T, input_dim)

    u_traj = u_traj.to(device=device, dtype=dtype)

    if u_traj.shape[0] == 1 and batch_size > 1:
        u_traj = u_traj.expand(batch_size, -1, -1)

    if u_traj.shape[0] != batch_size:
        raise ValueError(
            f"Input trajectory batch size mismatch: got {u_traj.shape[0]}, expected {batch_size}."
        )

    num_steps = u_traj.shape[1]

    if integrator not in ("euler", "rk4"):
        raise ValueError(f"Unsupported integrator '{integrator}'. Use 'euler' or 'rk4'.")

    step_fn = euler_step if integrator == "euler" else rk4_step

    x_hist = [x]
    y_hist = [system.output(x)]
    t_hist = [torch.tensor(0.0, dtype=dtype, device=device)]

    for k in range(num_steps):
        u_k = u_traj[:, k, :]
        x_next = step_fn(system, x, u_k, dt)

        if clamp_state_fn is not None:
            x_next = clamp_state_fn(x_next)

        y_next = system.output(x_next)

        x_hist.append(x_next)
        y_hist.append(y_next)
        t_hist.append(torch.tensor((k + 1) * dt, dtype=dtype, device=device))

        x = x_next

    x_out = torch.stack(x_hist, dim=1)
    y_out = torch.stack(y_hist, dim=1)
    t_out = torch.stack(t_hist, dim=0)

    if squeezed_x:
        x_out = x_out.squeeze(0)
        y_out = y_out.squeeze(0)
        u_traj = u_traj.squeeze(0)

    return {
        "t": t_out,
        "x": x_out,
        "y": y_out,
        "u": u_traj,
    }


def make_sine_input_function(
    amplitudes: Tensor,
    frequencies: Tensor,
) -> Callable[[Tensor], Tensor]:
    """
    Build an input function of the form:
        u(t) = sum_j a_j sin(w_j t)

    Supports scalar or vector-valued input.

    Case 1:
        amplitudes shape = (K,)
        frequencies shape = (K,)
        => returns scalar input shape (1,)

    Case 2:
        amplitudes shape = (input_dim, K)
        frequencies shape = (K,)
        => returns vector input shape (input_dim,)

    Returned function accepts scalar torch time t and returns tensor on same device.
    """
    if amplitudes.dim() not in (1, 2):
        raise ValueError("amplitudes must have shape (K,) or (input_dim, K).")
    if frequencies.dim() != 1:
        raise ValueError("frequencies must have shape (K,).")

    K = frequencies.shape[0]
    if amplitudes.shape[-1] != K:
        raise ValueError(
            f"Last dimension of amplitudes must match frequencies. Got {amplitudes.shape} and {frequencies.shape}."
        )

    def input_fn(t: Tensor) -> Tensor:
        t = t.to(device=frequencies.device, dtype=frequencies.dtype)
        sins = torch.sin(frequencies * t)  # (K,)

        if amplitudes.dim() == 1:
            val = torch.sum(amplitudes * sins).view(1)
        else:
            val = torch.sum(amplitudes * sins.unsqueeze(0), dim=-1)  # (input_dim,)
        return val

    return input_fn