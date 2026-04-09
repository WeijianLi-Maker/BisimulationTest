# systems/five_link.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class FiveLinkParams:
    """
    Parameters for a planar 5-link serial robot with revolute joints.

    Convention:
    - q_i are relative joint angles
    - absolute angle of link i is theta_i = q_1 + ... + q_i
    - base is fixed at the origin
    - z is vertical upward
    - x is horizontal

    lengths: full link lengths
    com_lengths: distance from proximal joint to COM of each link
    masses: link masses
    inertias: planar rotational inertia of each link about its COM
    damping: viscous joint damping coefficients
    gravity: gravitational acceleration
    B: input matrix
    """
    lengths: Tuple[float, float, float, float, float] = (0.5, 0.5, 0.5, 0.5, 0.5)
    com_lengths: Tuple[float, float, float, float, float] = (0.25, 0.25, 0.25, 0.25, 0.25)
    masses: Tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)
    inertias: Tuple[float, float, float, float, float] = (0.02, 0.02, 0.02, 0.02, 0.02)
    damping: Tuple[float, float, float, float, float] = (0.02, 0.02, 0.02, 0.02, 0.02)
    gravity: float = 9.81
    B: Optional[Tuple[Tuple[float, ...], ...]] = None


class FiveLinkSystem:
    """
    Differentiable planar 5-link robot model in PyTorch.

    State:
        x = [q, qdot] in R^10

    Input:
        u in R^5

    Dynamics:
        M(q) qddot + h(q, qdot) + G(q) + D qdot = B u

    Output:
        y1 = [p_com_x, v_com_x] in R^2

    Notes:
    - This is a fixed-base planar serial 5R robot model.
    - If your robot is a walking five-link with contact switching / floating base,
      you will need to modify coordinates and possibly the input matrix B.
    """

    def __init__(
        self,
        params: Optional[FiveLinkParams] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.params = params if params is not None else FiveLinkParams()
        self.device = device
        self.dtype = dtype

        self.nq = 5
        self.nx = 10
        self.nu = 5

        self.lengths = torch.tensor(self.params.lengths, dtype=dtype, device=device)
        self.com_lengths = torch.tensor(self.params.com_lengths, dtype=dtype, device=device)
        self.masses = torch.tensor(self.params.masses, dtype=dtype, device=device)
        self.inertias = torch.tensor(self.params.inertias, dtype=dtype, device=device)
        self.damping = torch.tensor(self.params.damping, dtype=dtype, device=device)
        self.gravity = torch.tensor(self.params.gravity, dtype=dtype, device=device)

        if self.params.B is None:
            self.B = torch.eye(self.nu, dtype=dtype, device=device)
        else:
            B = torch.tensor(self.params.B, dtype=dtype, device=device)
            if B.shape != (self.nq, self.nu):
                raise ValueError(f"B must have shape {(self.nq, self.nu)}, got {B.shape}.")
            self.B = B

    # -------------------------------------------------------------------------
    # Basic utilities
    # -------------------------------------------------------------------------
    def to(self, device: str) -> "FiveLinkSystem":
        self.device = device
        self.lengths = self.lengths.to(device)
        self.com_lengths = self.com_lengths.to(device)
        self.masses = self.masses.to(device)
        self.inertias = self.inertias.to(device)
        self.damping = self.damping.to(device)
        self.gravity = self.gravity.to(device)
        self.B = self.B.to(device)
        return self

    def split_state(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.nx:
            raise ValueError(f"Expected x.shape[-1] == {self.nx}, got {x.shape}.")
        q = x[..., :self.nq]
        qdot = x[..., self.nq:]
        return q, qdot

    def _ensure_batch(self, tensor: torch.Tensor, last_dim: int) -> Tuple[torch.Tensor, bool]:
        if tensor.shape[-1] != last_dim:
            raise ValueError(f"Expected last dimension {last_dim}, got {tensor.shape}.")
        squeezed = False
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
            squeezed = True
        return tensor, squeezed

    # -------------------------------------------------------------------------
    # Kinematics
    # -------------------------------------------------------------------------
    def absolute_angles(self, q: torch.Tensor) -> torch.Tensor:
        """
        theta_i = q_1 + ... + q_i
        q: (..., 5)
        returns theta: (..., 5)
        """
        return torch.cumsum(q, dim=-1)

    def joint_positions(self, q: torch.Tensor) -> torch.Tensor:
        """
        Returns positions of the six joints:
            p_0, p_1, ..., p_5
        where p_0 = base origin, p_5 = distal end of link 5

        Shape: (..., 6, 2)
        """
        q, squeezed = self._ensure_batch(q, self.nq)
        theta = self.absolute_angles(q)

        batch = q.shape[0]
        pos = torch.zeros(batch, 6, 2, dtype=q.dtype, device=q.device)

        current = torch.zeros(batch, 2, dtype=q.dtype, device=q.device)
        pos[:, 0, :] = current

        for i in range(self.nq):
            dx = self.lengths[i] * torch.sin(theta[:, i])
            dz = self.lengths[i] * torch.cos(theta[:, i])
            current = current + torch.stack([dx, dz], dim=-1)
            pos[:, i + 1, :] = current

        if squeezed:
            pos = pos.squeeze(0)
        return pos

    def com_positions(self, q: torch.Tensor) -> torch.Tensor:
        """
        COM position of each link.

        Shape: (..., 5, 2)
        """
        q, squeezed = self._ensure_batch(q, self.nq)
        theta = self.absolute_angles(q)
        joints = self.joint_positions(q)  # (B, 6, 2)

        batch = q.shape[0]
        coms = torch.zeros(batch, self.nq, 2, dtype=q.dtype, device=q.device)

        for i in range(self.nq):
            base_i = joints[:, i, :]  # proximal joint of link i
            dx = self.com_lengths[i] * torch.sin(theta[:, i])
            dz = self.com_lengths[i] * torch.cos(theta[:, i])
            coms[:, i, :] = base_i + torch.stack([dx, dz], dim=-1)

        if squeezed:
            coms = coms.squeeze(0)
        return coms

    def total_com_position(self, q: torch.Tensor) -> torch.Tensor:
        """
        Total center-of-mass position of the robot.

        Shape: (..., 2)
        """
        q, squeezed = self._ensure_batch(q, self.nq)
        coms = self.com_positions(q)  # (B, 5, 2)
        total_mass = torch.sum(self.masses)
        weighted = coms * self.masses.view(1, self.nq, 1)
        p_com = weighted.sum(dim=1) / total_mass

        if squeezed:
            p_com = p_com.squeeze(0)
        return p_com

    # -------------------------------------------------------------------------
    # Jacobians
    # -------------------------------------------------------------------------
    def _link_com_jacobians_single(self, q_single: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Jacobians for a single q.

        Returns:
            Jv: (5, 2, 5)  translational Jacobian of each link COM
            Jw: (5, 5)     angular velocity mapping row for each link
                          where omega_i = Jw[i] @ qdot
        """
        if q_single.dim() != 1 or q_single.shape[0] != self.nq:
            raise ValueError(f"Expected q_single shape ({self.nq},), got {q_single.shape}.")

        q_req = q_single.clone().detach().requires_grad_(True)

        coms = self.com_positions(q_req).squeeze(0) if q_req.dim() == 1 else self.com_positions(q_req)
        if coms.dim() == 3:
            coms = coms.squeeze(0)

        Jv = []
        for i in range(self.nq):
            grads = []
            for coord in range(2):
                grad_i = torch.autograd.grad(
                    coms[i, coord],
                    q_req,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=False,
                )[0]
                grads.append(grad_i)
            J_i = torch.stack(grads, dim=0)  # (2, 5)
            Jv.append(J_i)
        Jv = torch.stack(Jv, dim=0)  # (5, 2, 5)

        Jw = torch.zeros(self.nq, self.nq, dtype=q_single.dtype, device=q_single.device)
        for i in range(self.nq):
            Jw[i, : i + 1] = 1.0

        return Jv, Jw

    def com_velocity(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Total COM velocity.

        Shape: (..., 2)
        """
        q, squeezed_q = self._ensure_batch(q, self.nq)
        qdot, squeezed_qdot = self._ensure_batch(qdot, self.nq)
        if q.shape[0] != qdot.shape[0]:
            raise ValueError("q and qdot batch sizes must match.")

        total_mass = torch.sum(self.masses)
        velocities = []

        for b in range(q.shape[0]):
            Jv, _ = self._link_com_jacobians_single(q[b])
            v_links = torch.einsum("lij,j->li", Jv, qdot[b])  # (5, 2)
            v_com = (v_links * self.masses.view(-1, 1)).sum(dim=0) / total_mass
            velocities.append(v_com)

        vel = torch.stack(velocities, dim=0)

        if squeezed_q and squeezed_qdot:
            vel = vel.squeeze(0)
        return vel

    # -------------------------------------------------------------------------
    # Dynamics terms
    # -------------------------------------------------------------------------
    def mass_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        M(q), shape (..., 5, 5)
        """
        q, squeezed = self._ensure_batch(q, self.nq)
        Ms = []

        for b in range(q.shape[0]):
            Jv, Jw = self._link_com_jacobians_single(q[b])

            M = torch.zeros(self.nq, self.nq, dtype=q.dtype, device=q.device)
            for i in range(self.nq):
                M = M + self.masses[i] * (Jv[i].transpose(0, 1) @ Jv[i])
                M = M + self.inertias[i] * torch.outer(Jw[i], Jw[i])

            Ms.append(M)

        M = torch.stack(Ms, dim=0)
        if squeezed:
            M = M.squeeze(0)
        return M

    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """
        U(q) = sum_i m_i g z_i
        Shape: (...,)
        """
        q, squeezed = self._ensure_batch(q, self.nq)
        coms = self.com_positions(q)  # (B, 5, 2)
        z_coords = coms[..., 1]
        U = torch.sum(self.masses.view(1, -1) * self.gravity * z_coords, dim=-1)

        if squeezed:
            U = U.squeeze(0)
        return U

    def gravity_vector(self, q: torch.Tensor) -> torch.Tensor:
        """
        G(q) = dU/dq
        Shape: (..., 5)
        """
        q, squeezed = self._ensure_batch(q, self.nq)
        Gs = []

        for b in range(q.shape[0]):
            qb = q[b].clone().detach().requires_grad_(True)
            U = self.potential_energy(qb)
            if U.dim() > 0:
                U = U.squeeze()
            G = torch.autograd.grad(U, qb, create_graph=True)[0]
            Gs.append(G)

        G = torch.stack(Gs, dim=0)
        if squeezed:
            G = G.squeeze(0)
        return G

    def coriolis_centrifugal(self, q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
        """
        Computes C(q, qdot) qdot using Christoffel symbols.

        Shape: (..., 5)
        """
        q, squeezed_q = self._ensure_batch(q, self.nq)
        qdot, squeezed_qdot = self._ensure_batch(qdot, self.nq)
        if q.shape[0] != qdot.shape[0]:
            raise ValueError("q and qdot batch sizes must match.")

        h_terms = []

        for b in range(q.shape[0]):
            qb = q[b].clone().detach().requires_grad_(True)
            qdb = qdot[b]

            M = self.mass_matrix(qb)
            if M.dim() == 3:
                M = M.squeeze(0)

            # dM[k] = dM/dq_k, shape (5,5)
            dM = []
            for i in range(self.nq):
                row_grads = []
                for j in range(self.nq):
                    grad_ij = torch.autograd.grad(
                        M[i, j],
                        qb,
                        retain_graph=True,
                        create_graph=True,
                        allow_unused=True,
                    )[0]

                    if grad_ij is None:
                        grad_ij = torch.zeros_like(qb)

                    row_grads.append(grad_ij)

                row_grads = torch.stack(row_grads, dim=0)  # (5, 5)
                dM.append(row_grads)

            dM = torch.stack(dM, dim=0)  # (5, 5, 5)

            # h_i = sum_jk 1/2 (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) qd_j qd_k
            h = torch.zeros(self.nq, dtype=q.dtype, device=q.device)
            for i in range(self.nq):
                accum = 0.0
                for j in range(self.nq):
                    for k in range(self.nq):
                        gamma = 0.5 * (dM[i, j, k] + dM[i, k, j] - dM[j, k, i])
                        accum = accum + gamma * qdb[j] * qdb[k]
                h[i] = accum

            h_terms.append(h)

        h = torch.stack(h_terms, dim=0)
        if squeezed_q and squeezed_qdot:
            h = h.squeeze(0)
        return h

    # -------------------------------------------------------------------------
    # System maps
    # -------------------------------------------------------------------------
    def output(self, x: torch.Tensor) -> torch.Tensor:
        """
        y1 = [p_com_x, v_com_x]

        Input:
            x: (..., 10)

        Output:
            y1: (..., 2)
        """
        x, squeezed = self._ensure_batch(x, self.nx)
        q, qdot = self.split_state(x)

        p_com = self.total_com_position(q)   # (..., 2)
        v_com = self.com_velocity(q, qdot)   # (..., 2)

        y1 = torch.stack([p_com[..., 0], v_com[..., 0]], dim=-1)

        if squeezed:
            y1 = y1.squeeze(0)
        return y1

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Computes xdot = [qdot, qddot].

        Input:
            x: (..., 10)
            u: (..., 5)

        Output:
            xdot: (..., 10)
        """
        x, squeezed_x = self._ensure_batch(x, self.nx)
        u, squeezed_u = self._ensure_batch(u, self.nu)

        if x.shape[0] != u.shape[0]:
            raise ValueError("x and u batch sizes must match.")

        q, qdot = self.split_state(x)

        dx_list = []
        for b in range(x.shape[0]):
            qb = q[b]
            qdb = qdot[b]
            ub = u[b]

            M = self.mass_matrix(qb)
            if M.dim() == 3:
                M = M.squeeze(0)

            h = self.coriolis_centrifugal(qb, qdb)
            if h.dim() == 2:
                h = h.squeeze(0)

            G = self.gravity_vector(qb)
            if G.dim() == 2:
                G = G.squeeze(0)

            Dq = self.damping * qdb
            tau = self.B @ ub

            rhs = tau - h - G - Dq
            qdd = torch.linalg.solve(M, rhs)

            dxb = torch.cat([qdb, qdd], dim=-1)
            dx_list.append(dxb)

        dx = torch.stack(dx_list, dim=0)

        if squeezed_x and squeezed_u:
            dx = dx.squeeze(0)
        return dx

    # -------------------------------------------------------------------------
    # Convenience helpers
    # -------------------------------------------------------------------------
    def drift_and_actuation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns f0(x), G(x) such that:
            xdot = f0(x) + G(x) u

        Useful later if you want an explicit control-affine form.
        """
        x, squeezed = self._ensure_batch(x, self.nx)
        batch = x.shape[0]

        zeros_u = torch.zeros(batch, self.nu, dtype=x.dtype, device=x.device)
        f0 = self.dynamics(x, zeros_u)

        Gcols = []
        for i in range(self.nu):
            ei = torch.zeros(batch, self.nu, dtype=x.dtype, device=x.device)
            ei[:, i] = 1.0
            fi = self.dynamics(x, ei)
            Gcols.append((fi - f0).unsqueeze(-1))

        Gx = torch.cat(Gcols, dim=-1)  # (B, 10, 5)

        if squeezed:
            f0 = f0.squeeze(0)
            Gx = Gx.squeeze(0)
        return f0, Gx