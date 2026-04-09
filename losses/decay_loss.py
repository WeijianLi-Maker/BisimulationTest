import torch
import torch.nn.functional as F


def compute_decay_loss(
    sim_fn,
    interface_net,
    five_link_system,
    lip_system,
    batch,
    r: float,
):
    """
    Compute decay loss

        L_dec = mean( relu( Vdot + r V ) )

    batch should contain:
        batch["x"]: (B, 10)
        batch["z"]: (B, 2)
        batch["v"]: (B, 1)
    """
    x = batch["x"].clone().detach().requires_grad_(True)
    z = batch["z"].clone().detach().requires_grad_(True)
    v = batch["v"].clone().detach()

    # V(x, z)
    V = sim_fn(x, z)   # (B,)

    # interface u = pi(x, z, v)
    u = interface_net(x, z, v)   # (B, 5)

    # dynamics
    fx = five_link_system.dynamics(x, u)   # (B, 10)
    fz = lip_system.dynamics(z, v)         # (B, 2)

    # gradients
    dV_dx = torch.autograd.grad(
        V.sum(),
        x,
        create_graph=True,
        retain_graph=True,
    )[0]

    dV_dz = torch.autograd.grad(
        V.sum(),
        z,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Vdot = dV/dx * fx + dV/dz * fz
    Vdot = (dV_dx * fx).sum(dim=-1) + (dV_dz * fz).sum(dim=-1)
    violation = Vdot + r * V
    # loss = F.relu(violation).mean()

    u_reg = 1e-4 * (u ** 2).mean()
    loss = F.relu(violation).mean() + u_reg

    stats = {
        "loss": loss.item(),
        "V_mean": V.mean().item(),
        "Vdot_mean": Vdot.mean().item(),
        "violation_mean": violation.mean().item(),
        "relu_violation_mean": F.relu(violation).mean().item(),
    }

    return loss, stats
