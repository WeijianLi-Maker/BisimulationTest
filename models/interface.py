import torch
import torch.nn as nn


class InterfaceNet(nn.Module):
    """
    Neural network parameterization of the interface function

        u = pi_phi(x, z, v)

    Inputs:
        x: (..., 10)
        z: (..., 2)
        v: (..., 1)

    Output:
        u: (..., 5)
    """

    def __init__(
        self,
        x_dim: int = 10,
        z_dim: int = 2,
        v_dim: int = 1,
        u_dim: int = 5,
        hidden_dim: int = 32,
        num_hidden_layers: int = 1,
        activation: str = "tanh",
    ):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.v_dim = v_dim
        self.u_dim = u_dim

        in_dim = x_dim + z_dim + v_dim

        if activation == "relu":
            act = nn.ReLU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = [nn.Linear(in_dim, hidden_dim), act()]

        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]

        layers += [nn.Linear(hidden_dim, u_dim)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: shape (..., 10)
            z: shape (..., 2)
            v: shape (..., 1) or (...,)

        Returns:
            u: shape (..., 5)
        """
        if x.shape[-1] != self.x_dim:
            raise ValueError(f"Expected x.shape[-1] == {self.x_dim}, got {x.shape}.")
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"Expected z.shape[-1] == {self.z_dim}, got {z.shape}.")

        if v.dim() == x.dim() - 1:
            v = v.unsqueeze(-1)

        if v.shape[-1] != self.v_dim:
            raise ValueError(f"Expected v.shape[-1] == {self.v_dim}, got {v.shape}.")

        inp = torch.cat([x, z, v], dim=-1)
        u = torch.tanh(self.net(inp))
        return u
