import torch
import torch.nn as nn


class CNetwork(nn.Module):
    """
    Neural network parameterization of C_theta(x, z).

    Input:
        x: (..., 10)
        z: (..., 2)

    Output:
        C: (..., r, 2)
    """

    def __init__(
        self,
        x_dim: int = 10,
        z_dim: int = 2,
        out_rows: int = 2,
        out_cols: int = 2,
        hidden_dim: int = 32,
        num_hidden_layers: int = 1,
        activation: str = "tanh",
    ):
        super().__init__()

        self.x_dim = x_dim
        self.z_dim = z_dim
        self.out_rows = out_rows
        self.out_cols = out_cols

        in_dim = x_dim + z_dim

        if activation == "relu":
            act = nn.ReLU
        elif activation == "tanh":
            act = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = [nn.Linear(in_dim, hidden_dim), act()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers += [nn.Linear(hidden_dim, out_rows * out_cols)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.x_dim:
            raise ValueError(f"Expected x.shape[-1] == {self.x_dim}, got {x.shape}.")
        if z.shape[-1] != self.z_dim:
            raise ValueError(f"Expected z.shape[-1] == {self.z_dim}, got {z.shape}.")

        inp = torch.cat([x, z], dim=-1)
        # C = self.net(inp)
        C = 1.5 * torch.tanh(self.net(inp))
        C = C.view(*x.shape[:-1], self.out_rows, self.out_cols)
        return C


class SimulationFunction(nn.Module):
    """
    Structured simulation function

        V_theta(x, z) =
        (y1 - y2)^T (C_theta(x,z)^T C_theta(x,z) + m I) (y1 - y2)

    where
        y1 = g1(x)
        y2 = g2(z)
    """

    def __init__(self, c_net: CNetwork, five_link_system, lip_system, m: float = 0.1):
        super().__init__()
        self.c_net = c_net
        self.five_link_system = five_link_system
        self.lip_system = lip_system
        self.m = m

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y1 = self.five_link_system.output(x)   # (..., 2)
        y2 = self.lip_system.output(z)         # (..., 2)

        e = (y1 - y2).unsqueeze(-1)            # (..., 2, 1)
        C = self.c_net(x, z)                   # (..., r, 2)

        Ct = C.transpose(-1, -2)               # (..., 2, r)
        M = Ct @ C                             # (..., 2, 2)

        I = torch.eye(2, dtype=x.dtype, device=x.device)
        expand_shape = x.shape[:-1] + (2, 2)
        I = I.expand(*expand_shape)

        M = M + self.m * I

        V = e.transpose(-1, -2) @ M @ e        # (..., 1, 1)
        V = V.squeeze(-1).squeeze(-1)          # (...)

        return V
