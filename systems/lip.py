import torch


class LIPSystem:
    """
    Linear Inverted Pendulum (LIP) model.

    State:
        z = [p, p_dot] \in R^2

    Input:
        v \in R

    Dynamics:
        ddot{p} = omega^2 (p - v),
        omega = sqrt(g / h)

    State-space form:
        dot{z} = A z + B vhenh

    Output:
        y2 = z = [p, p_dot]
    """

    def __init__(self, g: float = 9.81, h: float = 1.0, device: str = "cpu", dtype=torch.float32):
        if h <= 0:
            raise ValueError("Parameter 'h' must be positive.")

        self.g = g
        self.h = h
        self.omega = (g / h) ** 0.5
        self.device = device
        self.dtype = dtype

        self.A = torch.tensor(
            [
                [0.0, 1.0],
                [self.omega ** 2, 0.0],
            ],
            dtype=self.dtype,
            device=self.device,
        )

        self.B = torch.tensor(
            [
                [0.0],
                [-self.omega ** 2],
            ],
            dtype=self.dtype,
            device=self.device,
        )

    def dynamics(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute dz = A z + B v.

        Args:
            z: shape (..., 2)
            v: shape (..., 1) or (...,)

        Returns:
            dz: shape (..., 2)
        """
        if z.shape[-1] != 2:
            raise ValueError(f"Expected z.shape[-1] == 2, got {z.shape}.")

        if v.dim() == z.dim() - 1:
            v = v.unsqueeze(-1)

        if v.shape[-1] != 1:
            raise ValueError(f"Expected v.shape[-1] == 1, got {v.shape}.")

        dz = z @ self.A.T + v @ self.B.T
        return dz

    def output(self, z: torch.Tensor) -> torch.Tensor:
        """
        Output map y2 = g2(z) = z.

        Args:
            z: shape (..., 2)

        Returns:
            y2: shape (..., 2)
        """
        if z.shape[-1] != 2:
            raise ValueError(f"Expected z.shape[-1] == 2, got {z.shape}.")
        return z

    def to(self, device: str):
        """
        Move system tensors to a new device.
        """
        self.device = device
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        return self