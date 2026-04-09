import torch

from systems.lip import LIPSystem
from systems.five_link import FiveLinkSystem, FiveLinkParams
from models.simulation_function import CNetwork, SimulationFunction


def main():
    lip = LIPSystem()

    robot = FiveLinkSystem(
        FiveLinkParams(
            lengths=(0.5, 0.5, 0.4, 0.4, 0.3),
            com_lengths=(0.25, 0.25, 0.2, 0.2, 0.15),
            masses=(2.0, 2.0, 1.5, 1.5, 1.0),
            inertias=(0.03, 0.03, 0.02, 0.02, 0.01),
            damping=(0.05, 0.05, 0.03, 0.03, 0.02),
        )
    )

    c_net = CNetwork()
    sim_fn = SimulationFunction(c_net, robot, lip, m=0.1)

    x = torch.randn(4, 10)
    z = torch.randn(4, 2)

    V = sim_fn(x, z)

    print("x shape:", x.shape)
    print("z shape:", z.shape)
    print("V shape:", V.shape)
    print("V =", V)
    assert V.shape == (4,)


if __name__ == "__main__":
    main()
