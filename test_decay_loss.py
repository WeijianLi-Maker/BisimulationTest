import torch

from systems.lip import LIPSystem
from systems.five_link import FiveLinkSystem, FiveLinkParams
from models.interface import InterfaceNet
from models.simulation_function import CNetwork, SimulationFunction
from losses.decay_loss import compute_decay_loss


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

    interface_net = InterfaceNet()
    c_net = CNetwork()
    sim_fn = SimulationFunction(c_net, robot, lip, m=0.1)

    batch = {
        "x": torch.randn(4, 10),
        "z": torch.randn(4, 2),
        "v": torch.randn(4, 1),
    }

    loss, stats = compute_decay_loss(
        sim_fn=sim_fn,
        interface_net=interface_net,
        five_link_system=robot,
        lip_system=lip,
        batch=batch,
        r=0.5,
    )

    print("loss =", loss)
    print("stats =", stats)

    loss.backward()
    print("backward passed")


if __name__ == "__main__":
    main()