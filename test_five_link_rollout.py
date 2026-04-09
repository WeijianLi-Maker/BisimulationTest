import torch

from systems.five_link import FiveLinkSystem, FiveLinkParams
from systems.rollout import rollout_system, make_sine_input_function


def main():
    robot = FiveLinkSystem(
        FiveLinkParams(
            lengths=(0.5, 0.5, 0.4, 0.4, 0.3),
            com_lengths=(0.25, 0.25, 0.2, 0.2, 0.15),
            masses=(2.0, 2.0, 1.5, 1.5, 1.0),
            inertias=(0.03, 0.03, 0.02, 0.02, 0.01),
            damping=(0.05, 0.05, 0.03, 0.03, 0.02),
        )
    )

    x0 = torch.tensor(
        [0.1, -0.1, 0.05, -0.05, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    freqs = torch.tensor([0.5, 1.0], dtype=torch.float32)
    amps = torch.tensor(
        [
            [0.1, -0.05],
            [0.05, 0.02],
            [-0.03, 0.04],
            [0.02, -0.01],
            [0.01, 0.03],
        ],
        dtype=torch.float32,
    )

    input_fn = make_sine_input_function(amps, freqs)

    out = rollout_system(
        system=robot,
        x0=x0,
        input_fn=input_fn,
        dt=0.01,
        num_steps=5,
        integrator="euler",
    )

    print("x shape:", out["x"].shape)
    print("y shape:", out["y"].shape)
    print("u shape:", out["u"].shape)
    print(out["x"][:2])
    print(out["y"][:2])


if __name__ == "__main__":
    main()