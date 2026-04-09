import torch

from systems.lip import LIPSystem
from systems.rollout import rollout_system, make_sine_input_function


def main():
    print("==== Start Test LIP Rollout ====")

    # ===== 1. 初始化系统 =====
    lip = LIPSystem()

    # ===== 2. 初始状态 =====
    z0 = torch.tensor([0.2, 0.0], dtype=torch.float32)

    # ===== 3. 构造正弦输入 =====
    freqs = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    amps = torch.tensor([0.2, -0.1, 0.05], dtype=torch.float32)

    input_fn = make_sine_input_function(amps, freqs)

    # ===== 4. rollout =====
    result = rollout_system(
        system=lip,
        x0=z0,
        input_fn=input_fn,
        dt=0.01,
        num_steps=200,
        integrator="rk4",
    )

    t = result["t"]
    x = result["x"]
    y = result["y"]
    u = result["u"]

    # ===== 5. 打印结果 =====
    print("t shape:", t.shape)
    print("x shape:", x.shape)
    print("y shape:", y.shape)
    print("u shape:", u.shape)

    print("\nFirst 5 states:")
    print(x[:5])

    print("\nFirst 5 outputs:")
    print(y[:5])

    print("\nFirst 5 inputs:")
    print(u[:5])

    print("==== Test Finished ====")


if __name__ == "__main__":
    main()