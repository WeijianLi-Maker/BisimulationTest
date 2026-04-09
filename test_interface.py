import torch
from models.interface import InterfaceNet


def main():
    net = InterfaceNet()

    x = torch.randn(4, 10)
    z = torch.randn(4, 2)
    v = torch.randn(4, 1)

    u = net(x, z, v)

    print("x shape:", x.shape)
    print("z shape:", z.shape)
    print("v shape:", v.shape)
    print("u shape:", u.shape)
    print(u)


if __name__ == "__main__":
    main()