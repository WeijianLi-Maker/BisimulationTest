import os
import sys
import json
import torch
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from systems.lip import LIPSystem
from systems.five_link import FiveLinkSystem, FiveLinkParams
from models.interface import InterfaceNet
from models.simulation_function import CNetwork, SimulationFunction
from losses.decay_loss import compute_decay_loss


def build_systems(device="cpu", dtype=torch.float32):
    lip = LIPSystem(g=9.81, h=1.0, device=device, dtype=dtype)

    five_params = FiveLinkParams(
        lengths=(0.5, 0.5, 0.4, 0.4, 0.3),
        com_lengths=(0.25, 0.25, 0.2, 0.2, 0.15),
        masses=(2.0, 2.0, 1.5, 1.5, 1.0),
        inertias=(0.03, 0.03, 0.02, 0.02, 0.01),
        damping=(0.05, 0.05, 0.03, 0.03, 0.02),
        gravity=9.81,
        B=None,
    )
    five_link = FiveLinkSystem(params=five_params, device=device, dtype=dtype)

    return lip, five_link


def save_checkpoint(path, epoch, interface_net, sim_fn, optimizer, history, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "interface_state_dict": interface_net.state_dict(),
            "sim_fn_state_dict": sim_fn.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )


def format_sci(value: float, digits: int = 2) -> str:
    """
    Format float as a * 10^b for easier magnitude inspection.
    Example: 1.23 * 10^3
    """
    if value == 0.0:
        return "0.0 * 10^0"
    s = f"{value:.{digits}e}"
    mantissa, exponent = s.split("e")
    return f"{mantissa} * 10^{int(exponent)}"


def main():
    # =========================
    # Config
    # =========================
    config = {
        "device": "cpu",
        "dtype": "float32",
        "data_path": os.path.join("data", "processed", "train_dataset.pt"),
        "batch_size": 32,
        "shuffle": True,
        "learning_rate": 2e-4,
        "num_epochs": 50,
        "r": 0.001,
        "grad_clip": 10.0,
        "m": 0.01,
        "checkpoint_dir": "checkpoints",
        "results_dir": "results",
        "checkpoint_name": "latest.pt",
        "history_name": "train_history.json",
    }

    device = config["device"]
    dtype = torch.float32 if config["dtype"] == "float32" else torch.float64

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["results_dir"], exist_ok=True)

    # =========================
    # Load data
    # =========================
    payload = torch.load(config["data_path"])
    dataset_payload = payload["dataset"]
    required_fields = {"x", "z", "u", "v"}
    missing_fields = sorted(required_fields - set(dataset_payload.keys()))
    if missing_fields:
        raise KeyError(
            f"Dataset is missing fields {missing_fields}. "
            "Regenerate it with scripts/generate_data.py."
        )

    x = dataset_payload["x"].to(dtype=dtype, device=device)
    z = dataset_payload["z"].to(dtype=dtype, device=device)
    u = dataset_payload["u"].to(dtype=dtype, device=device)
    v = dataset_payload["v"].to(dtype=dtype, device=device)

    dataset = TensorDataset(x, z, u, v)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
    )

    print("Loaded dataset:")
    print("  x shape:", x.shape)
    print("  z shape:", z.shape)
    print("  u shape:", u.shape)
    print("  v shape:", v.shape)

    # =========================
    # Build systems and models
    # =========================
    lip, five_link = build_systems(device=device, dtype=dtype)

    interface_net = InterfaceNet().to(device)
    c_net = CNetwork().to(device)
    sim_fn = SimulationFunction(c_net, five_link, lip, m=config["m"]).to(device)

    optimizer = torch.optim.Adam(
        list(interface_net.parameters()) + list(sim_fn.parameters()),
        lr= config["learning_rate"],
    )

    # =========================
    # Training history
    # =========================
    history = {
        "epoch": [],
        "avg_loss": [],
        "avg_V_mean": [],
        "avg_Vdot_mean": [],
        "avg_violation_mean": [],
        "avg_relu_violation_mean": [],
        "avg_u_data_mse": [],
    }

    # =========================
    # Training loop
    # =========================
    for epoch in range(config["num_epochs"]):
        interface_net.train()
        sim_fn.train()

        epoch_loss = 0.0
        epoch_V_mean = 0.0
        epoch_Vdot_mean = 0.0
        epoch_violation_mean = 0.0
        epoch_relu_violation_mean = 0.0
        epoch_u_data_mse = 0.0

        num_batches = 0

        for batch_x, batch_z, batch_u, batch_v in dataloader:
            batch = {
                "x": batch_x,
                "z": batch_z,
                "u": batch_u,
                "v": batch_v,
            }

            optimizer.zero_grad()

            loss, stats = compute_decay_loss(
                sim_fn=sim_fn,
                interface_net=interface_net,
                five_link_system=five_link,
                lip_system=lip,
                batch=batch,
                r=config["r"],
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(interface_net.parameters()) + list(sim_fn.parameters()),
                max_norm=config["grad_clip"],
            )

            optimizer.step()

            epoch_loss += loss.item()
            epoch_V_mean += stats["V_mean"]
            epoch_Vdot_mean += stats["Vdot_mean"]
            epoch_violation_mean += stats["violation_mean"]
            epoch_relu_violation_mean += stats["relu_violation_mean"]
            epoch_u_data_mse += stats["u_data_mse"]
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        avg_V_mean = epoch_V_mean / num_batches
        avg_Vdot_mean = epoch_Vdot_mean / num_batches
        avg_violation_mean = epoch_violation_mean / num_batches
        avg_relu_violation_mean = epoch_relu_violation_mean / num_batches
        avg_u_data_mse = epoch_u_data_mse / num_batches

        history["epoch"].append(epoch + 1)
        history["avg_loss"].append(avg_loss)
        history["avg_V_mean"].append(avg_V_mean)
        history["avg_Vdot_mean"].append(avg_Vdot_mean)
        history["avg_violation_mean"].append(avg_violation_mean)
        history["avg_relu_violation_mean"].append(avg_relu_violation_mean)
        history["avg_u_data_mse"].append(avg_u_data_mse)

        print(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"avg_loss={format_sci(avg_loss)} | "
            f"V_mean={format_sci(avg_V_mean)} | "
            f"Vdot_mean={format_sci(avg_Vdot_mean)} | "
            f"violation_mean={format_sci(avg_violation_mean)} | "
            f"u_data_mse={format_sci(avg_u_data_mse)}"
        )

        ckpt_path = os.path.join(config["checkpoint_dir"], config["checkpoint_name"])
        save_checkpoint(
            ckpt_path,
            epoch + 1,
            interface_net,
            sim_fn,
            optimizer,
            history,
            config,
        )

    # =========================
    # Save history as json
    # =========================
    history_path = os.path.join(config["results_dir"], config["history_name"])
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training finished.")
    print(f"Checkpoint saved to: {os.path.join(config['checkpoint_dir'], config['checkpoint_name'])}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main()
