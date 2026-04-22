import torch
import torch.nn as nn
import argparse
import data
import json
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tmr import run_with_tmr, TMRNoiseConfig
from mnist import MNIST

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")
    print("Using CPU")
    return torch.device("cpu")

def main():
    # Command Line Interface
    parser = argparse.ArgumentParser(
        prog="TMRSim",
        description="TMR simulator for neural networks",
        color=True
    )

    parser.add_argument("--task", choices=["mnist"], default="mnist")
    parser.add_argument("--model", choices=["mlp"], default="mlp")
    parser.add_argument("-c", "--noisecfg", required=True)
    args = parser.parse_args()

    with open(args.noisecfg, "r") as f:
        cfg = json.load(f)


    device = get_device()
    model = MNIST().to(device)

    train_loader, test_loader = data.get_dataloaders(args.model)

    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    for epoch in range(3):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
    print("Finished training, now running with TMR...")

    try:
        config = TMRNoiseConfig(noise_sd=cfg.noise_sd, noise_inference=cfg.noise_inference, noise_training=cfg.noise_training,
        add_one_time_noise=cfg.add_one_time_noise, add_quantization=cfg.add_quantization, quantize_fn=cfg.quantize_fn, 
        include_name_contains=cfg.include_name_contains, exclude_name_contains=cfg.exclude_name_contains)
    except KeyError as err:
        print("Failed to parse config file. Make sure to follow the pattern and inlcude every option.")
        raise err

    results = run_with_tmr(model, test_loader, device, config)
    print(f"TMR Accuracy: {results.tmr_accuracy:.4f}")
    print(f"Original Test Loss: {results.original_test_loss:.4f}, Original Accuracy: {results.original_accuracy:.4f}")
    print(f"TMR vs Original Diff: {results.tmr_diff:.4f}, TMR Fails (No Consensus): {results.tmr_fails:.4f}")

if __name__ == "__main__":
    main()