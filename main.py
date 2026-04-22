import torch
import torch.nn as nn
import argparse
import data
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
    parser.add_argument("--noisecfg")
    parser.add_argument("")




device = torch.device("cuda")
model = MNIST().to(device)

train_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
)

test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
)

batch_size = 64
test_loader = DataLoader(test_data, batch_size=batch_size)
train_loader = DataLoader(train_data, batch_size=batch_size)

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


config = TMRNoiseConfig(noise_sd=0.1, add_one_time_noise=True, add_quantization=False, quantize_fn=lambda x, y: torch.round(x * y) / y, num_levels=16)

results = run_with_tmr(model, test_loader, device, config)
print(f"TMR Accuracy: {results.tmr_accuracy:.4f}")
print(f"Original Test Loss: {results.original_test_loss:.4f}, Original Accuracy: {results.original_accuracy:.4f}")
print(f"TMR vs Original Diff: {results.tmr_diff:.4f}, TMR Fails (No Consensus): {results.tmr_fails:.4f}")

if __name__ == "__main__":
    main()