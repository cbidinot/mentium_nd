import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tmr import run_with_tmr, TMRNoiseConfig

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

device = torch.device("cuda")
model = NeuralNet().to(device)

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