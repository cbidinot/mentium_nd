from typing import Dict

import torch
import torch.nn as nn
import noise_generator as noise
from torch.utils.data import DataLoader


def run_with_tmr(model: nn.Model, test_loader: DataLoader, device: torch.device):
    """
    Run the model with TMR (Triple Modular Redundancy) enabled.

    Args:
        model (nn.Module): The PyTorch model to run with TMR.

    Returns:
        (float, float, float, float, float): A tuple containing:
            - TMR Accuracy: The accuracy of the TMR output compared to the labels.
            - Original Test Loss: The average loss of the original model on the test set.
            - Original Accuracy: The accuracy of the original model on the test set.
            - TMR vs Original Diff: The percentage of test samples where the TMR output differs from the original model's output.
            - TMR Fails (No Consensus): The percentage of test samples where TMR failed to reach a consensus (i.e., all three harts produced different outputs).
    """
    # Instantiate three noisy clones of the model for TMR
    hart1 = noise.clone_with_noisy_layers(model, noise_sd=0.1)
    hart2 = noise.clone_with_noisy_layers(model, noise_sd=0.2)
    hart3 = noise.clone_with_noisy_layers(model, noise_sd=0.1)

    # Move all models to the specified device
    hart1.to(device)
    hart2.to(device)
    hart3.to(device)
    model.to(device)

    hart1.eval()
    hart2.eval()
    hart3.eval()
    model.eval()
    tmr_correct, test_loss, correct, tmr_diff, tmr_fails, tmr_succeses = 0, 0, 0, 0, 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Evaluate each hart independently, plus the original model for comparison
            outputs1 = hart1(inputs)
            outputs2 = hart2(inputs)
            outputs3 = hart3(inputs)
            preds1 = outputs1.argmax(dim=1)
            preds2 = outputs2.argmax(dim=1)
            preds3 = outputs3.argmax(dim=1)
            outputs_base = model(inputs)
            
            # Perform majority voting across the three harts
            stacked_preds = torch.stack([preds1, preds2, preds3], dim=0) 
            mode_preds = torch.mode(stacked_preds, dim=0).values
            counts = (stacked_preds == mode_preds.unsqueeze(0)).sum(dim=0)
            tmr_preds = torch.where(counts >= 2, mode_preds, torch.full_like(preds1, -1))  # -1 indicates no consensus

            # Evaluate the TMR output against the labels, ignoring cases with no consensus (auto fail)
            mask = (tmr_preds != -1)
            tmr_correct += (tmr_preds[mask] == labels[mask]).sum().item()
            tmr_succeses += mask.sum().item()  # Count only cases where TMR had a consensus


            # Evaluate the original model for comparison
            test_loss += criterion(outputs_base, labels).item()
            correct += (outputs_base.argmax(1) == labels).sum().item()

            # Evaluate how many timees the TMR output differs from the original model's output
            tmr_diff += (tmr_preds != outputs_base.argmax(1)).sum().item()

            # Evaluate how many times TMR failed due to no consensus
            tmr_fails += (tmr_preds == -1).type(torch.float).sum().item()

    tmr_correct /= tmr_succeses  # Accuracy is based on cases where TMR had a consensus
    test_loss /= len(test_loader)
    correct /= len(test_loader.dataset)
    tmr_diff /= len(test_loader.dataset)
    tmr_fails /= len(test_loader.dataset)
    print(f"TMR Accuracy: {tmr_correct:.4f}")
    print(f"Original Test Loss: {test_loss:.4f}, Original Accuracy: {correct:.4f}")
    print(f"TMR vs Original Diff: {tmr_diff:.4f}, TMR Fails (No Consensus): {tmr_fails:.4f}")
    
    return (tmr_correct, test_loss, correct, tmr_diff, tmr_fails)
    


    