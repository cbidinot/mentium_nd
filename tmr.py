from typing import Dict

import torch
import torch.nn as nn
import noise_generator as noise
from torch.utils.data import DataLoader
from typing import Iterable, Optional


class TMRNoiseConfig: 
    def __init__(self, noise_sd: float, noise_inference : bool = True, noise_training: bool = False, add_one_time_noise: bool = False, add_quantization: bool = False, quantize_fn: callable = None, quantize_kwargs = None,
        include_name_contains: Optional[Iterable[str]] = None, exclude_name_contains: Optional[Iterable[str]] = None):
        self.noise_sd1 = noise_sd
        self.noise_sd2 = noise_sd
        self.noise_sd3 = noise_sd
        self.noise_training = noise_training
        self.noise_inference = noise_inference
        self.add_one_time_noise = add_one_time_noise
        self.add_quantization = add_quantization
        self.quantize_fn = quantize_fn
        self.quantize_kwargs = quantize_kwargs
        self.include_name_contains = include_name_contains
        self.exclude_name_contains = exclude_name_contains


def run_with_tmr(model: nn.Model, test_loader: DataLoader, device: torch.device, noise_config: TMRNoiseConfig) -> Dict[str, float]:
    """
    Run the model with TMR (Triple Modular Redundancy) enabled.

    Args:
        model (nn.Module): The PyTorch model to run with TMR.
        test_loader (DataLoader): The DataLoader for the test dataset.
        device (torch.device): The device to run the models on (e.g., 'cuda' or 'cpu').
        noise_config (TMRNoiseConfig): Configuration for the noise to apply to the TMR clones.

    Returns:
        Dict[str, float]: A dictionary containing the TMR accuracy, original test loss, original accuracy, TMR vs original difference, and TMR fails due to no consensus.
    """
    # Instantiate three noisy clones of the model for TMR
    hart1 = noise.clone_with_noisy_layers(model, noise_sd=noise_config.noise_sd1, add_one_time_noise=noise_config.add_one_time_noise, add_quantization=noise_config.add_quantization, quantize_fn=noise_config.quantize_fn, include_name_contains=noise_config.include_name_contains, exclude_name_contains=noise_config.exclude_name_contains)
    hart2 = noise.clone_with_noisy_layers(model, noise_sd=noise_config.noise_sd2, add_one_time_noise=noise_config.add_one_time_noise, add_quantization=noise_config.add_quantization, quantize_fn=noise_config.quantize_fn, include_name_contains=noise_config.include_name_contains, exclude_name_contains=noise_config.exclude_name_contains)
    hart3 = noise.clone_with_noisy_layers(model, noise_sd=noise_config.noise_sd3, add_one_time_noise=noise_config.add_one_time_noise, add_quantization=noise_config.add_quantization, quantize_fn=noise_config.quantize_fn, include_name_contains=noise_config.include_name_contains, exclude_name_contains=noise_config.exclude_name_contains)

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
    
    return {"tmr_accuracy": tmr_correct, "original_test_loss": test_loss, "original_accuracy": correct, "tmr_diff": tmr_diff, "tmr_fails": tmr_fails}


    