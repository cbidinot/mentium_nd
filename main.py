#!/bin/python3

import torch
import argparse
import data
import json
import torch.nn as nn
from torchvision import datasets, transforms
from tmr import run_with_tmr, TMRNoiseConfig
from noise_generator import clone_with_noisy_layers, quant_fn_map
from cnn import ConvNeuralNet

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
        description="TMR simulator for neural networks"
    )

    parser.add_argument("--task", choices=["cifar10", "cifar100"], default="cifar10")
    parser.add_argument("--model", choices=["cnn"], default="cnn")
    parser.add_argument("-c", "--noisecfg", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)

    args = parser.parse_args()
    
    if args.task ==  "cifar10":
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else: # cifar100
        class_names = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'laptop', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 
    'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 
    'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm' ]
 
    with open(args.noisecfg, "r") as f:
        cfg = json.load(f)

    quantize_fn_resolved = None
    if cfg.get("quantize_fn") is not None:
        quantize_fn_resolved = quant_fn_map.get(cfg["quantize_fn"])
        if quantize_fn_resolved is None:
            raise ValueError(f"Unknown quantize_fn '{cfg['quantize_fn']}'. Choose from: {list(quant_fn_map.keys())}")

    try:
        config = TMRNoiseConfig(one_time_sd=cfg["one_time_sd"], noise_sd=cfg["noise_sd"], noise_inference=cfg["noise_inference"], noise_training=cfg["noise_training"],
        add_one_time_noise=cfg["add_one_time_noise"], add_quantization=cfg["add_quantization"], quantize_fn=quantize_fn_resolved, quantize_kwargs=cfg["quantize_kwargs"],
        include_name_contains=cfg["include_name_contains"], exclude_name_contains=cfg["exclude_name_contains"])
    except KeyError as err:
        print("Failed to parse config file. Make sure to follow the pattern and inlcude every option.")
        raise err

    device = get_device()
    model_map = {"cnn": ConvNeuralNet}
    num_classes = len(class_names)
    model = model_map[args.model](num_classes=num_classes).to(device)

    train_loader, test_loader = data.get_dataloaders(args.task, train_batch_size=args.batch_size, test_batch_size=args.batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay = 0.005, momentum = 0.9)
   
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    if cfg["noise_training"]:
        # add noise but not one time noise
        model = clone_with_noisy_layers(model, one_time_noise_sd=0.0, layer_noise_sd=config.noise_sd, add_one_time_noise=config.add_one_time_noise, add_quantization=config.add_quantization, quantize_fn=config.quantize_fn, include_name_contains=config.include_name_contains, exclude_name_contains=config.exclude_name_contains)
    
    model.train()
    print("Starting training...")

    for epoch in range(args.epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            loss = criterion(model(inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    cnn(device, test_loader, model, test_dataset, train_dataset, class_names)
    
    print("Finished training, now running with TMR...")
    results = run_with_tmr(model, test_loader, device, config)

    print(f"TMR Accuracy: {results["tmr_accuracy"]:.4f}")
    print(f"Original Test Loss: {results["original_test_loss"]:.4f}, Original Accuracy: {results["original_accuracy"]:.4f}")
    print(f"TMR vs Original Diff: {results["tmr_diff"]:.4f}, TMR Fails (No Consensus): {results["tmr_fails"]:.4f}")

if __name__ == "__main__":
    main()
