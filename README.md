# MentiumND

## Introduction

This repository contains the code for a project partnering with Mentium Technologies using code from their `noise_setup` library.  This project entails injecting noise into a convolutional neural network, emulating real electrical noise that would occur in space due to radiation.  Using code from `tmr.py`, we implement a simulation of triple modular redundancy on the noisy models.  The code that simulates noise is within the program `noise_generator.py`.  The user is able to configure the noise generation in the provided `defaultcfg.json` file, with the attributes for noise configuration described below (see `noise_generator.py` under "Included Files").  There is additionally code for numerous quantization functions that can also be configured by the user.  The goal of this project is to see if and how well simulating TMR improves the performances of noisy models.  These results are shown via both console outputs displaying TMR data as well as visualizations written in the `cnn.py` file.

## Options
These are the different command line arguments supported by the program:
| Argument | Short | Type | Default | Required | Description |
|----------|-------|------|---------|----------|-------------|
| `--task` | | `str` | `"mnist"` | No | Dataset/task to use (`mnist`, `cifar10`, `cifar100`) |
| `--model` | | `str` | `"cnn"` | No | Model architecture (currently only supports `"cnn"`) |
| `--noisecfg` | `-c` | `str` | — | **Yes** | Path to noise configuration JSON file |
| `--batch_size` | | `int` | `32` | No | Number of samples per training batch |
| `--learning_rate` | | `float` | `1e-3` | No | Optimizer learning rate |
| `--epochs` | | `int` | `5` | No | Number of training epochs |

To run with default settings, simply execute `main.py` as an executable using `defaultcfg.json`:
```bash
./main.py -c ./defaultcfg.json
```
## Contents
### noise_generator.py

Implements all noise and quantization functionality. Provides `NoisyLinear` and `NoisyConv2d` layer classes that wrap their standard PyTorch equivalents and inject Gaussian noise into weights during the forward pass. Also provides `clone_with_noisy_layers`, the main function used by `tmr.py` to create noisy model copies.  Takes in as arguments:

| Field | Type | Description |
|-------|------|-------------|
| `noise_sd` | float | Standard deviation of persistent per-layer noise applied at inference |
| `one_time_sd` | float | Standard deviation of one-time parameter perturbation applied at clone time |
| `noise_inference` | bool | Enable noise during inference |
| `noise_training` | bool | Enable noise during training |
| `add_one_time_noise` | bool | Apply a one-time parameter perturbation when cloning |
| `add_quantization` | bool | Apply quantization to cloned model parameters |
| `quantize_fn` | string or null | Quantization function to use (see below) |
| `quantize_kwargs` | object or null | Arguments forwarded to the quantization function |
| `include_name_contains` | list or null | Only apply noise/quantization to layers whose name contains one of these strings |
| `exclude_name_contains` | list or null | Skip layers whose name contains one of these strings |


The `quantize_fn` can take in one of the following quantization functions:

| Value | Description | `quantize_kwargs` |
|-------|-------------|-------------------|
| `"quantile"` | Clips outliers by quantile then snaps to evenly-spaced levels | `{"num_levels": 15, "quantile": 0.01}` |
| `"symmetric"` | Symmetric uniform quantization anchored to the tensor's absolute max | `{"num_bits": 8}` |
| `"stochastic"` | Stochastic rounding — rounds up or down randomly, unbiased in expectation | `{"num_bits": 8}` |
| `"log"` | Logarithmically-spaced levels, finer resolution near zero | `{"num_bits": 4}` |

