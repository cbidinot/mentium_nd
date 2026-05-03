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
### Changing the Noise Configuration
Every setting related to noise generation can be configured through the noise configuration JSON file. These are the options available to tweak in the JSON file, and they must all be present at all times for parsing to work.
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `one_time_sd` | `float` | `1e-2` | Std dev of one-time noise injected into weights at initialization (simulates permanent hardware faults) |
| `noise_sd` | `float` | `1e-2` | Std dev of noise injected per forward pass  |
| `noise_inference` | `bool` | `true` | Whether to inject noise during inference |
| `noise_training` | `bool` | `false` | Whether to inject noise during training |
| `add_one_time_noise` | `bool` | `false` | Whether to apply one-time permanent noise to weights at startup |
| `add_quantization` | `bool` | `false` | Whether to apply quantization as an additional fault model |
| `quantize_fn` | `str \| null` | `null` | Name of the quantization function to use (if `add_quantization` is `true`) |
| `quantize_kwargs` | `dict \| null` | `null` | Keyword arguments passed to `quantize_fn` |
| `include_name_contains` | `list \| null` | `null` | Include layers whose names are on the list (by default every layer is included)
| `exclude_name_contains` | `list \| null` | `null` | Skip noise injection for layers whose names are on the list |

These are the available options and expected arguments for the quantization function. If no arguments are provided through `quantize_kwargs`, default values will be used.
| Function | Argument | Type | Default | Description |
|----------|----------|------|---------|-------------|
| `quantile` | | | | Quantizes by snapping values to levels derived from the tensor's own distribution, clipping outliers first |
| | `num_levels` | `int` | `15` | Number of evenly-spaced quantization levels to generate |
| | `quantile` | `float` | `0.01` | Fraction of values clipped from each tail before computing levels (e.g. `0.01` clips bottom and top 1%) |
| `symmetric` | | | | Symmetric uniform quantization anchored to the tensor's absolute max, mimicking fixed bit-width hardware (e.g. INT8) |
| | `num_bits` | `int` | `8` | Number of bits to simulate; determines `2^num_bits - 1` quantization levels |
| `stochastic` | | | | Like symmetric quantization but rounds up/down randomly, producing an unbiased estimator — better suited for training |
| | `num_bits` | `int` | `8` | Number of bits to simulate; determines `2^num_bits` quantization levels |
| `log` | | | | Quantizes on a log scale, giving finer resolution near zero; well suited for normally-distributed weights |
| | `num_bits` | `int` | `4` | Number of bits to simulate; determines `2^num_bits` log-spaced levels |
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

### tmr.py
 
Implements the TMR evaluation logic. Contains two components: `TMRNoiseConfig`, a configuration class for noise settings, and `run_with_tmr`, the main function that runs the TMR simulation.
 
#### How TMR is Simulated
 
`run_with_tmr` takes a trained model and creates three independent noisy clones of it using `clone_with_noisy_layers` from `noise_generator.py`. These clones represent three redundant hardware units (harts), each affected by independent noise. For every batch in the test set, all three clones produce predictions independently. Majority voting is then applied — if at least two of the three clones agree on a prediction, that prediction is used as the TMR output. If all three disagree and no consensus is reached, the sample is marked as `-1` and excluded from the accuracy calculation rather than counted as an error. The original unmodified model is also evaluated on the same batches in parallel for comparison.
 
#### Output
 
| Field | Description |
|-------|-------------|
| `tmr_accuracy` | Accuracy of the majority vote, computed only over samples where consensus was reached |
| `original_accuracy` | Accuracy of the unmodified base model |
| `original_test_loss` | Cross-entropy loss of the unmodified base model |
| `tmr_diff` | Fraction of samples where the TMR output differs from the base model |
| `tmr_fails` | Fraction of samples where all three clones disagreed and no consensus was reached |
