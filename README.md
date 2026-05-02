# What's Included in this Repo:

## noise_generator.py

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

