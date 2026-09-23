# Fusion of Neural Nonlinearity with Synaptic Weighting in Memristive Crossbar Array via Controlled Signal-to-Noise Ratio

This repository provides the Python implementations accompanying Supplementary Software 1 and 2 of the manuscript. It includes a compact stochastic model for resistive random-access memory (RRAM), a memristor-aware neural-network training and inference framework, and a self-contained stochastic activation demonstration.

The framework explores intrinsic read noise as a computational resource for stochastic-sampled normalized sigmoid (SNS) activation. The accompanying manuscript and Supplementary Information describe the device measurements, circuit mapping and hardware evaluation.

## 1. Repository contents

| File | Purpose |
| --- | --- |
| `compact_rram_stochastic_model.py` | Supplementary Software 1: two-state read-current fluctuations with Arrhenius-type dwell times and state-dependent Gaussian noise. |
| `main.py` | Supplementary Software 2: CIFAR-10 data preparation, hardware-aware training, checkpointing and repeated inference. |
| `bs_resnet.py` | Network construction and training, testing and cumulative class-voting routines. |
| `bs_activation.py` | Stochastic binary and continuous activation functions and custom gradient calculations. |
| `memristor_optimizer.py` | Weight-to-conductance mapping, conductance updates, pulse-accumulating optimizers and potentiation/depression plots. |
| `run_demo.py` | Standalone illustration of stochastic comparator sampling and a logistic approximation. |
| `requirements.txt` | Python dependencies. |
| `LICENSE` | MIT License. |

## 2. Setup

A standard computer is sufficient for the two short demonstrations. Neural-network training benefits from a supported GPU. `main.py` selects CUDA, then Apple MPS, then CPU according to availability. The network workflow uses PyTorch and TorchVision; the short demonstrations need only NumPy and Matplotlib.

### Obtain the code

Clone this repository, or download and extract its ZIP archive. Run commands from the directory containing `main.py` and `requirements.txt`.

```bash
git clone https://github.com/ZiangChen77/RRAM-Stochastic-Activation.git
cd RRAM-Stochastic-Activation
```

### Create an environment

Use a Python version supported by the PyTorch/TorchVision packages selected for your operating system and hardware, then create a virtual environment:

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or in a Linux/macOS shell:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

For a quick run of only the two demonstrations:

```bash
python -m pip install numpy matplotlib
```

For GPU use, select compatible PyTorch and TorchVision builds using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). Dependency installation time depends on the package downloads and the selected CPU/GPU build.

## 3. Quick stochastic activation demonstration

```bash
python run_demo.py
```

At each fixed input value, the script adds Gaussian noise, performs binary comparator decisions and estimates the output probability from their sample mean. It overlays a logistic approximation to the normal cumulative distribution function (CDF). The comparator threshold, noise amplitude and voltage scale are illustrative demonstration settings.

The default settings are exposed in `DEMO_PARAMETERS`:

| Setting | Value |
| --- | --- |
| Random seed | 1234 |
| Comparator threshold | 0.24 V |
| Gaussian noise standard deviation | 0.425 V |
| Input range | -1.5 to 1.5 V |
| Input points | 100 |
| Binary samples per fixed input | 5,000 |
| Logistic gain | 4.0 V^-1 |

Colors, fonts, line width and output resolution are exposed in `PLOT_PARAMETERS`.

Expected console output:

```text
[INFO] Initializing stochastic comparator demo...
[INFO] Illustrative demonstration parameters:
       - Random seed: 1234
       - Comparator threshold (V_bias): 0.24 V
       - Gaussian noise standard deviation: 0.425 V
       - Binary samples per fixed input: 5000
[SUCCESS] Demo completed. Plot saved as 'stochastic_activation_demo.png'.
```

The PNG is saved in the working directory. A fixed random seed makes the sampled result repeatable in the same NumPy environment.

## 4. Compact RRAM stochastic model — Supplementary Software 1

```bash
python compact_rram_stochastic_model.py
```

The script generates `Dynamic_Noise_Model.png` in the working directory and displays the current trace. Its defaults simulate 0.2 s with a 10-microsecond sampling interval, yielding 20,000 samples at a read voltage of 0.0765 V and a nominal resistance of 45 kOhm.

The model parameters are collected at the top of the script:

| Parameter | Role |
| --- | --- |
| `W_ON`, `W_OFF`, `TAU_0`, `TEMP`, `K_B` | Arrhenius-type mean dwell times: `tau = TAU_0 * exp(W / (K_B * TEMP))`. |
| `V_READ`, `R_DEV` | Nominal read current: `I = V_READ / R_DEV`. |
| `OFFSET_MEAN_R` | Relative separation of the high- and low-current state means. |
| `OFFSET_BASE_MEAN` | State-mean fluctuation scale when entering a state. |
| `NOISE_AMPLITUDE` | Gaussian noise standard deviation relative to the current mean of the active state. |
| `TSAMPLE`, `TOTAL_TIME` | Sampling interval and simulated duration. |

Consistent with Supplementary Software 1, the high-to-low and low-to-high transition probabilities are `1 - exp(-TSAMPLE / tau_ON)` and `1 - exp(-TSAMPLE / tau_OFF)`, respectively. A transition occurs when a uniform random draw is below the relevant probability. The Gaussian noise scale uses the mean of the active state.

## 5. Hardware-aware training and inference — Supplementary Software 2

```bash
python main.py
```

### Inputs and default workflow

1. Download CIFAR-10 to `./data` on the first run. Training uses random cropping and horizontal flipping; training and test images use the normalization specified in `main.py`.
2. Construct the network with `Sigmoid_BS` activation and attach the memristor model to its parameters.
3. Train for up to 200 epochs, using software batch sizes of 128 for training and 100 for testing.
4. Evaluate stochastic inference and continuous-sigmoid inference with the same network weights at each epoch; the continuous-sigmoid model receives the current trained state through `load_state_dict`.
5. After training, evaluate up to 50 sequential stochastic passes with cumulative voting over the predicted class of each image. Existing saved repetition results are reused when available.

### Conductance mapping

The conductance values in `main.py`, the `Memristor` constructor and its standalone example match the final Supplementary Software 2:

```python
Gmin = 2.0e-6  # S: 2 microSiemens
Gmax = 3.2e-5  # S: 32 microSiemens
```

For `Wmin = -1.0` and `Wmax = 1.0`, the mapping is:

```text
Gscale = (Gmax - Gmin) / (Wmax - Wmin) = 15e-6 S
Gref   = (Gmin + Gmax) / 2            = 17e-6 S
G      = Gref + Gscale * w
```

Here, `w` is dimensionless, so `Gscale` has units of siemens. The 2–32 microSiemens interval is the model conductance-mapping window. The remaining update settings are retained from Supplementary Software 2 and are configured in the `memristor_parameters` dictionary. In `main.py`, `Np = Nd = 8`, the nonlinear update coefficients are zero and the variability coefficients are zero. The separate potentiation/depression example in `memristor_optimizer.py` has its own update settings.

### Outputs

Results are stored under `./train_results_resnet/`:

- `resnet18_Sigmoid_bs.pth`: network state, epoch-wise metrics and repeated-inference results when available.
- `resnet18_Sigmoid_bs_accuracy.svg`: training and test accuracy.
- `resnet18_Sigmoid_bs_loss.svg`: training loss.
- `resnet18_Sigmoid_bs_weight_updates.svg`: recorded weight-update counts.
- `resnet18_Sigmoid_bs_ltp_ltd_w.svg` and `resnet18_Sigmoid_bs_ltp_ltd_G.svg`: weight and conductance trajectories for potentiation/depression.
- `resnet18_Sigmoid_bs_accuracy_bs_repeats.svg`: cumulative-voting accuracy against repetition count.

The script loads an existing checkpoint with the same filename. Use a fresh copy of the repository or set a new `filename` in `main.py` to start a separate run and preserve previous results. Training duration depends on the processor, device backend and selected epoch count.

## 6. Demonstration validation

The two short demonstrations were checked on Windows using Python 3.13.3, NumPy 2.2.6 and Matplotlib 3.10.3. Validation includes execution, generated plots and the stochastic-model parameter calculations. The network workflow and its settings are described separately in Section 5.

In this validation session, the activation demonstration took approximately 0.3 s and the compact noise model approximately 2.4 s, measured after importing their dependencies and including PNG generation. These timings are illustrative and depend on the computer and plotting backend.

## License

This project is distributed under the [MIT License](LICENSE).

## Citation

If this software contributes to your research, please cite the accompanying manuscript:

*Fusion of neural nonlinearity with synaptic weighting in memristive crossbar array via controlled signal-to-noise ratio.*
