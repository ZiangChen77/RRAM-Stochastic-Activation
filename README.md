# Fusion of Neural Nonlinearity with Synaptic Weighting in Memristive Crossbar Array via Controlled Signal-to-Noise Ratio

This repository contains the official Python implementation and hardware-aware simulation framework for the manuscript: "Fusion of neural nonlinearity with synaptic weighting in memristive crossbar array via controlled signal-to-noise ratio" (Manuscript No. NCOMMS-26-041311).

This framework models the stochastic computing paradigm for 1T1R RRAM arrays, where intrinsic read noise is repurposed as a functional computational resource to physically fuse array-level multiply-accumulate (MAC) operations and Sigmoid activation. It also implements the mapping of Batch Normalization (BN) parameters directly onto tunable circuit knobs (comparator bias and read voltage).

---

## 1. System Requirements

### Operating Systems
* Tested on: Windows 10/11, Ubuntu 20.04 LTS, and macOS.
* The code is platform-independent and runs on any standard operating system supporting Python 3.8+.

### Software Dependencies
* Python version: Python 3.8, 3.9, or 3.10.
* Required Libraries:
  * torch >= 1.12.0
  * torchvision >= 0.13.0
  * numpy >= 1.21.0
  * scipy >= 1.7.0
  * matplotlib >= 3.5.0
  * tqdm >= 4.64.0

### Hardware Requirements
* Standard Desktop: No non-standard hardware is required.
* CPU: Standard Intel/AMD processor (suitable for running the Demo and evaluations).
* GPU (Optional but Recommended): NVIDIA GPU with CUDA support is highly recommended for running the full PreActResNet-18 training on CIFAR-10.

---

## 2. Installation Guide

### Step 1: Clone the Repository
git clone https://github.com/ZiangChen77/RRAM-Stochastic-Activation

cd RRAM-Stochastic-Activation

### Step 2: Create a Virtual Environment (Recommended)
For Anaconda users:
conda create -n rram_stochastic python=3.9

conda activate rram_stochastic

Or using venv:
python -m venv venv

source venv/bin/activate  # On Windows use: venv\Scripts\activate

### Step 3: Install Dependencies
pip install -r requirements.txt

* Typical Installation Time: Less than 2 minutes on a standard desktop computer.

---

## 3. Demo (Quick Run)

We provide a self-contained demo script (run_demo.py) that simulates the physical fusion of RRAM array-level MAC operations and Sigmoid activation using simulated 1T1R sensing fluctuations.

### Run the Demo:
python run_demo.py

### Expected Output:
1. Console Output:
   [INFO] Initializing RRAM stochastic activation demo...
   
   [INFO] Running stochastic hardware simulation with 5000 samples per point...
   
   [INFO] Successfully mapped Batch Normalization parameters:
   
          - Comparator Bias (V_bias): 0.24 V
          - Read Voltage (V_r): 0.40 V
   
   [SUCCESS] Demo completed. Plot saved as 'stochastic_activation_demo.png'.

3. Generated File: A plot named 'stochastic_activation_demo.png' will be saved in the root directory, showing the simulated cumulative distribution function (CDF) of RRAM noise matching the theoretical Sigmoid curve.
* Expected Run Time: Less than 5 seconds on a standard CPU.

---

## 4. Instructions for Use

This framework allows users to run both the physical memristor noise model and the hardware-aware neural network training.

### Model 1: Compact Stochastic Behavioral Model for RRAM
To run the physical dynamic noise simulation of the memristor (Supplementary Software 1):
python compact_rram_stochastic_model.py

* This script runs the stochastic simulation loop of the memristor read current and generates a dynamic noise trace plot saved as 'Dynamic_Noise_Model.png'.

### Model 2: Hardware-Aware Neural Network Simulation (Supplementary Software 2)
The CIFAR-10 dataset will be automatically downloaded to the './data' directory upon running the script.

To start the hardware-aware training of PreActResNet-18 on CIFAR-10 from scratch (using the RRAM noise-induced Sigmoid activation and BN-to-circuit mapping):
python main.py

* Training Details: The network will train for 200 epochs. All training checkpoints, accuracy/loss plots, and weight update statistics will be automatically saved in the './train_results_resnet/' directory.

---

## 5. Reproduction of Quantitative Results

To reproduce the key results reported in the manuscript:
* Table 1 & Figure 5 (Inference Accuracy on CIFAR-10): Run the training script (main.py) to completion. Once the training is completed, the script automatically executes multiple repeated inference evaluations (50 repeats) under stochastic hardware fluctuations to generate the final statistical accuracy results, which will be output to the console and plotted.

---

## License
This project is licensed under the MIT License.

## Citation
If you find this work or code useful for your research, please cite our paper:
"Fusion of neural nonlinearity with synaptic weighting in memristive crossbar array via controlled signal-to-noise ratio"
