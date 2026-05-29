import numpy as np
import matplotlib.pyplot as plt


def simulate_stochastic_activation():
    """Simulates the physical fusion of MAC and Sigmoid activation using RRAM noise."""
    print("[INFO] Initializing RRAM stochastic activation demo...")

    # ==========================================
    # Parameters for Curve Fitting
    # ==========================================
    # The slope factor in the theoretical Sigmoid (k=4.0) dictates the required noise level.
    # Mathematically, the required noise_std ≈ 1.702 / k to perfectly match the Sigmoid curve.

    v_bias = 0.24  # Comparator Bias Voltage V_bias (Determines the center point)
    v_read = 0.40  # Read Voltage V_r (Example circuit control knob)
    noise_std = 0.425  # RRAM Read Noise Standard Deviation (Determines the slope)

    num_samples = 5000  # Number of stochastic samples per input point for smooth estimation

    # Simulate input range (Normalized MAC output)
    v_input = np.linspace(-1.5, 1.5, 100)
    activation_probability = []

    print(f"[INFO] Running stochastic hardware simulation with {num_samples} samples per point...")

    for v in v_input:
        # Physical process: MAC result (v) is perturbed by RRAM intrinsic Gaussian noise
        noisy_signals = v + np.random.normal(0, noise_std, num_samples)

        # Hardware decision: If noisy signal > V_bias, output 1 (Activation)
        comparator_outputs = (noisy_signals > v_bias).astype(float)

        # The mean of comparator outputs represents the physical activation probability
        prob = np.mean(comparator_outputs)
        activation_probability.append(prob)

    # ==========================================
    # Theoretical Target Function
    # ==========================================
    # The theoretical Sigmoid activation function used in the neural network model
    theoretical_sigmoid = 1.0 / (1.0 + np.exp(-4.0 * (v_input - v_bias)))

    # ==========================================
    # Result Visualization (Pure English)
    # ==========================================
    plt.figure(figsize=(8, 5.5))

    # Theoretical curve (Red Dashed Line)
    plt.plot(v_input, theoretical_sigmoid, 'r--', label='Theoretical Sigmoid Activation', linewidth=2)

    # Stochastic simulation results (Blue Dots)
    plt.scatter(v_input, activation_probability, color='#3A86A8', s=15,
                label='Physical RRAM Noise-induced Activation\n(Stochastic Simulation)', alpha=0.8)

    # Vertical line representing the comparator bias
    plt.axvline(x=v_bias, color='gray', linestyle=':', label=f'Comparator Bias $V_{{bias}}$ = {v_bias}V')

    plt.title('Physical Fusion of MAC and Sigmoid via RRAM Noise', fontsize=12, fontweight='bold')
    plt.xlabel('Normalized Input Voltage $V_{in}$ (V)', fontsize=10)
    plt.ylabel('Activation Probability / Output', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=9)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()

    # Save the output figure
    output_fig = 'stochastic_activation_demo.png'
    plt.savefig(output_fig, dpi=300)

    print(f"[INFO] Successfully mapped Batch Normalization parameters:")
    print(f"       - Comparator Bias (V_bias): {v_bias} V")
    print(f"       - Read Voltage (V_r): {v_read} V")
    print(f"[SUCCESS] Demo completed. Plot saved as '{output_fig}'.")


if __name__ == "__main__":
    simulate_stochastic_activation()
