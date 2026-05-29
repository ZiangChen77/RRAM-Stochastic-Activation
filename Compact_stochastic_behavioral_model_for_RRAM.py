import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# Kinetic Monte Carlo Model Parameters
# ==========================================
# Activation energy parameters for transition kinetics
W_ON = 0.1e-19  # Energy barrier for ON state transition (J)
W_OFF = 0.1e-19  # Energy barrier for OFF state transition (J)
TAU_0 = 1e-6  # Characteristic time constant (s)
K_B = 1.38e-23  # Boltzmann constant (J/K)
TEMP = 300  # Operating temperature (K)

# Device and Sensing Parameters
R_DEV = 45000  # Nominal device resistance (Ω)
V_READ = 0.0765  # Read voltage (V)
TOTAL_TIME = 0.2  # Total simulation time (s)
TSAMPLE = 1e-5  # Temporal resolution (s)

# Statistical Offset and Noise Parameters
OFFSET_BASE_MEAN = 0.001  # Mean shift factor for conductance states
OFFSET_MEAN_R = 0.007  # Relative resistance/conductance variation
NOISE_AMPLITUDE = 0.005  # Broadband noise amplitude factor


# ==========================================
# Transition Dynamics Calculation
# ==========================================
def calculate_dwell_time(energy, tau_ref):
    """Calculates state dwell time using Arrhenius-type transition kinetics."""
    return tau_ref * np.exp(energy / (K_B * TEMP))


tau_ON = calculate_dwell_time(W_ON, TAU_0)
tau_OFF = calculate_dwell_time(W_OFF, TAU_0)

# Simulation setup
num_samples = int(TOTAL_TIME / TSAMPLE)
current_trace = np.zeros(num_samples)

# Initial baseline currents based on Ohm's Law
high_base_mean = (V_READ / R_DEV) * (1 + OFFSET_MEAN_R)
low_base_mean = (V_READ / R_DEV) * (1 - OFFSET_MEAN_R)

# Pre-calculate fluctuation scales
low_mean_fluctuation = low_base_mean * OFFSET_BASE_MEAN
high_mean_fluctuation = high_base_mean * OFFSET_BASE_MEAN

# Initialize state-dependent statistical means
low_curr_mean = np.random.normal(low_base_mean, low_mean_fluctuation)
high_curr_mean = np.random.normal(high_base_mean, high_mean_fluctuation)

# ==========================================
# Stochastic Simulation Loop
# ==========================================
current_state = 'high'

for i in range(num_samples):
    rand_num = np.random.uniform(0, 1)

    if current_state == 'high':
        # Transition condition: High -> Low (driven by tau_ON)
        if rand_num > (TSAMPLE / tau_ON):
            current_state = 'low'
            low_curr_mean = np.random.normal(low_base_mean, low_mean_fluctuation)
            low_curr_std = high_curr_mean * NOISE_AMPLITUDE
            current_trace[i] = np.random.normal(low_curr_mean, low_curr_std)
        else:
            high_curr_std = low_curr_mean * NOISE_AMPLITUDE
            current_trace[i] = np.random.normal(high_curr_mean, high_curr_std)

    elif current_state == 'low':
        # Transition condition: Low -> High (driven by tau_OFF)
        if rand_num > (TSAMPLE / tau_OFF):
            current_state = 'high'
            high_curr_mean = np.random.normal(high_base_mean, high_mean_fluctuation)
            high_curr_std = low_curr_mean * NOISE_AMPLITUDE
            current_trace[i] = np.random.normal(high_curr_mean, high_curr_std)
        else:
            low_curr_std = high_curr_mean * NOISE_AMPLITUDE
            current_trace[i] = np.random.normal(low_curr_mean, low_curr_std)

# ==========================================
# Result Visualization
# ==========================================
time_vector = np.arange(0, num_samples * TSAMPLE, TSAMPLE)

plt.figure(figsize=(10, 6))
plt.plot(time_vector, current_trace, label='Read current (A)', color='#3A86A8', linewidth=0.8)
plt.title('Dynamic Noise Model of Memristor')
plt.xlabel('Time (s)')
plt.ylabel('Current (A)')
plt.yscale('log')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('Dynamic_Noise_Model.png', dpi=300)
plt.show()
