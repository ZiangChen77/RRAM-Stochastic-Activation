"""Illustrate stochastic comparator sampling and a logistic approximation."""

import numpy as np
import matplotlib.pyplot as plt


# Adjustable demonstration parameters. These illustrative voltages are not
# extracted device measurements or a calibrated read-voltage-to-BN mapping.
DEMO_PARAMETERS = {
    'seed': 1234,
    'v_bias': 0.24,       # Comparator threshold (V)
    'noise_std': 0.425,   # Gaussian input-noise standard deviation (V)
    'num_samples': 5000, # Binary decisions per fixed input value
    'input_min': -1.5,   # Comparator input range (V)
    'input_max': 1.5,
    'num_inputs': 100,
    'sigmoid_gain': 4.0, # Logistic gain (1/V)
}

# Plot settings are exposed here for easy customization.
PLOT_PARAMETERS = {
    'figure_size': (8, 5.5),
    'font_family': 'Arial',
    'title_fontsize': 12,
    'label_fontsize': 11,
    'tick_fontsize': 10,
    'legend_fontsize': 9,
    'logistic_color': '#d62728',
    'sample_color': '#1f77b4',
    'bias_color': '#7f7f7f',
    'line_width': 2.0,
    'marker_size': 15,
    'legend_location': 'lower right',
    'output_file': 'stochastic_activation_demo.png',
    'dpi': 300,
}


def simulate_stochastic_activation():
    """Estimate binary-output probabilities at fixed comparator inputs."""
    params = DEMO_PARAMETERS
    style = PLOT_PARAMETERS
    rng = np.random.default_rng(params['seed'])
    v_bias = params['v_bias']
    noise_std = params['noise_std']
    num_samples = params['num_samples']
    v_input = np.linspace(
        params['input_min'], params['input_max'], params['num_inputs']
    )
    activation_probability = []

    print('[INFO] Initializing stochastic comparator demo...')
    print('[INFO] Illustrative demonstration parameters:')
    print(f"       - Random seed: {params['seed']}")
    print(f'       - Comparator threshold (V_bias): {v_bias} V')
    print(f'       - Gaussian noise standard deviation: {noise_std} V')
    print(f'       - Binary samples per fixed input: {num_samples}')

    for v in v_input:
        noisy_signals = v + rng.normal(0, noise_std, num_samples)
        comparator_outputs = (noisy_signals > v_bias).astype(float)
        # The sample mean estimates a probability for this fixed input.
        activation_probability.append(np.mean(comparator_outputs))

    # Gaussian comparator noise gives a normal-CDF response. A logistic curve
    # with gain about 1.702 / noise_std approximates, rather than equals, it.
    logistic_approximation = 1.0 / (
        1.0 + np.exp(-params['sigmoid_gain'] * (v_input - v_bias))
    )

    with plt.rc_context({
        'font.family': style['font_family'],
        'mathtext.fontset': 'stix',
        'xtick.labelsize': style['tick_fontsize'],
        'ytick.labelsize': style['tick_fontsize'],
    }):
        fig, ax = plt.subplots(figsize=style['figure_size'])
        ax.plot(
            v_input, logistic_approximation,
            color=style['logistic_color'], linestyle='--',
            linewidth=style['line_width'], label='Logistic approximation',
        )
        ax.scatter(
            v_input, activation_probability,
            color=style['sample_color'], s=style['marker_size'], alpha=0.8,
            label='Stochastic comparator sampling',
        )
        ax.axvline(
            x=v_bias, color=style['bias_color'], linestyle=':',
            label=rf'Comparator threshold $V_{{\mathrm{{bias}}}}$ = {v_bias} V',
        )
        ax.set_title(
            'Stochastic comparator activation', fontsize=style['title_fontsize']
        )
        ax.set_xlabel('Comparator input (V)', fontsize=style['label_fontsize'])
        ax.set_ylabel('Output probability', fontsize=style['label_fontsize'])
        ax.tick_params(direction='out')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.25)
        ax.legend(
            loc=style['legend_location'], fontsize=style['legend_fontsize'],
            frameon=False,
        )
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(style['output_file'], dpi=style['dpi'])
        plt.close(fig)

    print(f"[SUCCESS] Demo completed. Plot saved as '{style['output_file']}'.")


if __name__ == '__main__':
    simulate_stochastic_activation()
