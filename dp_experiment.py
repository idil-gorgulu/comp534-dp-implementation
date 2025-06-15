import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import jensenshannon

# --- Configuration Parameters ---
DATA_SIZES = [1000, 10000, 100000]
TRUE_MEAN = 25.0
NUM_RUNS = 50
DATA_GENERATION_RANGE = [5.0, 45.0]
UNIFORM_NOISE_RANGE = [-1.0, 1.0]
GAUSSIAN_NOISE_STD_DEV = 1.0
EPSILON = 1.0

# --- Directory Setup ---
RESULTS_DIR = 'results'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)


# --- Core Functions (generate_data, naive_*, dp_laplace_mean, calculate_mse) ---
def generate_data(size, mean):
    low, high = DATA_GENERATION_RANGE
    data = np.random.uniform(low, high, size)
    return data


def naive_uniform_mean(data):
    noisy_data = data + np.random.uniform(UNIFORM_NOISE_RANGE[0], UNIFORM_NOISE_RANGE[1], size=len(data))
    return np.mean(noisy_data)


def naive_gaussian_mean(data):
    noisy_data = data + np.random.normal(0, GAUSSIAN_NOISE_STD_DEV, size=len(data))
    return np.mean(noisy_data)


def dp_laplace_mean(data, epsilon):
    true_mean = np.mean(data)
    # Sensitivity for mean query over bounded domain [5, 45]
    sensitivity = (DATA_GENERATION_RANGE[1] - DATA_GENERATION_RANGE[0]) / len(data)
    scale = sensitivity / epsilon
    noisy_mean = true_mean + np.random.laplace(0, scale)
    return noisy_mean


def calculate_mse(true_value, noisy_values):
    return np.mean((np.array(noisy_values) - true_value) ** 2)


# --- Main Experiment Logic (run_main_experiment, run_epsilon_experiment) ---
def run_main_experiment():
    results = []
    print("Running main experiment (MSE vs. Data Size)...")
    for size in DATA_SIZES:
        print(f"  Processing data size: {size}")
        data = generate_data(size, TRUE_MEAN)
        actual_mean = np.mean(data)
        print(f"    Generated data with actual sample mean: {actual_mean:.6f}")
        data_filename = os.path.join(RESULTS_DIR, f'data_{size}.csv')
        np.savetxt(data_filename, data, delimiter=',', fmt='%1.18f')
        uniform_results, gaussian_results, dp_results = [], [], []
        for _ in range(NUM_RUNS):
            uniform_results.append(naive_uniform_mean(data))
            gaussian_results.append(naive_gaussian_mean(data))
            dp_results.append(dp_laplace_mean(data, EPSILON))
        results.append({'Data Size': size, 'Algorithm': 'Naive Uniform', 'Avg. Noisy Mean': np.mean(uniform_results),
                        'MSE': calculate_mse(actual_mean, uniform_results)})
        results.append({'Data Size': size, 'Algorithm': 'Naive Gaussian', 'Avg. Noisy Mean': np.mean(gaussian_results),
                        'MSE': calculate_mse(actual_mean, gaussian_results)})
        results.append(
            {'Data Size': size, 'Algorithm': f'Laplace DP (ε={EPSILON})', 'Avg. Noisy Mean': np.mean(dp_results),
             'MSE': calculate_mse(actual_mean, dp_results)})
    return pd.DataFrame(results)


def run_epsilon_experiment():
    print("\nRunning Epsilon vs. Utility experiment...")
    epsilon_range = np.linspace(0.1, 2.0, 20)
    mse_values = []
    data = generate_data(10000, TRUE_MEAN)
    actual_mean = np.mean(data)
    for eps in epsilon_range:
        dp_results = [dp_laplace_mean(data, eps) for _ in range(NUM_RUNS)]
        mse_values.append(calculate_mse(actual_mean, dp_results))
    return epsilon_range, mse_values


# --- NEW INFERENCE ATTACK SECTION ---

def run_inference_analysis():
    """
    Simulates a differencing attack and calculates the JS Divergence
    to quantify the distinguishability of the output distributions.
    """
    print("\nRunning Inference Attack Analysis...")
    sim_size = 1000
    sim_runs = 10000

    # Database D1 and its neighbor D2
    d1 = generate_data(sim_size, TRUE_MEAN)
    d2 = d1[:-1]

    # Get query results
    results = {
        'Naive Uniform': {'d1': [], 'd2': []},
        'Naive Gaussian': {'d1': [], 'd2': []},
        'Laplace DP': {'d1': [], 'd2': []}
    }
    for _ in range(sim_runs):
        results['Naive Uniform']['d1'].append(naive_uniform_mean(d1))
        results['Naive Uniform']['d2'].append(naive_uniform_mean(d2))
        results['Naive Gaussian']['d1'].append(naive_gaussian_mean(d1))
        results['Naive Gaussian']['d2'].append(naive_gaussian_mean(d2))
        results['Laplace DP']['d1'].append(dp_laplace_mean(d1, EPSILON))
        results['Laplace DP']['d2'].append(dp_laplace_mean(d2, EPSILON))

    # Calculate JS Divergence for each method
    distinguishability_scores = {}
    for method in results:
        # Define common bins for a fair comparison
        all_values = results[method]['d1'] + results[method]['d2']
        bins = np.histogram_bin_edges(all_values, bins=100)

        # Create histograms (counts, not density)
        counts_p, _ = np.histogram(results[method]['d1'], bins=bins)
        counts_q, _ = np.histogram(results[method]['d2'], bins=bins)

        # Convert to probability distributions (sum = 1)
        p = counts_p / np.sum(counts_p)
        q = counts_q / np.sum(counts_q)

        # Add small epsilon to prevent numerical issues
        p = p + 1e-10
        q = q + 1e-10

        # Renormalize after adding epsilon
        p = p / np.sum(p)
        q = q / np.sum(q)

        # Calculate JS Divergence (squared JS distance)
        js_distance = jensenshannon(p, q)
        distinguishability_scores[method] = js_distance ** 2  # Square to get divergence

    return distinguishability_scores


def plot_distinguishability(scores):
    """
    Plots Figure 3 as a bar chart of distinguishability scores.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = list(scores.keys())
    values = list(scores.values())

    # Rename for clarity in the plot
    methods = [m.replace('Laplace DP', f'Laplace DP (ε={EPSILON})') for m in methods]

    bar = ax.bar(methods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Distinguishability (JS Divergence)', fontsize=12)
    ax.set_title('Figure 3: Inference Attack Resilience', fontsize=16)

    # ax.set_yscale('log')  # Commented out
    ax.grid(True, which="both", ls="--", axis='y')

    # Add text labels on top of bars
    ax.bar_label(bar, fmt='%.4f')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'figure_3_inference_resilience.png'))
    print("Saved Figure 3: Inference Attack Resilience (Bar Chart)")


# --- Plotting Functions (plot_mse_vs_datasize, plot_epsilon_vs_utility) ---
def plot_mse_vs_datasize(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=df, x='Data Size', y='MSE', hue='Algorithm', marker='o', ax=ax)
    ax.set_title('Figure 1: Utility Comparison (MSE vs. Data Size)', fontsize=16)
    ax.set_xlabel('Data Size (n)', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=12)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(title='Anonymization Method')
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'figure_1_mse_vs_datasize.png'))
    print("\nSaved Figure 1: MSE vs. Data Size")


def plot_epsilon_vs_utility(epsilon_range, mse_values):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epsilon_range, mse_values, marker='o', linestyle='-')
    ax.set_title('Figure 2: Privacy-Utility Trade-off in Laplace DP', fontsize=16)
    ax.set_xlabel('Privacy Budget (ε) - Lower is More Private', fontsize=12)
    ax.set_ylabel('Mean Squared Error (MSE) - Lower is More Useful', fontsize=12)
    ax.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'figure_2_epsilon_vs_utility.png'))
    print("Saved Figure 2: Epsilon vs. Utility")


# --- Execution ---
if __name__ == "__main__":
    results_df = run_main_experiment()
    print("\n--- Results Table ---")
    print(results_df.to_string())
    results_df.to_csv(os.path.join(RESULTS_DIR, 'results_table.csv'), index=False)

    plot_mse_vs_datasize(results_df)

    epsilon_range, mse_values = run_epsilon_experiment()
    plot_epsilon_vs_utility(epsilon_range, mse_values)

    dist_scores = run_inference_analysis()
    print("\n--- Distinguishability Scores (JS Divergence) ---")
    print(dist_scores)
    plot_distinguishability(dist_scores)

    print(f"\nAll experiments complete. All artifacts are in the '{RESULTS_DIR}/' directory.")