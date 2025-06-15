# comp534-dp-implementation
## Differential Privacy Research Project in COMP534 term project

### Team Members:

* Alper Şahin 80417
* İdil Görgülü 79323
* Pelin Önen 76167

## Overview

This project implements and compares different anonymization techniques for protecting databases from inference attacks, with a focus on differential privacy. The research demonstrates the theoretical and practical advantages of differential privacy over naive noise-addition methods.

## Research Question

How does differential privacy compare to naive anonymization approaches in terms of:
- **Utility**: How much accuracy is preserved in query results?
- **Privacy**: How well does each method protect against inference attacks?

## Methodology

### Data Generation
- Generates synthetic datasets with values uniformly distributed in [5, 45]
- Dataset sizes: 1,000, 10,000, and 100,000 records
- Target mean: 25.0

### Anonymization Methods Compared

1. **Naive Uniform Noise**: Adds uniform random noise [-1, 1] to each data point
2. **Naive Gaussian Noise**: Adds Gaussian noise (μ=0, σ=1) to each data point  
3. **Laplace Differential Privacy**: Adds calibrated Laplace noise to query results

### Evaluation Metrics

- **Mean Squared Error (MSE)**: Measures utility loss
- **Jensen-Shannon Divergence**: Quantifies privacy through inference attack simulation

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup
1. Clone or download this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Experiment
```bash
python dp_experiment.py
```

### Output Files
The script generates several outputs in the `results/` directory:

#### Data Files
- `data_1000.csv`, `data_10000.csv`, `data_100000.csv`: Generated datasets
- `results_table.csv`: Quantitative results summary

#### Visualizations
- `figure_1_mse_vs_datasize.png`: Utility comparison across dataset sizes
- `figure_2_epsilon_vs_utility.png`: Privacy-utility trade-off analysis
- `figure_3_inference_resilience.png`: Inference attack resilience comparison

## Experiment Details

### Experiment 1: Utility Analysis (MSE vs Dataset Size)
- **Purpose**: Compare accuracy preservation across different dataset sizes
- **Metric**: Mean Squared Error between true and noisy means
- **Expected Result**: DP should show better scaling with larger datasets

### Experiment 2: Privacy Budget Analysis (ε vs Utility)
- **Purpose**: Demonstrate privacy-utility trade-off in differential privacy
- **Parameter Range**: ε ∈ [0.1, 2.0]
- **Expected Result**: Lower ε (more privacy) increases MSE (less utility)

### Experiment 3: Inference Attack Simulation
- **Purpose**: Evaluate resistance to differencing attacks
- **Method**: Compare query distributions on neighboring datasets (D vs D')
- **Metric**: Jensen-Shannon divergence (lower = better privacy)
- **Expected Result**: DP should have lowest distinguishability

## Key Configuration Parameters

```python
# Dataset parameters
DATA_SIZES = [1000, 10000, 100000]
TRUE_MEAN = 25.0
DATA_GENERATION_RANGE = [5.0, 45.0]

# Noise parameters
UNIFORM_NOISE_RANGE = [-1.0, 1.0]
GAUSSIAN_NOISE_STD_DEV = 1.0

# Differential privacy parameters
EPSILON = 1.0  # Privacy budget

# Experiment parameters
NUM_RUNS = 50  # Repetitions for stable statistics
```

## Theoretical Background

### Differential Privacy
A randomized algorithm M satisfies ε-differential privacy if for all neighboring datasets D and D' (differing by one record):

```
P(M(D) ∈ S) ≤ e^ε × P(M(D') ∈ S)
```

### Laplace Mechanism
For a query f with sensitivity Δf, the Laplace mechanism adds noise:
```
Noise ~ Laplace(0, Δf/ε)
```

For mean queries on bounded data [a,b]:
```
Sensitivity = (b-a)/n
```

## Expected Results

### Key Findings
1. **Utility**: DP achieves comparable or better utility than naive methods
2. **Privacy**: DP provides superior protection against inference attacks  
3. **Scalability**: DP's noise decreases with dataset size (1/n scaling)
4. **Formal Guarantees**: Only DP provides mathematical privacy guarantees

### Performance Metrics
- **MSE**: DP typically shows 10-50% better utility on large datasets
- **JS Divergence**: DP usually achieves 3-10% better privacy than naive methods

## Research Implications

### Advantages of Differential Privacy
1. **Mathematical Rigor**: Formal privacy guarantees vs ad-hoc approaches
2. **Intelligent Calibration**: Noise scales with query sensitivity and dataset size
3. **Composability**: Multiple queries can be analyzed with cumulative privacy cost
4. **Robustness**: Protection against arbitrary auxiliary information

### Limitations
1. **Parameter Selection**: Choosing appropriate ε requires domain expertise
2. **Worst-case Guarantees**: May be conservative for typical use cases
3. **Computational Overhead**: Slightly more complex than naive methods

## File Structure
```
differential-privacy-research/
├── differential_privacy_experiment.py  # Main experiment script
├── requirements.txt                    # Python dependencies
├── README.md                          # This documentation
└── results/                           # Generated outputs (created at runtime)
    ├── data_*.csv                     # Synthetic datasets
    ├── results_table.csv              # Quantitative results
    └── figure_*.png                   # Visualizations
```

## Extensions and Future Work

### Potential Enhancements
1. **Multiple ε Values**: Test various privacy budgets simultaneously
2. **Real Datasets**: Apply to actual sensitive data
3. **Advanced Attacks**: Implement more sophisticated inference attacks
4. **Other DP Mechanisms**: Compare Gaussian, exponential mechanisms
5. **Composition Analysis**: Study privacy costs of multiple queries