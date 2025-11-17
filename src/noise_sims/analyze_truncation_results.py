# -*- coding: utf-8 -*-
"""
Analysis of the impact of different truncation parameters (Lambda) on model performance
Including energy evaluation for optimized parameters
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import os
import sys

# Add parent directory to path for importing simulators
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# TFIM Hamiltonian parameters
n_qubits = 4
J_coupling = -0.1
h_field = -0.5

def load_results(filename):
    """Load results from JSON file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return {}

def extract_lambda_results(base_dir, lambda_values):
    """
    Extract results for different Lambda values from directory structure
    
    Args:
        base_dir: Base directory containing lambda_X subdirectories
        lambda_values: List of Lambda values to extract
    
    Returns:
        Dictionary with Lambda as key and results as values
    """
    lambda_results = {}
    
    for lambda_val in lambda_values:
        lambda_dir = os.path.join(base_dir, f'lambda_{lambda_val}')
        results_file = os.path.join(lambda_dir, 'ML_fitting_results.json')
        
        if os.path.exists(results_file):
            results = load_results(results_file)
            lambda_results[lambda_val] = []
            
            for key, value in results.items():
                lambda_results[lambda_val].append({
                    'mse': value['mse'],
                    'r2': value['metrics']['R^2'],
                    'pearson': value['metrics']['Pearson Correlation']
                })
        else:
            print(f"Results file not found: {results_file}")
    
    return lambda_results

def calculate_num_features(input_dim, lambda_val):
    """Calculate number of features for given Lambda truncation"""
    n_features = 1  # Include zero Hamming weight
    for hamming_weight in range(1, int(lambda_val) + 1):
        n_features += len(list(combinations(range(input_dim), hamming_weight))) * (2**hamming_weight)
    return n_features

def analyze_training_results(lambda_values=[1, 2, 3, 4, 5, 6, 7]):
    """Analyze ML model training results for different Lambda values"""
    
    input_dim = 7  # n_layers * (2*n_qubits - 1) = 1 * 7
    base_dir = 'results/data/noise_sims_data'
    
    datasets = [
        {'name': 'shadow', 'dir': os.path.join(base_dir, 'shadow_results')},
        {'name': 'noise_shadow', 'dir': os.path.join(base_dir, 'noise_shadow_results')},
        {'name': 'tls_noise_shadow', 'dir': os.path.join(base_dir, 'tls_noise_shadow_results')}
    ]
    
    print("\n" + "="*100)
    print("ANALYSIS OF TRUNCATION PARAMETER (Lambda) IMPACT ON MODEL PERFORMANCE")
    print("="*100 + "\n")
    
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*100}")
        print(f"Dataset: {dataset['name']}")
        print(f"{'='*100}\n")
        
        lambda_results = extract_lambda_results(dataset['dir'], lambda_values)
        all_results[dataset['name']] = lambda_results
        
        # Create results table
        print(f"{'Lambda':<10} {'MSE':<18} {'R²':<18} {'Pearson Corr.':<18} {'# Features':<15}")
        print("-" * 100)
        
        for lambda_val in sorted(lambda_results.keys()):
            results = lambda_results[lambda_val]
            if results:
                avg_mse = np.mean([r['mse'] for r in results])
                avg_r2 = np.mean([r['r2'] for r in results])
                avg_pearson = np.mean([r['pearson'] for r in results])
                n_features = calculate_num_features(input_dim, lambda_val)
                
                print(f"{lambda_val:<10} {avg_mse:<18.6f} {avg_r2:<18.6f} {avg_pearson:<18.6f} {n_features:<15}")
        
        print()
    
    return all_results

def analyze_optimization_results(lambda_values=[1, 2, 3, 4, 5, 6, 7]):
    """Analyze optimization results including energy evaluation"""
    
    base_dir = 'results/data/noise_sims_data'
    
    datasets = [
        {'name': 'shadow', 'dir': os.path.join(base_dir, 'shadow_results')},
        {'name': 'noise_shadow', 'dir': os.path.join(base_dir, 'noise_shadow_results')},
        {'name': 'tls_noise_shadow', 'dir': os.path.join(base_dir, 'tls_noise_shadow_results')}
    ]
    
    print("\n" + "="*100)
    print("OPTIMIZATION RESULTS ANALYSIS (Adam Optimizer)")
    print("="*100 + "\n")
    
    optimization_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*100}")
        print(f"Dataset: {dataset['name']}")
        print(f"{'='*100}\n")
        
        dataset_results = {}
        
        print(f"{'Lambda':<10} {'Avg Final Loss':<20} {'Min Loss':<20} {'Max Loss':<20} {'Std Dev':<20}")
        print("-" * 100)
        
        for lambda_val in lambda_values:
            lambda_dir = os.path.join(dataset['dir'], f'lambda_{lambda_val}')
            adam_results_file = os.path.join(lambda_dir, 'Adam_Optimization_Results.json')
            
            if os.path.exists(adam_results_file):
                results = load_results(adam_results_file)
                
                if results:
                    losses = []
                    for key, value in results.items():
                        if value['loss_history']:
                            losses.append(value['loss_history'][-1])
                    
                    if losses:
                        avg_loss = np.mean(losses)
                        min_loss = np.min(losses)
                        max_loss = np.max(losses)
                        std_loss = np.std(losses)
                        
                        dataset_results[lambda_val] = {
                            'losses': losses,
                            'avg': avg_loss,
                            'min': min_loss,
                            'max': max_loss,
                            'std': std_loss
                        }
                        
                        print(f"{lambda_val:<10} {avg_loss:<20.6f} {min_loss:<20.6f} {max_loss:<20.6f} {std_loss:<20.6f}")
        
        optimization_results[dataset['name']] = dataset_results
        print()
    
    return optimization_results

def plot_comprehensive_analysis(training_results, optimization_results):
    """Create comprehensive plots comparing all metrics"""
    
    lambda_vals = sorted(list(training_results['shadow'].keys()))
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Analysis of Truncation Parameter Impact', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    datasets = ['shadow', 'noise_shadow', 'tls_noise_shadow']
    colors = {'shadow': 'blue', 'noise_shadow': 'red', 'tls_noise_shadow': 'green'}
    markers = {'shadow': 'o', 'noise_shadow': 's', 'tls_noise_shadow': '^'}
    
    # Plot 1: MSE vs Lambda
    ax = axes[0]
    for dataset in datasets:
        mse_vals = [np.mean([r['mse'] for r in training_results[dataset][l]]) 
                    for l in lambda_vals if l in training_results[dataset]]
        ax.plot(lambda_vals[:len(mse_vals)], mse_vals, 
               f'{markers[dataset]}-', label=dataset, color=colors[dataset], 
               linewidth=2, markersize=8)
    ax.set_xlabel('Lambda (Truncation Parameter)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
    ax.set_title('MSE vs Truncation Parameter', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(lambda_vals)
    
    # Plot 2: R² vs Lambda
    ax = axes[1]
    for dataset in datasets:
        r2_vals = [np.mean([r['r2'] for r in training_results[dataset][l]]) 
                  for l in lambda_vals if l in training_results[dataset]]
        ax.plot(lambda_vals[:len(r2_vals)], r2_vals, 
               f'{markers[dataset]}-', label=dataset, color=colors[dataset], 
               linewidth=2, markersize=8)
    ax.set_xlabel('Lambda (Truncation Parameter)', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax.set_title('R² Score vs Truncation Parameter', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(lambda_vals)
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, linewidth=2, label='0.99 Threshold')
    
    # Plot 3: Pearson Correlation vs Lambda
    ax = axes[2]
    for dataset in datasets:
        pearson_vals = [np.mean([r['pearson'] for r in training_results[dataset][l]]) 
                       for l in lambda_vals if l in training_results[dataset]]
        ax.plot(lambda_vals[:len(pearson_vals)], pearson_vals, 
               f'{markers[dataset]}-', label=dataset, color=colors[dataset], 
               linewidth=2, markersize=8)
    ax.set_xlabel('Lambda (Truncation Parameter)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pearson Correlation', fontsize=12, fontweight='bold')
    ax.set_title('Pearson Correlation vs Truncation Parameter', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(lambda_vals)
    ax.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, linewidth=2, label='0.99 Threshold')
    
    # Plot 4: Optimization Loss vs Lambda
    ax = axes[3]
    for dataset in datasets:
        if dataset in optimization_results:
            loss_vals = [optimization_results[dataset][l]['avg'] 
                        for l in lambda_vals if l in optimization_results[dataset]]
            loss_std = [optimization_results[dataset][l]['std'] 
                       for l in lambda_vals if l in optimization_results[dataset]]
            valid_lambdas = [l for l in lambda_vals if l in optimization_results[dataset]]
            
            ax.errorbar(valid_lambdas, loss_vals, yerr=loss_std,
                       fmt=f'{markers[dataset]}-', label=dataset, color=colors[dataset], 
                       linewidth=2, markersize=8, capsize=5)
    ax.set_xlabel('Lambda (Truncation Parameter)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final Optimization Loss', fontsize=12, fontweight='bold')
    ax.set_title('Optimization Loss vs Truncation Parameter', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(lambda_vals)
    
    # Plot 5: Number of Features vs Lambda
    ax = axes[4]
    input_dim = 7
    n_features = [calculate_num_features(input_dim, l) for l in lambda_vals]
    ax.plot(lambda_vals, n_features, 'o-', color='purple', linewidth=2, markersize=8)
    ax.set_xlabel('Lambda (Truncation Parameter)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Dimension vs Truncation Parameter', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(lambda_vals)
    ax.set_yscale('log')
    
    # Hide the 6th subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = 'results/data/noise_sims_data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'comprehensive_truncation_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive analysis plot saved to: {output_file}")
    
    return fig

def save_summary_report(training_results, optimization_results):
    """Save comprehensive summary report to text file"""
    
    output_dir = 'results/data/noise_sims_data'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'truncation_analysis_report.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("COMPREHENSIVE ANALYSIS REPORT: TRUNCATION PARAMETER IMPACT\n")
        f.write(f"TFIM Parameters: J={J_coupling}, h={h_field}, n_qubits={n_qubits}\n")
        f.write("="*100 + "\n\n")
        
        # Training results
        f.write("="*100 + "\n")
        f.write("SECTION 1: ML MODEL TRAINING PERFORMANCE\n")
        f.write("="*100 + "\n\n")
        
        for dataset in ['shadow', 'noise_shadow', 'tls_noise_shadow']:
            f.write(f"\nDataset: {dataset}\n")
            f.write("-"*100 + "\n")
            f.write(f"{'Lambda':<10} {'MSE':<18} {'R²':<18} {'Pearson':<18} {'# Features':<15}\n")
            f.write("-"*100 + "\n")
            
            for lambda_val in sorted(training_results[dataset].keys()):
                results = training_results[dataset][lambda_val]
                avg_mse = np.mean([r['mse'] for r in results])
                avg_r2 = np.mean([r['r2'] for r in results])
                avg_pearson = np.mean([r['pearson'] for r in results])
                n_features = calculate_num_features(7, lambda_val)
                
                f.write(f"{lambda_val:<10} {avg_mse:<18.6f} {avg_r2:<18.6f} {avg_pearson:<18.6f} {n_features:<15}\n")
            f.write("\n")
        
        # Optimization results
        f.write("\n" + "="*100 + "\n")
        f.write("SECTION 2: OPTIMIZATION PERFORMANCE\n")
        f.write("="*100 + "\n\n")
        
        for dataset in ['shadow', 'noise_shadow', 'tls_noise_shadow']:
            if dataset in optimization_results:
                f.write(f"\nDataset: {dataset}\n")
                f.write("-"*100 + "\n")
                f.write(f"{'Lambda':<10} {'Avg Loss':<20} {'Min Loss':<20} {'Max Loss':<20} {'Std Dev':<20}\n")
                f.write("-"*100 + "\n")
                
                for lambda_val in sorted(optimization_results[dataset].keys()):
                    res = optimization_results[dataset][lambda_val]
                    f.write(f"{lambda_val:<10} {res['avg']:<20.6f} {res['min']:<20.6f} {res['max']:<20.6f} {res['std']:<20.6f}\n")
                f.write("\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*100 + "\n")
    
    print(f"\nComprehensive report saved to: {output_file}")

if __name__ == "__main__":
    lambda_values = [1, 2, 3, 4, 5, 6, 7]
    
    print("\n" + "="*100)
    print("STARTING COMPREHENSIVE TRUNCATION ANALYSIS")
    print("="*100 + "\n")
    
    # Analyze training results
    training_results = analyze_training_results(lambda_values)
    
    # Analyze optimization results
    optimization_results = analyze_optimization_results(lambda_values)
    
    # Create comprehensive plots
    plot_comprehensive_analysis(training_results, optimization_results)
    
    # Save summary report
    save_summary_report(training_results, optimization_results)
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE!")
    print("="*100 + "\n")
