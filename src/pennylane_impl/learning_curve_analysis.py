# -*- coding: utf-8 -*-
"""
Learning curve analysis - test model performance with varying training data sizes.
"""
import os
import json
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from itertools import combinations, product

# Parameter settings
n_qubits = 10
n_layers = 1
J_value = -0.1
h_value = -0.5
Lambda = 2

# Test different training data sizes (total 3000 data points)
train_sizes = [50, 100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1500, 1800, 2000, 2250]
dataset_file_path = 'toy_projects/4_b/'

def load_dataset(file_path, J_, h_, idea_):
    """Load dataset from JSON file."""
    if idea_:
        file_name = file_path + f"data_ideal_4b.json"
        with open(file_name, 'r') as file:
            data = json.load(file)
    else:
        file_name = file_path + f"data_shadow_4b.json"
        with open(file_name, 'r') as file:
            data = json.load(file)
    return data

def generate_frequency_vectors(input_dim, Lambda):
    """Generate frequency vectors with truncated Hamming weight up to Lambda."""
    frequency_vectors = []
    frequency_vectors.append(np.zeros(input_dim))
    for hamming_weight in range(1, Lambda + 1):
        for indices in combinations(range(input_dim), hamming_weight):
            for signs in product([-1, 1], repeat=hamming_weight):
                freq_vector = np.zeros(input_dim)
                for idx, sign in zip(indices, signs):
                    freq_vector[idx] = sign
                frequency_vectors.append(freq_vector)
    return np.array(frequency_vectors)

def feature_map_deterministic(data, Lambda):
    """Deterministic feature mapping function."""
    n_samples, input_dim = data.shape
    sampled_W = generate_frequency_vectors(input_dim, Lambda)
    data_expanded = np.expand_dims(data, axis=1)
    W_expanded = np.expand_dims(sampled_W, axis=0)
    
    f_x_w = np.where(
        W_expanded == 0,
        1,
        np.where(W_expanded == 1, np.cos(data_expanded), np.sin(data_expanded))
    )
    
    features = np.prod(f_x_w, axis=2)
    hamming_weights = np.count_nonzero(sampled_W, axis=1)
    weight_factors = 2 ** hamming_weights
    features *= weight_factors
    
    return features

def evaluate_model(train_size, test_size, J_, h_, idea_setting_):
    """Evaluate model performance with specified training data size."""
    # Load data data
    if idea_setting_:
        data_all = load_dataset(dataset_file_path, J_, h_, True)
        train_data = np.array(data_all['para'])[:train_size]
        train_data = np.reshape(train_data, newshape=(train_size, -1))
        train_labels = np.array(data_all['labels'])[:train_size]
        
        test_data = np.array(data_all['para'])[-test_size:]
        test_data = np.reshape(test_data, newshape=(test_size, -1))
        test_labels = np.array(data_all['labels'])[-test_size:]
    else:
        data_all = load_dataset(dataset_file_path, J_, h_, False)
        train_data = np.array(data_all['parameters'])[:train_size]
        train_data = np.reshape(train_data, newshape=(train_size, -1))
        train_labels = np.array(data_all['shadow_labels'])[:train_size]
        train_labels = np.squeeze(train_labels)
        
        test_data = np.array(data_all['parameters'])[-test_size:]
        test_data = np.reshape(test_data, newshape=(test_size, -1))
        test_labels = np.array(data_all['shadow_labels'])[-test_size:]
        test_labels = np.squeeze(test_labels)
    
    # Feature mapping
    feature_map_train = feature_map_deterministic(train_data, Lambda)
    feature_map_test = feature_map_deterministic(test_data, Lambda)
    
    # Train model
    model = Ridge(alpha=1)
    model.fit(feature_map_train, train_labels)
    
    # Prediction
    predictions = model.predict(feature_map_test)
    
    # Compute metrics
    mse = mean_squared_error(test_labels, predictions)
    r2 = r2_score(test_labels, predictions)
    pearson_corr, _ = pearsonr(test_labels, predictions)
    
    return {
        'train_size': train_size,
        'mse': mse,
        'r2': r2,
        'pearson': pearson_corr
    }

def run_learning_curve_analysis(idea_setting=False):
    """Run learning curve analysis."""
    results = []
    test_size = 750  # Use 750 samples as test set from 3000 total
    
    setting_name = "Ideal" if idea_setting else "Shadow"
    print(f"\n{'='*60}")
    print(f"Starting learning curve analysis for {setting_name} data")
    print(f"{'='*60}\n")
    
    for train_size in train_sizes:
        print(f"Training data size: {train_size}")
        result = evaluate_model(train_size, test_size, J_value, h_value, idea_setting)
        results.append(result)
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  R²: {result['r2']:.6f}")
        print(f"  Pearson: {result['pearson']:.6f}")
        print()
    
    # Save results
    output_file = f'learning_curve_results_{setting_name}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")
    
    return results

def plot_learning_curves(results_ideal, results_shadow):
    """Plot learning curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Extract data
    train_sizes_list = [r['train_size'] for r in results_ideal]
    
    # MSE curve
    mse_ideal = [r['mse'] for r in results_ideal]
    mse_shadow = [r['mse'] for r in results_shadow]
    axes[0].plot(train_sizes_list, mse_ideal, 'o-', label='Ideal', linewidth=2, markersize=6)
    axes[0].plot(train_sizes_list, mse_shadow, 's-', label='Shadow', linewidth=2, markersize=6)
    axes[0].set_xlabel('Training Data Size', fontsize=12)
    axes[0].set_ylabel('MSE', fontsize=12)
    axes[0].set_title('Mean Squared Error vs Training Data Size', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # R² curve
    r2_ideal = [r['r2'] for r in results_ideal]
    r2_shadow = [r['r2'] for r in results_shadow]
    axes[1].plot(train_sizes_list, r2_ideal, 'o-', label='Ideal', linewidth=2, markersize=6)
    axes[1].plot(train_sizes_list, r2_shadow, 's-', label='Shadow', linewidth=2, markersize=6)
    axes[1].set_xlabel('Training Data Size', fontsize=12)
    axes[1].set_ylabel('R²', fontsize=12)
    axes[1].set_title('R² Score vs Training Data Size', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xscale('log')
    axes[1].axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='0.99 threshold')
    
    # Pearson correlation curve
    pearson_ideal = [r['pearson'] for r in results_ideal]
    pearson_shadow = [r['pearson'] for r in results_shadow]
    axes[2].plot(train_sizes_list, pearson_ideal, 'o-', label='Ideal', linewidth=2, markersize=6)
    axes[2].plot(train_sizes_list, pearson_shadow, 's-', label='Shadow', linewidth=2, markersize=6)
    axes[2].set_xlabel('Training Data Size', fontsize=12)
    axes[2].set_ylabel('Pearson Correlation', fontsize=12)
    axes[2].set_title('Pearson Correlation vs Training Data Size', fontsize=14)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xscale('log')
    axes[2].axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='0.99 threshold')
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    print("\nLearning curves saved as learning_curves.png")
    plt.show()

def analyze_convergence(results, threshold=0.99):
    """Analyze convergence point."""
    print(f"\n{'='*60}")
    print("Convergence Analysis (R² ≥ {:.2f} as convergence criterion)".format(threshold))
    print(f"{'='*60}\n")
    
    convergence_point = None
    for result in results:
        if result['r2'] >= threshold:
            convergence_point = result['train_size']
            print(f"✓ Convergence point: {convergence_point} training samples")
            print(f"  R² = {result['r2']:.6f}")
            print(f"  MSE = {result['mse']:.6f}")
            print(f"  Pearson = {result['pearson']:.6f}")
            break
    
    if convergence_point is None:
        print("✗ Convergence criterion not met within tested data range")
        print(f"  Max R² = {results[-1]['r2']:.6f} (using {results[-1]['train_size']} samples)")
    
    return convergence_point

if __name__ == "__main__":
    # Run analysis for Ideal data
    results_ideal = run_learning_curve_analysis(idea_setting=True)
    
    # Run analysis for Shadow data
    results_shadow = run_learning_curve_analysis(idea_setting=False)
    
    # Analyze convergence
    print("\n" + "="*60)
    print("Ideal Data Convergence Analysis")
    analyze_convergence(results_ideal)
    
    print("\n" + "="*60)
    print("Shadow Data Convergence Analysis")
    analyze_convergence(results_shadow)
    
    # Plot learning curves
    plot_learning_curves(results_ideal, results_shadow)
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\nRecommended minimum training data sizes:")
    
    # Analyze with different thresholds
    for threshold in [0.95, 0.98, 0.99]:
        print(f"\nWhen R² ≥ {threshold}:")
        for setting_name, results in [("Ideal", results_ideal), ("Shadow", results_shadow)]:
            for result in results:
                if result['r2'] >= threshold:
                    print(f"  {setting_name}: {result['train_size']} samples")
                    break
            else:
                print(f"  {setting_name}: Criterion not met")
