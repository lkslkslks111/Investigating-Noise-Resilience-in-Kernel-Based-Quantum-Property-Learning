# -*- coding: utf-8 -*-
"""
# **This file contains all suggorates used for pre-training VQE with large-scale TFIM models**
"""
import os
import json
from numpy import linalg as LA
from scipy.sparse.linalg import eigsh
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from itertools import combinations, product
import optax
import jax
from sklearn.neural_network import MLPRegressor
from joblib import dump, load

 
 
n_qubits = 4
n_layers = 1  # Number of layers in the ansatz
T = 100  # Number of snapshots for shadow dataset
n_samples = 3000  # Number of data samples to generate
J_list = [-0.1]  # Coupling strength
h_list = [-0.5] # Field strength

"""# Surrogates Pretraining [data loading]"""

# Dataset Loading (shadow only)
def load_dataset(file_path):
    """加载shadow数据集"""
    file_name = file_path + "noise_data_shadow.json"
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

# Build Train and test dataset
num_train = 2250
num_test = 750

dataset_file_path = 'results/data/noise_sims_data/'

# Metricsa

def evaluate_predictions(predictions, test_results):
    """
    Evaluate the relation between predictions and test results using R^2
    and Pearson correlation coefficient.

    Args:
    - predictions (numpy.ndarray): Predicted values.
    - test_results (numpy.ndarray): True test results.

    Returns:
    - dict: A dictionary containing R^2 and Pearson correlation coefficient.
    """
    # Compute R^2 score
    r2 = r2_score(test_results, predictions)

    # Compute Pearson correlation coefficient
    pearson_corr, _ = pearsonr(test_results, predictions)

    return {"R^2": r2, "Pearson Correlation": pearson_corr}

"""# Proposed ML Surrogates"""

def generate_frequency_vectors(input_dim, Lambda):
    """
    Generate frequency vectors with truncated Hamming weight up to Lambda.

    Args:
        input_dim (int): Dimensionality of the input data (number of features).
        Lambda (int): Maximum Hamming weight for the truncated frequency set.

    Returns:
        numpy.ndarray: Frequency matrix of shape (n_vectors, input_dim),
                       where n_vectors is determined by input_dim and Lambda.
    """
    frequency_vectors = []
    frequency_vectors.append(np.zeros(input_dim)) # frequence vector with zero Hamming weight
    for hamming_weight in range(1, Lambda + 1):
        for indices in combinations(range(input_dim), hamming_weight):
            for signs in product([-1, 1], repeat=hamming_weight):
                freq_vector = np.zeros(input_dim)
                for idx, sign in zip(indices, signs):
                    freq_vector[idx] = sign
                frequency_vectors.append(freq_vector)

    return np.array(frequency_vectors)

def feature_map_deterministic(data, Lambda):
    """
    Compute a deterministic feature map based on truncated Hamming weight.

    Args:
        data (numpy.ndarray): Input matrix of shape (n_samples, input_dim).
        Lambda (int): Maximum Hamming weight for the truncated frequency set.

    Returns:
        numpy.ndarray: Feature map matrix of shape (n_samples, n_vectors),
                       where n_vectors is determined by input_dim and Lambda.
    """
    n_samples, input_dim = data.shape

    # Generate deterministic frequency vectors
    sampled_W = generate_frequency_vectors(input_dim, Lambda)  # Shape: (n_vectors, input_dim)

    # Expand data and sampled_W to align dimensions
    data_expanded = np.expand_dims(data, axis=1)   
    W_expanded = np.expand_dims(sampled_W, axis=0)   

    f_x_w = np.where(
        W_expanded == 0,  # If w_j == 0
        1,
        np.where(W_expanded == 1, np.cos(data_expanded), np.sin(data_expanded))
    )   

    # Compute the product across dimensions to get g(x; w)
    features = np.prod(f_x_w, axis=2)   

    # Weight features by 2^||w||_0
    hamming_weights = np.count_nonzero(sampled_W, axis=1)   
    weight_factors = 2 ** hamming_weights   
    features *= weight_factors   

    return features

##################################
## Implementation of Surrogate ###
##################################


# Generate deterministic feature map by using the shadow_dataset
def ML_predictor(J_, h_, num_train_, num_test_, Lambda_, results_file_path, model_file_path_, data_filename):

    # 加载指定的数据文件
    # Check if file is in pennylane_data/4b directory (noiseless data)
    if data_filename == "data_shadow_4b.json":
        file_name = 'results/data/pennylane_data/4b/' + data_filename
    else:
        file_name = dataset_file_path + data_filename
    
    with open(file_name, 'r') as file:
        data_all = json.load(file)
    
    train_data =  np.array(data_all['parameters'])[:num_train_]
    train_data = np.reshape(train_data, newshape = (num_train_, -1))
    train_labels = np.array(data_all['shadow_labels'])[:num_train_]
    train_labels  = np.squeeze(train_labels)

    test_data = np.array(data_all['parameters'])[-num_test_:]
    test_data = np.reshape(test_data, newshape = (num_test_, -1))
    test_labels = np.array(data_all['shadow_labels'])[-num_test_:]
    test_labels  = np.squeeze(test_labels)

    # Feature map implementation
    feature_map_trunc = feature_map_deterministic(train_data, Lambda_)
    print("Feature Map Shape:", feature_map_trunc.shape)
    # Model training
    model_Trunc = Ridge(alpha=1)
    model_Trunc.fit(feature_map_trunc, train_labels)

    # Predict on new transformed data
    phi_new = feature_map_deterministic(test_data, Lambda_)
    predictions = model_Trunc.predict(phi_new)

    # Evaluate
    metrics = evaluate_predictions(predictions, test_labels)
    mse = mean_squared_error(test_labels, predictions)
    print(f"Mean Squared Error: {mse}")
    print("Evaluation Metrics:")
    print(f"R^2: {metrics['R^2']:.4f}")
    print(f"Pearson Correlation: {metrics['Pearson Correlation']:.4f}")

    # Prepare results for JSON
    result = {
        "J": J_,
        "h": h_,
        "Lambda": Lambda_,  # 添加Lambda信息
        "mse": mse,
        "metrics": {
            "R^2": metrics["R^2"],
            "Pearson Correlation": metrics["Pearson Correlation"]
        }
    }

    # Save results to JSON
    try:
        with open(results_file_path, "r") as f:
            all_results = json.load(f)
    except FileNotFoundError:
        all_results = {}

 
    key = f"J={J_}_h={h_}_Lambda={Lambda_}"  # 在key中包含Lambda
    all_results[key] = result

    with open(results_file_path, "w") as f:
        json.dump(all_results, f, indent=4)

     
    dump(model_Trunc, model_file_path_)
    print(f"Model saved to {model_file_path_}")
    return model_Trunc


"""Start model training"""

# para settings
input_dim = n_layers * (2*n_qubits -1)

# 测试不同的Lambda值（截断参数）- 从1到7
Lambda_values = [1, 2, 3, 4, 5, 6, 7]

# 定义三个数据集配置
datasets = [
    {
        "filename": "data_shadow_4b.json",  # Noiseless data
        "name": "shadow",
        "base_dir": 'results/data/noise_sims_data/shadow_results'
    },
    {
        "filename": "noise_data_shadow.json",  # Global T1T2 noise
        "name": "noise_shadow",
        "base_dir": 'results/data/noise_sims_data/noise_shadow_results'
    },
    {
        "filename": "noise_data_shadow_tls.json",  # TLS noise
        "name": "tls_noise_shadow",
        "base_dir": 'results/data/noise_sims_data/tls_noise_shadow_results'
    }
]

# 为每个Lambda值和数据集训练模型
for Lambda in Lambda_values:
    print(f"\n{'#'*60}")
    print(f"# Testing Lambda = {Lambda} (Truncation Parameter)")
    print(f"{'#'*60}\n")
    
    for dataset_config in datasets:
        # 为每个Lambda创建单独的输出文件夹
        lambda_dir = os.path.join(dataset_config['base_dir'], f'lambda_{Lambda}')
        os.makedirs(lambda_dir, exist_ok=True)
        
        results_file = os.path.join(lambda_dir, 'ML_fitting_results.json')
        
        print(f"\n{'='*60}")
        print(f"Training {dataset_config['name']} Dataset (Lambda={Lambda})")
        print(f"File: {dataset_config['filename']}")
        print(f"Output Directory: {lambda_dir}")
        print(f"{'='*60}\n")
        
        for j_ in J_list:
            for h_ in h_list:
                print(f"J={j_}, h={h_}, Lambda={Lambda}")
                model_file_path = os.path.join(lambda_dir, f"ML_model_J={j_}.joblib")
                trained_model = ML_predictor(
                    j_, h_, num_train, num_test, Lambda, 
                    results_file, 
                    model_file_path,
                    dataset_config['filename']
                )



"""# **The second part is finding the minimum of the trained model**"""


"""Code implementation for Adam"""

# Compute gradients
def compute_gradient_auto_grad_adam(predictor, feature_map, params, Lambda, sampled_W):
    """
    Compute gradients of the predictor function with respect to circuit parameters.

    Args:
        predictor (object): Trained regression model (e.g., Ridge from sklearn).
        feature_map (callable): Function to compute features.
        params (numpy.ndarray): Circuit parameters of shape (input_dim,).
        Lambda (int): Number of Lambda.
        sampled_W (numpy.ndarray): Frequency matrix for the feature map.

    Returns:
        numpy.ndarray: Gradient vector of the same shape as params.
    """
    input_dim = params.shape[-1]

    # Step 1: Compute feature map
    features = feature_map(params.reshape(1, -1), Lambda)  # Shape: (1, num_features)

    # Step 2: Extract predictor weights
    if hasattr(predictor, "coef_"):
        weights = predictor.coef_.flatten()  
    else:
        raise ValueError("Predictor must have 'coef_' attribute.")

    # Step 3: Compute the derivative of the feature map
    params_expanded = params.reshape(1, -1)  

    # Derivative based on trigonometric features
    feature_map_derivative = np.where(
        sampled_W == 0,
        0,  # If W == 0 -> derivative = 0
        np.where(sampled_W == 1, -np.sin(params_expanded), np.cos(params_expanded))
    ) 

    # Step 4: Combine derivatives with predictor weights
    gradient = np.einsum('j,j...->...', weights, feature_map_derivative)

    return gradient


# Optimizer function
def optimize_with_adam(predictor, feature_map, initial_params, Lambda, sampled_W, learning_rate=0.01, max_iterations=500):
    """
    Optimize circuit parameters using Adam to minimize predictor output.

    Args:
        predictor (object): Trained regression model (e.g., Ridge).
        feature_map (callable): Feature mapping function.
        initial_params (np.ndarray): Initial circuit parameters.
        Lambda (int): Number of Lambda.
        sampled_W (np.ndarray): Frequency matrix.
        learning_rate (float): Adam learning rate.
        max_iterations (int): Number of optimization steps.

    Returns:
        np.ndarray: Optimized parameters.
    """
    params = np.copy(initial_params)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    loss_history = []
    vqe_energy_history = []

    for iteration in range(max_iterations):

        # Print progress every certain steps
        if iteration % 20 == 0 or iteration == max_iterations - 1:
            features = feature_map(params.reshape(1, -1), Lambda)
            current_cost = predictor.predict(features)[0]
            param_numpy = jax.device_get(params)
            loss_history.append(current_cost)
            vqe_energy_history.append({
                "iteration": iteration,
                "parameters": param_numpy.tolist(),  
            })


        gradients = compute_gradient_auto_grad_adam(predictor, feature_map, params, Lambda, sampled_W)

        updates, opt_state = optimizer.update(gradients, opt_state)
        params = optax.apply_updates(params, updates)

    return params, loss_history, vqe_energy_history


# Simulation results for shadow labels

# para settings
input_dim = n_layers * (2*n_qubits -1)
repeat_times = 5

pop_size = 100
n_generations = 50

# 为每个Lambda值和数据集运行优化
for Lambda in Lambda_values:
    print(f"\n{'#'*60}")
    print(f"# Optimizing Models with Lambda = {Lambda}")
    print(f"{'#'*60}\n")
    
    for dataset_config in datasets:
        lambda_dir = os.path.join(dataset_config['base_dir'], f'lambda_{Lambda}')
        
        print(f"\n{'='*60}")
        print(f"Optimizing {dataset_config['name']} Dataset Models (Lambda={Lambda})")
        print(f"Output Directory: {lambda_dir}")
        print(f"{'='*60}\n")
        
        for j_ in J_list:
            for h_ in h_list:
                print(f"J={j_}, h={h_}, Lambda={Lambda}")
                model_file_path = os.path.join(lambda_dir, f"ML_model_J={j_}.joblib")
                loaded_trained_model = load(model_file_path)
                print("Model loaded successfully")
                
                # ########################################################
                # ### Minimization of surrogate loss by Adam optimizer ###
                # ########################################################
                for iter_ in range(repeat_times):
                    initial_params = np.random.uniform(-np.pi, np.pi, size=(n_layers, 2 * n_qubits - 1))
                    frequency_set = generate_frequency_vectors(input_dim, Lambda)

                    adam_results_file = os.path.join(lambda_dir, 'Adam_Optimization_Results.json')

                    optimized_params, loss_history, vqe_energy_history = optimize_with_adam(
                        predictor=loaded_trained_model,  # Trained Ridge model
                        feature_map=feature_map_deterministic,
                        initial_params=initial_params,
                        Lambda=Lambda,
                        sampled_W=frequency_set,
                        learning_rate=0.008,
                        max_iterations=501
                    )

                    param_history_file = os.path.join(lambda_dir, f'Adam_params_J={j_}_Repeat{iter_}.json')
                    vqe_energy_history.append({
                        "iteration": 0,
                        "parameters": initial_params.tolist(),  # Convert numpy array to list for JSON compatibility
                    })
                    vqe_energy_history.append({
                        "iteration": 501,
                        "parameters": optimized_params.tolist(),  # Convert numpy array to list for JSON compatibility
                    })
                    with open(param_history_file, "w") as f:
                        json.dump(vqe_energy_history, f, indent=4)

                    print(f"Parameter history saved to {param_history_file}")

                    # Prepare results for JSON
                    result = {
                        "J": j_,
                        "h": h_,
                        "Lambda": Lambda,
                        "iteration": iter_,
                        "loss_history": loss_history,
                    }

                    try:
                        with open(adam_results_file, "r") as f:
                            all_results = json.load(f)
                    except FileNotFoundError:
                        all_results = {}

                    key = f"J={j_}_h={h_}_iter={iter_}"
                    all_results[key] = result

                    with open(adam_results_file, "w") as f:
                        json.dump(all_results, f, indent=4)
 

 