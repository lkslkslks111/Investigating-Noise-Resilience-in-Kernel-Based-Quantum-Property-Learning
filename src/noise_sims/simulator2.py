import pennylane as qml
import pennylane.numpy as np
import os
import json
import time
from pennylane.operation import Channel
from pennylane.measurements import MeasurementProcess

# ============================================================
# Configuration and Setup
# ============================================================
seed = int(666)
np.random.seed(seed)
print(f"[INFO] Running with SEED = {seed}")

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

filename = os.path.join(output_dir, f"noise_shadow_{seed:02d}.json")
print(f"[INFO] Output = {filename}")

data_json = {
    "seed": seed,
    "shadow_labels": [],
    "parameters": []
}


# ============================================================
# Helper Functions
# ============================================================
def append_to_json_shadow(labels, para, filename="data.json"):
    """Append shadow measurement results to JSON file."""
    if not os.path.exists(filename):
        data = {"shadow_labels": [], "parameters": []}
    else:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"shadow_labels": [], "parameters": []}

    def ensure_list(x):
        """Convert various data types to Python list."""
        if x is None:
            return []
        # Handle torch.Tensor
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except ImportError:
            pass

        # Convert numpy types to Python list
        if isinstance(x, (np.ndarray, np.generic)):
            return np.atleast_1d(x).tolist()

        # Convert scalar to list
        if isinstance(x, (float, int)):
            return [float(x)]

        # Handle other iterables
        return list(x)

    data["shadow_labels"].extend(ensure_list(labels))
    data["parameters"].append(ensure_list(para))

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ============================================================
# Quantum Circuit Definition
# ============================================================
def quantum_circuit(params):
    """TFIM ansatz circuit with Hadamard + RX + IsingZZ gates."""
    for w in range(num_qubits):
        qml.Hadamard(wires=w)
        qml.RX(params[w], wires=w)
    for w in range(num_qubits - 1):
        qml.IsingZZ(params[w + num_qubits], wires=[w, w + 1])


# ============================================================
# System Parameters
# ============================================================
num_qubits = 4
h = -0.5
J = -0.1
num_shots = 100
n = 3000

# Construct TFIM Hamiltonian: H = sum_i(h*X_i) + sum_i(J*Z_i*Z_{i+1})
observables = (
    [qml.PauliX(i) for i in range(num_qubits)] +
    [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(num_qubits-1)]
)
coeffs = [h]*num_qubits + [J]*(num_qubits-1)
H = qml.Hamiltonian(coeffs, observables)


# ============================================================
# TLS-Style T1/T2 Fluctuation Model
# ============================================================
# Base thermal relaxation parameters (in seconds)
T1_BASE = 0.02
T2_BASE = 0.03
TG_BASE = 0.001

# Parameter space dimension
d_param = 2 * num_qubits - 1

# TLS hotspot in parameter space: center and width
tls_center = np.zeros(d_param)
tls_sigma = 1.0

def tls_weight(params):
    """Compute TLS weight w(params) in (0, 1], higher near tls_center."""
    diff = params - tls_center
    r2 = np.sum(diff**2)
    return np.exp(-r2 / (2.0 * tls_sigma**2))

def update_drift(prev_drift, scale=0.1, beta=0.1):
    """
    Centered random walk with mean-reversion to prevent unbounded drift.
    
    Args:
        prev_drift: Previous drift value
        scale: Random noise strength
        beta: Mean-reversion coefficient (larger = faster reversion)
    """
    return prev_drift * (1 - beta) + np.random.normal(scale=scale)

def effective_t1_t2(params, drift,
                    t1_base=T1_BASE, t2_base=T2_BASE,
                    max_rel_change=0.5):
    """
    Compute effective T1/T2 based on parameter position and drift.
    
    Args:
        params: Circuit parameters
        drift: Current drift value
        t1_base: Base T1 time
        t2_base: Base T2 time
        max_rel_change: Maximum relative change allowed (e.g., 0.5 = ±50%)
    """
    w = tls_weight(params)
    local = np.clip(w * drift, -max_rel_change, max_rel_change)
    factor = 1.0 + local

    t1_eff = t1_base * factor
    t2_eff = t2_base * factor

    # Ensure positive values
    t1_eff = float(max(t1_eff, 1e-4))
    t2_eff = float(max(t2_eff, 1e-4))
    return t1_eff, t2_eff


# ============================================================
# Noise Model Construction
# ============================================================
@qml.BooleanFn
def fcond_all_gates(op):
    """Apply noise to all quantum gates (exclude channels and measurements)."""
    return (not isinstance(op, Channel)) and (not isinstance(op, MeasurementProcess))

def thermal_noise(op, **kwargs):
    """Add thermal relaxation error after each gate."""
    for wire in op.wires:
        qml.ThermalRelaxationError(
            0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"], wires=wire
        )

def make_noisy_qnode(t1_eff, t2_eff):
    """Construct QNode with specified T1/T2 noise model."""
    metadata = dict(t1=t1_eff, t2=t2_eff, tg=TG_BASE)

    noise_model = qml.NoiseModel(
        {fcond_all_gates: thermal_noise},
        **metadata
    )
    noisy_quantum_circuit = qml.add_noise(quantum_circuit, noise_model)

    dev_noisy = qml.device("default.mixed", wires=num_qubits, shots=num_shots)

    @qml.defer_measurements
    @qml.qnode(dev_noisy, diff_method=None)
    def noisy_shadow_circuit_qnode(params):
        noisy_quantum_circuit(params)
        return qml.shadow_expval(H)

    return noisy_shadow_circuit_qnode


# ============================================================
# Main Execution with TLS Noise
# ============================================================
def noise_shadow_state_reconstruction(num_qubits, drift):
    """Generate random parameters and compute shadow expectation with TLS noise."""
    # Generate random parameters
    params = np.random.rand(num_qubits * 2 - 1) * 2 * np.pi

    # Compute effective T1/T2 for this parameter configuration
    t1_eff, t2_eff = effective_t1_t2(params, drift)

    # Construct QNode with these T1/T2 values
    noisy_qnode = make_noisy_qnode(t1_eff, t2_eff)

    # Classical shadow estimation
    E_shadow = noisy_qnode(params)
    return params, E_shadow, t1_eff, t2_eff

start_time = time.time()

drift = 0.0  # Initial TLS drift

for i in range(n):
    iter_start = time.time()

    # Update TLS drift at each iteration
    drift = update_drift(drift, scale=0.05, beta=0.1)

    params, E_shadow, t1_eff, t2_eff = noise_shadow_state_reconstruction(num_qubits, drift)

    append_to_json_shadow(
        labels=E_shadow,
        para=params,
        filename="noise_data_shadow_tls.json"
    )
    iter_time = time.time() - iter_start


total_time = time.time() - start_time
print(f"[INFO] Total time: {total_time:.2f}s")


