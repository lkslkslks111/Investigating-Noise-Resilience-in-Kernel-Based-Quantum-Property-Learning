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
# Noise Model Setup
# ============================================================
# Thermal relaxation parameters (in seconds)
metadata = dict(t1=0.02, t2=0.03, tg=0.001)

@qml.BooleanFn
def fcond_all_gates(op):
    """Apply noise to all quantum gates (exclude channels and measurements)."""
    return (not isinstance(op, Channel)) and (not isinstance(op, MeasurementProcess))

def thermal_noise(op, **kwargs):
    """Add thermal relaxation error after each gate."""
    for wire in op.wires:
        qml.ThermalRelaxationError(0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"], wire)

noise_model = qml.NoiseModel({fcond_all_gates: thermal_noise}, **metadata)
noisy_quantum_circuit = qml.add_noise(quantum_circuit, noise_model)


# ============================================================
# Quantum Device and Circuit Execution
# ============================================================
dev_noisy = qml.device("default.mixed", wires=num_qubits, shots=num_shots)

@qml.defer_measurements
@qml.qnode(dev_noisy, diff_method=None)
def noisy_shadow_circuit_qnode(params):
    """Execute noisy circuit with classical shadow measurement."""
    noisy_quantum_circuit(params)
    return qml.shadow_expval(H)

def noise_shadow_state_reconstruction(num_qubits):
    """Generate random parameters and compute shadow expectation value."""
    params = np.random.rand(num_qubits*2 - 1) * np.pi * 2
    E_shadow = noisy_shadow_circuit_qnode(params)
    return params, E_shadow


# ============================================================
# Main Execution Loop
# ============================================================
start_time = time.time()

for i in range(n):
    iter_start = time.time()
    params, E_shadow = noise_shadow_state_reconstruction(num_qubits)
    append_to_json_shadow(
        labels=E_shadow, 
        para=params, 
        filename="noise_data_shadow.json"
    )
    iter_time = time.time() - iter_start

total_time = time.time() - start_time

