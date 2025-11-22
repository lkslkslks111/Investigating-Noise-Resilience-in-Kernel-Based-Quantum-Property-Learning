"""4-qubit TFIM simulator using PennyLane for ideal and shadow measurements."""
import pennylane as qml
import pennylane.numpy as np
import time
from pennylane.operation import Channel
from pennylane.measurements import MeasurementProcess
import json
import os
import numpy as np

np.random.seed(666)

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

    # 写回文件
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def append_to_json_ideal(labels, para, filename="data.json"):
    """Append ideal measurement results to JSON file."""
    if not os.path.exists(filename):
        data = {"labels": [], "para": []}
    else:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"labels": [], "para": []}

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

    data["labels"].extend(ensure_list(labels))
    data["para"].append(ensure_list(para))

    # 写回文件
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
# Quantum Devices and Circuit Execution
# ============================================================
dev_shadow = qml.device("default.qubit", wires=num_qubits, shots=num_shots)
dev_exact = qml.device("lightning.qubit", wires=num_qubits)

@qml.defer_measurements
@qml.qnode(dev_shadow)
def shadow_circuit(params):
    """Execute circuit with classical shadow measurement."""
    quantum_circuit(params)
    return qml.shadow_expval(H)


@qml.qnode(dev_exact)
def exact_circuit(params):
    """Execute circuit with exact expectation value."""
    quantum_circuit(params)
    return qml.expval(H)


def ideal_and_shadow_state_reconstruction(num_qubits):
    """Generate random parameters and compute both exact and shadow expectation values."""
    params = np.random.rand(num_qubits*2 - 1) * np.pi * 2
    E_shadow = shadow_circuit(params)
    E_exact = exact_circuit(params)
    return params, E_shadow, E_exact


# ============================================================
# Main Execution Loop
# ============================================================
for i in range(n):
    params, E_shadow, E_exact = ideal_and_shadow_state_reconstruction(num_qubits)
    append_to_json_ideal(
        labels=E_exact, 
        para=params, 
        filename=os.path.join("results", "data", "pennylane_data", "4b", "data_ideal_4b.json")
    )
    append_to_json_shadow(
        labels=E_shadow, 
        para=params, 
        filename=os.path.join("results", "data", "pennylane_data", "4b", "data_shadow_4b.json")
    )
    )