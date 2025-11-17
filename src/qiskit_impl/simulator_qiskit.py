"""10-qubit TFIM simulator using Qiskit with classical shadows."""
import os
import json
import numpy as np
from numpy.random import default_rng

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_addon_obp.utils.povm import POVMSampler, POVMPostProcessor, ClassicalShadows


# ============================================================
# Global Parameters
# ============================================================
np.random.seed(666)
rng = default_rng(666)

num_qubits = 10
h = -0.5
J = -0.1
num_shots = 100
n = 7500


# ============================================================
# TFIM Hamiltonian Construction
# ============================================================
# Qiskit uses right-to-left ordering: logical qubit i -> position (num_qubits - 1 - i)

def pauli_string_x(i, num_qubits):
    """Create Pauli X string for qubit i."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - i] = "X"
    return "".join(chars)


def pauli_string_zz(i, j, num_qubits):
    """Create Pauli ZZ string for qubits i and j."""
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - i] = "Z"
    chars[num_qubits - 1 - j] = "Z"
    return "".join(chars)


# Construct Hamiltonian: H = sum_i(h*X_i) + sum_i(J*Z_i*Z_{i+1})
labels = []
coeffs = []

# Single-qubit X_i terms
for i in range(num_qubits):
    labels.append(pauli_string_x(i, num_qubits))
    coeffs.append(h)

# Nearest-neighbor Z_i Z_{i+1} terms
for i in range(num_qubits - 1):
    labels.append(pauli_string_zz(i, i + 1, num_qubits))
    coeffs.append(J)

H_op = SparsePauliOp(labels, coeffs=np.array(coeffs, dtype=float))


# ============================================================
# Helper Functions for JSON Output
# ============================================================
def append_to_json_shadow(labels, para, filename="data_shadow.json"):
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

        if isinstance(x, (float, int)):
            return [float(x)]

        return list(x)

    data["shadow_labels"].extend(ensure_list(labels))
    data["parameters"].append(ensure_list(para))

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def append_to_json_ideal(labels, para, filename="data_ideal.json"):
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
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except ImportError:
            pass

        if isinstance(x, (np.ndarray, np.generic)):
            return np.atleast_1d(x).tolist()

        if isinstance(x, (float, int)):
            return [float(x)]

        return list(x)

    data["labels"].extend(ensure_list(labels))
    data["para"].append(ensure_list(para))

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ============================================================
# Quantum Circuit Construction
# ============================================================
def build_ising_circuit(params):
    """
    Build TFIM ansatz circuit.
    
    Args:
        params: Array of length (2*num_qubits - 1)
                First num_qubits elements -> RX rotation angles
                Last num_qubits-1 elements -> IsingZZ coupling angles
    """
    qc = QuantumCircuit(num_qubits)

    # Apply Hadamard and RX gates
    for w in range(num_qubits):
        qc.h(w)
        qc.rx(float(params[w]), w)

    # Apply Ising ZZ couplings (PennyLane IsingZZ ~ Qiskit rzz)
    for w in range(num_qubits - 1):
        theta = float(params[w + num_qubits])
        qc.rzz(theta, w, w + 1)

    return qc


# ============================================================
# Expectation Value Computation
# ============================================================
def compute_exact_energy(circuit, h_op: SparsePauliOp) -> float:
    """Compute exact expectation value from statevector."""
    sv = Statevector.from_instruction(circuit)
    val = sv.expectation_value(h_op)
    return float(np.real_if_close(val))


def compute_shadow_energy(circuit, h_op: SparsePauliOp,
                          povm_sampler: POVMSampler,
                          cs_povm: ClassicalShadows,
                          shots: int) -> float:
    """Compute classical shadow estimate of expectation value."""
    job = povm_sampler.run([circuit], shots=shots, povm=cs_povm)
    result = job.result()
    pub_result = result[0]

    post_processor = POVMPostProcessor(pub_result)
    exp_val, std = post_processor.get_expectation_value(h_op)
    return float(np.real_if_close(exp_val))


# ============================================================
# Classical Shadows Sampler Initialization
# ============================================================
# Use StatevectorSampler for ideal noise-free simulation
sampler = StatevectorSampler(seed=rng)
povm_sampler = POVMSampler(sampler=sampler)

# Classical shadows POVM (random X/Y/Z basis measurements)
cs_povm = ClassicalShadows(num_qubits=num_qubits, seed=1234)


# ============================================================
# Main Execution Loop
# ============================================================
num_params = 2 * num_qubits - 1

for i in range(n):
    # Generate random parameters
    params = rng.random(num_params) * 2 * np.pi

    qc = build_ising_circuit(params)

    # Compute exact expectation value
    E_exact = compute_exact_energy(qc, H_op)

    # Compute classical shadow estimate
    E_shadow = compute_shadow_energy(qc, H_op, povm_sampler, cs_povm, num_shots)

    # Write results to JSON files
    append_to_json_ideal(labels=E_exact, para=params, filename="data_ideal_qiskit.json")
    append_to_json_shadow(labels=E_shadow, para=params, filename="data_shadow_qiskit.json")

    if (i + 1) % 100 == 0:
        print(f"[INFO] Finished {i+1}/{n} samples")

print(f"[INFO] Completed all {n} samples")
