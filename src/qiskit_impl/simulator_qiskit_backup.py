import os
import json
import numpy as np
from numpy.random import default_rng

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from qiskit.primitives import StatevectorSampler

# ================== 全局参数 ==================

np.random.seed(666)
rng = default_rng(666)

num_qubits = 10
h = -0.5
J = -0.1
num_shots = 100      # classical shadows 的 shot 数
n = 7500             # 采样多少组参数

# ================== 构造 Ising 哈密顿量 H ==================
# PennyLane: H = sum_i h X_i + sum_i J Z_i Z_{i+1}
# Qiskit 的 SparsePauliOp 使用 “右边是 qubit 0” 的约定，
# 所以我们要做一个 index 映射：逻辑 qubit i -> Pauli 字符串中位置 (num_qubits - 1 - i)

def pauli_string_x(i, num_qubits):
    # X_i 作用在逻辑 qubit i
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - i] = "X"
    return "".join(chars)

def pauli_string_zz(i, j, num_qubits):
    # Z_i Z_j 作用在逻辑 qubit i, j
    chars = ["I"] * num_qubits
    chars[num_qubits - 1 - i] = "Z"
    chars[num_qubits - 1 - j] = "Z"
    return "".join(chars)

labels = []
coeffs = []

# 单体 X_i 项
for i in range(num_qubits):
    labels.append(pauli_string_x(i, num_qubits))
    coeffs.append(h)

# 相邻 Z_i Z_{i+1} 项
for i in range(num_qubits - 1):
    labels.append(pauli_string_zz(i, i + 1, num_qubits))
    coeffs.append(J)

H_op = SparsePauliOp(labels, coeffs=np.array(coeffs, dtype=float))


# ================== JSON 写入工具（基本照搬你原来的） ==================

def append_to_json_shadow(labels, para, filename="data_shadow.json"):
    # 如果文件不存在，就创建一个空结构
    if not os.path.exists(filename):
        data = {"shadow_labels": [], "parameters": []}
    else:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"shadow_labels": [], "parameters": []}

    def ensure_list(x):
        if x is None:
            return []
        # torch.Tensor -> numpy
        try:
            import torch
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
        except ImportError:
            pass

        # numpy 类型 -> python list
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
    if not os.path.exists(filename):
        data = {"labels": [], "para": []}
    else:
        with open(filename, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {"labels": [], "para": []}

    def ensure_list(x):
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


# ================== 电路构造：对应你的 quantum_circuit(params) ==================

def build_ising_circuit(params):
    """
    params: 长度 = 2*num_qubits - 1
      前 num_qubits 个 -> 每个 qubit 的 RX
      后 num_qubits-1 个 -> 每个相邻 Ising ZZ 耦合的角度
    """
    qc = QuantumCircuit(num_qubits)

    # H + RX
    for w in range(num_qubits):
        qc.h(w)
        qc.rx(float(params[w]), w)

    # Ising ZZ 耦合：PennyLane IsingZZ(phi) ~ Qiskit rzz(phi)
    for w in range(num_qubits - 1):
        theta = float(params[w + num_qubits])
        qc.rzz(theta, w, w + 1)

    return qc


# ================== 期望值计算 ==================

def compute_exact_energy(circuit, h_op: SparsePauliOp) -> float:
    """理想期望值，对应原来的 lightning.qubit + expval(H)"""
    sv = Statevector.from_instruction(circuit)
    val = sv.expectation_value(h_op)
    return float(np.real_if_close(val))


def compute_shadow_energy(circuit, h_op: SparsePauliOp,
                          povm_sampler: POVMSampler,
                          cs_povm: ClassicalShadows,
                          shots: int) -> float:
    """classical shadows 估计的期望值，对应原来的 shadow_expval(H)"""
    # 这里直接用 POVMSampler.run 的简单接口：
    # job = povm_sampler.run([qc], shots=shots, povm=measurement)
    job = povm_sampler.run([circuit], shots=shots, povm=cs_povm)
    result = job.result()
    pub_result = result[0]

    post_processor = POVMPostProcessor(pub_result)
    exp_val, std = post_processor.get_expectation_value(h_op)
    return float(np.real_if_close(exp_val))


# ================== classical shadows Sampler 初始化 ==================

# 理想噪声自由模拟，用 StatevectorSampler（和你原来 default.qubit 的语义比较接近）
sampler = StatevectorSampler(seed=rng)
povm_sampler = POVMSampler(sampler=sampler)

# Classical shadows POVM（随机 X/Y/Z 基测量）
cs_povm = ClassicalShadows(num_qubits=num_qubits, seed=1234)


# ================== 主循环 ==================

num_params = 2 * num_qubits - 1

for i in range(n):
    # 和你原来一样的参数采样方式
    params = rng.random(num_params) * 2 * np.pi

    qc = build_ising_circuit(params)

    # 理想值（exact）
    E_exact = compute_exact_energy(qc, H_op)

    # classical shadows 估计值
    E_shadow = compute_shadow_energy(qc, H_op, povm_sampler, cs_povm, num_shots)

    # 写入 JSON，结构与原始代码保持一致
    append_to_json_ideal(labels=E_exact, para=params, filename="data_ideal_qiskit.json")
    append_to_json_shadow(labels=E_shadow, para=params, filename="data_shadow_qiskit.json")

    if (i + 1) % 100 == 0:
        print(f"[INFO] finished {i+1}/{n} samples")
