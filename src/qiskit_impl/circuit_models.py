"""
Quantum circuit models for kernel-based learning using Qiskit.
"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit.primitives import Sampler
import numpy as np


class QiskitQuantumKernel:
    """基于 Qiskit 的量子核"""
    
    def __init__(self, n_qubits, n_layers=1, shots=1024):
        """
        初始化量子核
        
        Args:
            n_qubits: 量子比特数
            n_layers: 电路层数
            shots: 测量次数
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots
        self.simulator = AerSimulator()
        
    def feature_map_circuit(self, x):
        """
        创建特征映射电路
        
        Args:
            x: 输入特征向量
            
        Returns:
            QuantumCircuit: 特征映射电路
        """
        qr = QuantumRegister(self.n_qubits, 'q')
        qc = QuantumCircuit(qr)
        
        # 应用旋转门
        for i in range(self.n_qubits):
            qc.ry(x[i % len(x)], qr[i])
        
        # 应用纠缠门
        for i in range(self.n_qubits - 1):
            qc.cx(qr[i], qr[i + 1])
            
        return qc
    
    def kernel_circuit(self, x1, x2):
        """
        构建核电路
        
        Args:
            x1: 第一个输入向量
            x2: 第二个输入向量
            
        Returns:
            QuantumCircuit: 完整的核电路
        """
        qc = QuantumCircuit(self.n_qubits)
        
        # 应用第一个特征映射
        qc_x1 = self.feature_map_circuit(x1)
        qc.compose(qc_x1, inplace=True)
        
        # 应用第二个特征映射的逆
        qc_x2 = self.feature_map_circuit(x2)
        qc.compose(qc_x2.inverse(), inplace=True)
        
        # 添加测量
        qc.measure_all()
        
        return qc
    
    def compute_kernel_value(self, x1, x2):
        """
        计算量子核值
        
        Args:
            x1: 第一个输入向量
            x2: 第二个输入向量
            
        Returns:
            核值
        """
        qc = self.kernel_circuit(x1, x2)
        
        # 运行电路
        result = self.simulator.run(qc, shots=self.shots).result()
        counts = result.get_counts()
        
        # 计算 |0...0> 态的概率
        zero_state = '0' * self.n_qubits
        return counts.get(zero_state, 0) / self.shots
    
    def compute_kernel_matrix(self, X):
        """
        计算核矩阵
        
        Args:
            X: 输入数据矩阵
            
        Returns:
            核矩阵
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i, n_samples):
                K[i, j] = self.compute_kernel_value(X[i], X[j])
                K[j, i] = K[i, j]
                
        return K


def create_10bit_qiskit_circuit():
    """创建 10-bit Qiskit 量子电路"""
    return QiskitQuantumKernel(n_qubits=10, n_layers=1, shots=1024)
