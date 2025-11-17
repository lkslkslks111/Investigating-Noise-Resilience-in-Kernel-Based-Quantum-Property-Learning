"""
4-bit 系统的全局噪声模拟
"""

import numpy as np
import pandas as pd
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
import matplotlib.pyplot as plt


def create_4bit_circuit(x):
    """
    创建 4-bit 量子电路
    
    Args:
        x: 输入参数
        
    Returns:
        量子电路
    """
    qc = QuantumCircuit(4, 4)
    
    # 应用旋转门
    for i in range(4):
        qc.ry(x[i % len(x)], i)
    
    # 应用纠缠门
    for i in range(3):
        qc.cx(i, i + 1)
    
    # 测量
    qc.measure(range(4), range(4))
    
    return qc


def run_with_global_noise(noise_levels, n_samples=100):
    """
    在不同全局噪声水平下运行实验
    
    Args:
        noise_levels: 噪声水平列表
        n_samples: 样本数量
        
    Returns:
        结果字典
    """
    results = {
        'noise_level': [],
        'fidelity': [],
        'success_rate': []
    }
    
    np.random.seed(42)
    X = np.random.randn(n_samples, 4)
    
    for noise_level in noise_levels:
        print(f"\n运行噪声水平: {noise_level}")
        
        # 创建噪声模型
        noise_model = NoiseModel()
        error = depolarizing_error(noise_level, 1)
        noise_model.add_all_qubit_quantum_error(error, ['ry', 'cx'])
        
        # 创建模拟器
        simulator = AerSimulator(noise_model=noise_model)
        
        fidelities = []
        success_rates = []
        
        for x in X[:10]:  # 使用前10个样本进行演示
            qc = create_4bit_circuit(x)
            result = simulator.run(qc, shots=1024).result()
            counts = result.get_counts()
            
            # 计算保真度（零态概率）
            zero_state = '0000'
            fidelity = counts.get(zero_state, 0) / 1024
            fidelities.append(fidelity)
            
            # 计算成功率（任意标准）
            success_rate = sum(counts.values()) / 1024
            success_rates.append(success_rate)
        
        results['noise_level'].append(noise_level)
        results['fidelity'].append(np.mean(fidelities))
        results['success_rate'].append(np.mean(success_rates))
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("4-bit Quantum System - Global Noise Simulation")
    print("=" * 60)
    
    # 定义噪声水平
    noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    
    # 运行实验
    print("\n开始运行全局噪声模拟...")
    results = run_with_global_noise(noise_levels)
    
    # 保存数据
    output_dir = Path(__file__).parent.parent.parent / "results" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    output_file = output_dir / "4bit_global_noise_out.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 创建可视化
    fig_dir = Path(__file__).parent.parent.parent / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['noise_level'], results['fidelity'], 'o-', label='Fidelity')
    plt.xlabel('Noise Level')
    plt.ylabel('Fidelity')
    plt.title('4-bit System: Fidelity vs Global Noise Level')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    fig_file = fig_dir / "4bit_global_noise_plot.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {fig_file}")
    
    print("\n" + "=" * 60)
    print("运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
