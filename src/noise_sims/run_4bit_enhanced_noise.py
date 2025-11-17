"""
4-bit 系统的增强噪声模拟
"""

import numpy as np
import pandas as pd
from pathlib import Path
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
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


def create_enhanced_noise_model(depol_param, t1, t2, gate_time):
    """
    创建增强的噪声模型（包括退极化和热弛豫）
    
    Args:
        depol_param: 退极化参数
        t1: T1 弛豫时间
        t2: T2 退相干时间
        gate_time: 门操作时间
        
    Returns:
        NoiseModel
    """
    noise_model = NoiseModel()
    
    # 退极化噪声
    depol_error = depolarizing_error(depol_param, 1)
    
    # 热弛豫噪声
    thermal_error = thermal_relaxation_error(t1, t2, gate_time)
    
    # 组合噪声
    combined_error = depol_error.compose(thermal_error)
    
    # 添加到所有门
    noise_model.add_all_qubit_quantum_error(combined_error, ['ry', 'cx'])
    
    return noise_model


def run_with_enhanced_noise(noise_configs, n_samples=100):
    """
    在不同增强噪声配置下运行实验
    
    Args:
        noise_configs: 噪声配置列表
        n_samples: 样本数量
        
    Returns:
        结果字典
    """
    results = {
        'config_name': [],
        'depol_param': [],
        'fidelity': [],
        'std_fidelity': []
    }
    
    np.random.seed(42)
    X = np.random.randn(n_samples, 4)
    
    for config in noise_configs:
        print(f"\n运行配置: {config['name']}")
        
        # 创建噪声模型
        noise_model = create_enhanced_noise_model(
            config['depol_param'],
            config['t1'],
            config['t2'],
            config['gate_time']
        )
        
        # 创建模拟器
        simulator = AerSimulator(noise_model=noise_model)
        
        fidelities = []
        
        for x in X[:10]:  # 使用前10个样本进行演示
            qc = create_4bit_circuit(x)
            result = simulator.run(qc, shots=1024).result()
            counts = result.get_counts()
            
            # 计算保真度
            zero_state = '0000'
            fidelity = counts.get(zero_state, 0) / 1024
            fidelities.append(fidelity)
        
        results['config_name'].append(config['name'])
        results['depol_param'].append(config['depol_param'])
        results['fidelity'].append(np.mean(fidelities))
        results['std_fidelity'].append(np.std(fidelities))
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("4-bit Quantum System - Enhanced Noise Simulation")
    print("=" * 60)
    
    # 定义噪声配置
    noise_configs = [
        {
            'name': 'No Noise',
            'depol_param': 0.0,
            't1': 1e6,
            't2': 1e6,
            'gate_time': 0.1
        },
        {
            'name': 'Low Noise',
            'depol_param': 0.01,
            't1': 50000,
            't2': 70000,
            'gate_time': 0.1
        },
        {
            'name': 'Medium Noise',
            'depol_param': 0.05,
            't1': 30000,
            't2': 40000,
            'gate_time': 0.1
        },
        {
            'name': 'High Noise',
            'depol_param': 0.1,
            't1': 10000,
            't2': 15000,
            'gate_time': 0.1
        },
        {
            'name': 'Very High Noise',
            'depol_param': 0.2,
            't1': 5000,
            't2': 7000,
            'gate_time': 0.1
        }
    ]
    
    # 运行实验
    print("\n开始运行增强噪声模拟...")
    results = run_with_enhanced_noise(noise_configs)
    
    # 保存数据
    output_dir = Path(__file__).parent.parent.parent / "results" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    output_file = output_dir / "4bit_enhanced_noise_out.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # 创建可视化
    fig_dir = Path(__file__).parent.parent.parent / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(results['config_name']))
    plt.bar(x_pos, results['fidelity'], yerr=results['std_fidelity'], 
            capsize=5, alpha=0.7, color='steelblue')
    plt.xticks(x_pos, results['config_name'], rotation=45, ha='right')
    plt.ylabel('Fidelity')
    plt.title('4-bit System: Fidelity under Enhanced Noise Models')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    fig_file = fig_dir / "4bit_enhanced_noise_plot.png"
    plt.savefig(fig_file, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {fig_file}")
    
    print("\n" + "=" * 60)
    print("运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
