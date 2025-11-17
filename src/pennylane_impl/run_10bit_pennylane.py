"""
运行 10-bit 系统的 PennyLane 实现
"""

import numpy as np
import pandas as pd
from pathlib import Path
from circuit_models import create_10bit_circuit
from tqdm import tqdm


def generate_sample_data(n_samples=100, n_features=10):
    """
    生成示例数据
    
    Args:
        n_samples: 样本数量
        n_features: 特征维度
        
    Returns:
        X: 特征矩阵
        y: 标签向量
    """
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)
    y = (np.sum(X, axis=1) > 0).astype(int)
    return X, y


def main():
    """主函数"""
    print("=" * 60)
    print("10-bit Quantum Kernel Learning - PennyLane Implementation")
    print("=" * 60)
    
    # 生成数据
    print("\n生成示例数据...")
    X_train, y_train = generate_sample_data(n_samples=50, n_features=10)
    X_test, y_test = generate_sample_data(n_samples=20, n_features=10)
    
    # 创建量子电路
    print("创建 10-bit 量子电路...")
    circuit = create_10bit_circuit()
    
    # 计算核矩阵（使用小样本以加快计算）
    print("\n计算训练集核矩阵...")
    K_train = circuit.compute_kernel_matrix(X_train[:10])  # 仅使用前10个样本进行演示
    
    print("\n训练集核矩阵形状:", K_train.shape)
    print("核矩阵对角线元素（应接近1）:", np.diag(K_train)[:5])
    
    # 保存结果
    output_dir = Path(__file__).parent.parent.parent / "results" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "10bit_pennylane_out.csv"
    df = pd.DataFrame(K_train)
    df.to_csv(output_file, index=False)
    
    print(f"\n结果已保存到: {output_file}")
    print("\n" + "=" * 60)
    print("运行完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
