# Investigating Noise Resilience in Kernel-Based Quantum Property Learning

## 项目简介

本项目研究基于核方法的量子性质学习中的噪声韧性问题。主要包括：
- 复现和验证原始论文结果
- 10-bit 量子系统的 PennyLane 和 Qiskit 实现
- 4-bit 系统的噪声模拟分析

## 项目结构

```
.
├── original_code/          # 导师提供的原始代码
├── src/                    # 自己实现的核心代码
│   ├── pennylane_impl/     # PennyLane 实现
│   ├── qiskit_impl/        # Qiskit 实现
│   └── noise_sims/         # 噪声模拟实验
├── notebooks/              # Jupyter notebooks 用于分析和可视化
├── results/                # 实验结果
│   ├── figures/            # 图表
│   └── data/               # 数据输出
└── requirements.txt        # Python 依赖
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用说明

### 1. 复现原始结果

参见 `notebooks/1_Replicate_Original_Results.ipynb`

### 2. 10-bit 系统验证

**PennyLane 实现：**
```bash
python src/pennylane_impl/run_10bit_pennylane.py
```

**Qiskit 实现：**
```bash
python src/qiskit_impl/run_10bit_qiskit.py
```

详细分析参见 `notebooks/2_10bit_Verification.ipynb`

### 3. 4-bit 噪声模拟

**全局噪声：**
```bash
python src/noise_sims/run_4bit_global_noise.py
```

**增强噪声：**
```bash
python src/noise_sims/run_4bit_enhanced_noise.py
```

详细分析参见 `notebooks/3_4bit_Noise_Simulation_Analysis.ipynb`

## 实验结果

所有实验结果和可视化图表保存在 `results/` 目录下。

## 主要发现

（在此总结你的研究发现）

## 参考文献

（添加相关论文引用）

## 作者

- 姓名：[你的名字]
- 导师：[导师名字]
- 日期：2025年11月

## 许可证

（根据需要添加许可证信息）
