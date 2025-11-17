# Investigating Noise Resilience in Kernel-Based Quantum Property Learning
# 基于核方法的量子性质学习中的噪声韧性研究

**Author / 作者:** [Your Name]  
**Date / 日期:** November 17, 2025 / 2025年11月17日  
**Inspired by / 项目灵感:** Yuxuan Du, et al. "Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits" (arXiv:2408.12199v2) and the associated [GitHub Repository](https://github.com/yuxuan-du/Efficient_Predicting_Bounded_Gate_QC)

---

## 1. Project Overview / 项目概述

**English:**
This project is an extension study of the paper "Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits". The original paper proposed a kernel-based ML model using **Classical Shadows** and **Truncated Trigonometric Expansions** to efficiently learn linear properties of quantum circuits.

This research begins with a core question:

> **Can the "truncation" mechanism in the original paper not only balance computational cost with accuracy, but also serve as an implicit filter to effectively remove the main components of quantum hardware noise (such as thermal relaxation noise)?**

To verify this hypothesis, this project makes the following contributions:

1. **Reproduction & Reconstruction:** Independently implemented the quantum ansatz circuits from the original paper using PennyLane and Qiskit for the "Pretraining Hamiltonian-variational ansatz" task, reproducing VQE results similar to the original paper on a 10-qubit TFIM (Transverse Field Ising Model).

2. **Extension & Noise Simulation:** Due to computational limitations, we constructed 4-qubit TFIM simulations. To test our core hypothesis, we designed three datasets:
   - **Noiseless (`shadow`):** Ideal noise-free simulation
   - **Global Noise (`noise_shadow`):** Applied global thermal relaxation noise
   - **TLS-like Noise (`tls_noise_shadow`):** Applied a TLS (Two-Level System)-like thermal relaxation noise that intensifies at specific parameter points

3. **Truncation Analysis:** Using the original paper's training procedure, we systematically trained and analyzed the impact of truncation parameter Λ from 1 to 7 on model performance across the three noise datasets.

**中文:**
本项目是论文 "Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits" 的一项拓展性研究。原论文提出了一种基于**经典阴影 (Classical Shadows)** 和**截断三角展开 (Truncated Trigonometric Expansions)** 的核方法 (Kernel-based ML model)，可以高效地学习量子电路的线性特性。

本研究从一个核心问题出发：

> **原论文中的"截断"机制，是否不仅能平衡计算开销与精度，还能作为一种隐式的滤波器，有效剔除量子硬件噪声（如热弛豫噪声）的主要分量？**

为验证这一假设，本项目做出了以下贡献：

1. **复现与重构:** 使用 PennyLane 和 Qiskit 独立实现了原论文中用于 "Pretraining Hamiltonian-variational ansatz" 任务的量子拟设 (ansatz) 电路，并在10-qubit TFIM（横场伊辛模型）上复现了与原论文相近的VQE结果。

2. **拓展与噪声模拟:** 由于算力限制，我们构建了4-qubit的TFIM模拟。为检验我们的核心假设，我们设计了三个数据集：
   - **Noiseless (`shadow`):** 理想的无噪声模拟
   - **Global Noise (`noise_shadow`):** 施加全局热弛豫噪声
   - **TLS-like Noise (`tls_noise_shadow`):** 施加一种在特定参数点增强的、类似TLS（二能级系统）的热弛豫噪声

3. **截断分析:** 我们使用原论文的训练程序，在上述三个噪声数据集上，系统地训练和分析了截断参数 Λ 从 1 到 7 对模型性能的影响。

---

## 2. Core Method: The Truncated Kernel / 核心方法：截断三角核

**English:**
To understand the core of this project, here is a brief introduction to the key formulas from the original paper:

**Objective:** Learn the expectation value of a quantum circuit $f(x, O) = \text{Tr}(\rho(x)O)$.

**Model:** The ML model proposed in the paper is a kernel estimator $h_s(x, O)$:

$$h_{s}(x,O)=\frac{1}{n}\sum_{i=1}^{n}\kappa_{\Lambda}(x,x^{(i)})g(x^{(i)},O)$$

where $g(x^{(i)},O)$ is the estimate obtained from classical shadows.

**Kernel Function:** The core is the **Truncated Trigonometric Monomial Kernel** $\kappa_{\Lambda}$:

$$\kappa_{\Lambda}(x,x^{(i)})=\sum_{\omega,||\omega||_{0}\le\Lambda}2^{||\omega||_{0}}\Phi_{\omega}(x)\Phi_{\omega}(x^{(i)})$$

It depends on a trigonometric basis $\Phi_{\omega}(x)$:

$$\Phi_{\omega}(x)=\prod_{i=1}^{d}\begin{cases}1&\text{if } \omega_{i}=0\\ \cos(x_{i})&\text{if } \omega_{i}=1\\ \sin(x_{i})&\text{if } \omega_{i}=-1\end{cases}$$

**Key Parameter Λ (Lambda):**
This is the **Truncation Parameter** Λ that we study. It limits the order of the Fourier series used by the kernel function (only keeping terms with $||\omega||_{0} \le \Lambda$).

- **Low Λ:** Fewer features, faster computation, but may "underfit"
- **High Λ:** More features, slower computation, but may "overfit"

The original paper proved that Λ controls the trade-off between **accuracy** and **computational cost**. The number of features $p$ grows with Λ as: $p = \sum_{k=0}^{\Lambda}\binom{d}{k}2^{k}$.

**Our Hypothesis:** Λ not only controls computational cost but also controls the model's sensitivity to **noise**. We hypothesize that there exists an "optimal" Λ that retains enough signal (`cos`, `sin`) for learning while **truncating** the complex, erroneous high-order terms introduced by high-frequency noise (such as TLS).

**中文:**
为了理解本项目的核心，这里简要介绍原论文中的关键公式：

**目标：** 学习一个量子电路的期望值 $f(x, O) = \text{Tr}(\rho(x)O)$。

**模型：** 论文提出的ML模型是一个核估计器 $h_s(x, O)$：

$$h_{s}(x,O)=\frac{1}{n}\sum_{i=1}^{n}\kappa_{\Lambda}(x,x^{(i)})g(x^{(i)},O)$$

其中 $g(x^{(i)},O)$ 是从经典阴影中得到的估计值。

**核函数：** 核心是**截断三角单项式核 (Truncated Trigonometric Monomial Kernel)** $\kappa_{\Lambda}$：

$$\kappa_{\Lambda}(x,x^{(i)})=\sum_{\omega,||\omega||_{0}\le\Lambda}2^{||\omega||_{0}}\Phi_{\omega}(x)\Phi_{\omega}(x^{(i)})$$

它依赖于一个三角基 $\Phi_{\omega}(x)$：

$$\Phi_{\omega}(x)=\prod_{i=1}^{d}\begin{cases}1&\text{if } \omega_{i}=0\\ \cos(x_{i})&\text{if } \omega_{i}=1\\ \sin(x_{i})&\text{if } \omega_{i}=-1\end{cases}$$

**关键参数 Λ (Lambda):**
这就是我们研究的**截断参数 (Truncation Parameter)** Λ。它限制了核函数所使用的傅里叶级数的阶数（只保留 $||\omega||_{0} \le \Lambda$ 的项）。

- **低 Λ:** 特征少，计算快，但可能"欠拟合"
- **高 Λ:** 特征多，计算慢，但可能"过拟合"

原论文证明了 Λ 控制着**精度**和**计算开销**之间的权衡。特征数量 $p$ 随 Λ 的增长关系为：$p = \sum_{k=0}^{\Lambda}\binom{d}{k}2^{k}$。

**我们的假设：** Λ 不仅控制计算开销，还控制着模型对**噪声**的敏感度。我们推测存在一个"最佳"Λ，它能保留足够的信号（`cos`, `sin`）来进行学习，同时**截断**掉由高频噪声（如TLS）引入的复杂、错误的高阶项。

---

## 3. Key Findings and Analysis / 关键发现与分析

**English:**
Our 4-qubit noise experiment data (shown in the figure below) strongly supports our hypothesis.

*(The figure above is the output of `analyze_truncation_results.py`, showing trends of all key metrics as Λ varies)*

**Finding 1: An "Optimal Truncation Point" Exists (Lambda = 2)**

- Observe the **MSE** (Mean Squared Error), **R² Score**, and **Pearson Correlation** plots (top row).
- At Λ=1, the model is too simple with poor performance (underfitting).
- At **Λ=2**, all three metrics simultaneously reach their **optimal values**: lowest MSE (~0.03), highest R² (~0.9).

**Finding 2: Truncation Effectively "Filters Out" Noise**

- This is the core conclusion of this project. At the "optimal point" Λ=2, the three curves—`Shadow` (noiseless, blue), `noise_shadow` (global noise, red), `tls_noise_shadow` (TLS noise, green)—**almost completely overlap**.
- This strongly demonstrates that when Λ=2, the model not only predicts most accurately but is also **completely insensitive to both types of thermal relaxation noise we introduced**, successfully verifying our hypothesis.

**Finding 3: "Curse of Dimensionality" Leads to Overfitting**

- Why does performance drop dramatically when Λ > 2?
- The **Feature Dimension** plot (bottom right) provides the answer: the number of features grows **exponentially** with Λ.
- When Λ ≥ 3, the model becomes too complex and begins to learn **noise** rather than **signal**, leading to "overfitting". This is particularly evident in the R² plot, where R² becomes negative as Λ increases, meaning the model's predictions are worse than random guessing.

**中文:**
我们的4-qubit噪声实验数据（如下图所示）有力地支持了我们的假设。

*(上图是 `analyze_truncation_results.py` 的输出，展示了所有关键指标随 Λ 变化的趋势)*

**发现 1：存在"最佳截断点" (Lambda = 2)**

- 观察 **MSE** (均方误差)、**R² Score** 和 **Pearson Correlation** 图（上排）。
- 在 Λ=1 时，模型过于简单，性能不佳（欠拟合）。
- 在 **Λ=2** 时，所有三个指标同时达到**最佳值**：MSE最低 (约 0.03)，R²最高 (约 0.9)。

**发现 2：截断可有效"滤除"噪声**

- 这是本项目的核心结论。在 Λ=2 这个"最佳点"上，三条曲线——`Shadow` (无噪声, 蓝色), `noise_shadow` (全局噪声, 红色), `tls_noise_shadow` (TLS噪声, 绿色)——**几乎完全重合**。
- 这有力地证明，当 Λ=2 时，模型不仅预测最准，而且**对我们引入的两种热弛豫噪声完全不敏感**，成功验证了我们的假设。

**发现 3："维度灾难"导致过拟合**

- 为什么 Λ > 2 时性能反而急剧下降？
- **Feature Dimension** (特征维度) 图（右下）给出了答案：特征数量随 Λ **指数级增长**。
- 当 Λ ≥ 3 时，模型变得过于复杂，它开始学习数据中的**噪声**而不是**信号**，导致"过拟合"。这在 R² 图中表现得尤为明显，当 Λ 增大时，R² 变为负数，意味着模型的预测效果还不如一个随机猜测。

---

## 4. Repository Structure / 仓库结构

## 4. Repository Structure / 仓库结构

**English:**
```
.
├── src/                           # Core implementation code
│   ├── noise_sims/               # (Core Contribution) 4-qubit noise simulation experiments
│   │   ├── simulator1.py         # 1. Global thermal relaxation noise simulator
│   │   ├── simulator2.py         # 2. TLS-like noise simulator
│   │   ├── pre_train_4_qubit_tfim_benchmark.py  # 3. Run pretraining for Lambda=1..7
│   │   └── analyze_truncation_results.py  # 4. Generate 6-panel analysis figure
│   ├── pennylane_impl/           # PennyLane implementations
│   │   ├── simulator_4b.py       # 4-qubit TFIM simulator
│   │   ├── simulator_10b.py      # 10-qubit TFIM simulator
│   │   ├── pre_train_4-qubit_tfim_benchmark.py
│   │   ├── pre_train_10qubit_tfim_benchmark.py
│   │   └── learning_curve_analysis.py
│   └── qiskit_impl/              # Qiskit implementations
│       ├── simulator_qiskit.py   # 10-qubit Qiskit simulator
│       └── pre_train_10-qubit_tfim_benchmark.py
├── results/                      # Raw experimental data
│   └── data/
│       ├── noise_sims_data/      # 4-qubit noise simulation results
│       │   ├── shadow_results/   # Noiseless (Lambda 1-7)
│       │   ├── noise_shadow_results/  # Global noise (Lambda 1-7)
│       │   └── tls_noise_shadow_results/  # TLS noise (Lambda 1-7)
│       ├── pennylane_data/       # PennyLane simulation data (4b & 10b)
│       └── qiskit_data/          # Qiskit simulation data (10b)
└── requirements.txt              # Python dependencies
```

**中文:**
```
.
├── src/                           # 核心实现代码
│   ├── noise_sims/               # (核心贡献) 4-qubit 噪声模拟实验
│   │   ├── simulator1.py         # 1. 全局热弛豫噪声模拟器
│   │   ├── simulator2.py         # 2. TLS-like 噪声模拟器
│   │   ├── pre_train_4_qubit_tfim_benchmark.py  # 3. 运行 Lambda=1..7 的预训练
│   │   └── analyze_truncation_results.py  # 4. 生成6格分析图
│   ├── pennylane_impl/           # PennyLane 实现
│   │   ├── simulator_4b.py       # 4-qubit TFIM 模拟器
│   │   ├── simulator_10b.py      # 10-qubit TFIM 模拟器
│   │   ├── pre_train_4-qubit_tfim_benchmark.py
│   │   ├── pre_train_10qubit_tfim_benchmark.py
│   │   └── learning_curve_analysis.py
│   └── qiskit_impl/              # Qiskit 实现
│       ├── simulator_qiskit.py   # 10-qubit Qiskit 模拟器
│       └── pre_train_10-qubit_tfim_benchmark.py
├── results/                      # 实验原始数据
│   └── data/
│       ├── noise_sims_data/      # 4-qubit 噪声模拟结果
│       │   ├── shadow_results/   # Noiseless (Lambda 1-7)
│       │   ├── noise_shadow_results/  # Global noise (Lambda 1-7)
│       │   └── tls_noise_shadow_results/  # TLS noise (Lambda 1-7)
│       ├── pennylane_data/       # PennyLane 模拟数据 (4b & 10b)
│       └── qiskit_data/          # Qiskit 模拟数据 (10b)
└── requirements.txt              # Python 依赖
```

---

## 5. Installation / 安装

**English:**
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/quantum-kernel-noise-resilience.git
   cd quantum-kernel-noise-resilience
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Key dependencies:**
   - PennyLane (≥0.37.0)
   - Qiskit (≥1.0.0)
   - Qiskit Aer / Qiskit IBM Runtime
   - Qiskit Addon OBP (for classical shadows)
   - NumPy, SciPy, scikit-learn, Matplotlib
   - JAX, Optax (for optimization)

**中文:**
1. **克隆仓库：**
   ```bash
   git clone https://github.com/yourusername/quantum-kernel-noise-resilience.git
   cd quantum-kernel-noise-resilience
   ```

2. **安装依赖：**
   ```bash
   pip install -r requirements.txt
   ```

   **关键依赖：**
   - PennyLane (≥0.37.0)
   - Qiskit (≥1.0.0)
   - Qiskit Aer / Qiskit IBM Runtime
   - Qiskit Addon OBP (用于经典阴影)
   - NumPy, SciPy, scikit-learn, Matplotlib
   - JAX, Optax (用于优化)

---

## 6. Usage / 使用说明

**English:**

### 6.1 Reproduce Core Noise Experiments (Figure 1 in paper)
**Run the complete 4-qubit noise simulation pipeline:**

```bash
# Step 1: Generate noiseless baseline data (Lambda 1-7)
python src/noise_sims/simulator1.py --noise_type none

# Step 2: Generate global thermal noise data
python src/noise_sims/simulator1.py --noise_type global

# Step 3: Generate TLS-like noise data
python src/noise_sims/simulator2.py

# Step 4: Run ML model training for all Lambda values (1-7) and all noise conditions
python src/noise_sims/pre_train_4_qubit_tfim_benchmark.py

# Step 5: Generate analysis plots (6-panel figure)
python src/noise_sims/analyze_truncation_results.py
```

**Output:**
- Raw data saved to `results/data/noise_sims_data/`
- Analysis plots in script output directory

### 6.2 PennyLane Simulations

**4-qubit experiment:**
```bash
python src/pennylane_impl/simulator_4b.py
python src/pennylane_impl/pre_train_4-qubit_tfim_benchmark.py
```

**10-qubit experiment:**
```bash
python src/pennylane_impl/simulator_10b.py
python src/pennylane_impl/pre_train_10qubit_tfim_benchmark.py
```

**Learning curve analysis:**
```bash
python src/pennylane_impl/learning_curve_analysis.py
```

### 6.3 Qiskit Simulations

**10-qubit Qiskit implementation:**
```bash
python src/qiskit_impl/simulator_qiskit.py
python src/qiskit_impl/pre_train_10-qubit_tfim_benchmark.py
```

---

**中文:**

### 6.1 重现核心噪声实验（论文图1）
**运行完整的4-qubit噪声模拟流程：**

```bash
# 步骤1: 生成无噪声基线数据 (Lambda 1-7)
python src/noise_sims/simulator1.py --noise_type none

# 步骤2: 生成全局热噪声数据
python src/noise_sims/simulator1.py --noise_type global

# 步骤3: 生成TLS-like噪声数据
python src/noise_sims/simulator2.py

# 步骤4: 运行所有Lambda值（1-7）和所有噪声条件的ML模型训练
python src/noise_sims/pre_train_4_qubit_tfim_benchmark.py

# 步骤5: 生成分析图（6格图）
python src/noise_sims/analyze_truncation_results.py
```

**输出：**
- 原始数据保存至 `results/data/noise_sims_data/`
- 分析图在脚本输出目录

### 6.2 PennyLane 模拟

**4-qubit 实验：**
```bash
python src/pennylane_impl/simulator_4b.py
python src/pennylane_impl/pre_train_4-qubit_tfim_benchmark.py
```

**10-qubit 实验：**
```bash
python src/pennylane_impl/simulator_10b.py
python src/pennylane_impl/pre_train_10qubit_tfim_benchmark.py
```

**学习曲线分析：**
```bash
python src/pennylane_impl/learning_curve_analysis.py
```

### 6.3 Qiskit 模拟

**10-qubit Qiskit 实现：**
```bash
python src/qiskit_impl/simulator_qiskit.py
python src/qiskit_impl/pre_train_10-qubit_tfim_benchmark.py
```

---

## 7. Experimental Results / 实验结果

**English:**
All experimental results and visualizations are saved in the `results/` directory:
- **shadow_results/**: Noiseless classical shadow measurements (Lambda 1-7)
- **noise_shadow_results/**: Global thermal noise results (fixed T1/T2, Lambda 1-7)
- **tls_noise_shadow_results/**: TLS fluctuating noise results (Lambda 1-7)

Each subdirectory contains:
- Adam optimization results
- ML model fitting performance metrics
- Trained model checkpoints (.joblib files)

**中文:**
所有实验结果和可视化保存在 `results/` 目录中：
- **shadow_results/**: 无噪声经典阴影测量（Lambda 1-7）
- **noise_shadow_results/**: 全局热噪声结果（固定 T1/T2，Lambda 1-7）
- **tls_noise_shadow_results/**: TLS 波动噪声结果（Lambda 1-7）

每个子目录包含：
- Adam 优化结果
- ML 模型拟合性能指标
- 训练模型检查点（.joblib 文件）

---

## 8. Key Findings Summary / 关键发现总结

**English:**

Our experiments reveal a **critical "sweet spot"** for kernel truncation:

1. **Lambda=2 achieves optimal noise resilience** across all three noise models
2. **Performance degradation pattern:**
   - Lambda=1 (too simple): Fails to capture quantum correlations
   - Lambda=3-7 (too complex): Accumulates noise faster than information
   - Lambda=2 (Goldilocks zone): Best balance of expressivity vs. noise susceptibility

3. **Noise model comparison:**
   - TLS-like noise (worst): Performance drops significantly at Lambda≥3
   - Global thermal noise (moderate): Moderate degradation pattern
   - Noiseless (best): Consistent high performance for Lambda≥2

4. **Feature dimension explosion:** The number of features grows exponentially with Lambda ($p = \sum_{k=0}^{\Lambda}\binom{d}{k}2^{k}$), leading to overfitting when Lambda>2

**中文:**

我们的实验揭示了一个**关键的"最优点"**：

1. **Lambda=2在所有三种噪声模型中实现最优噪声鲁棒性**
2. **性能退化模式：**
   - Lambda=1（过于简单）：无法捕获量子关联
   - Lambda=3-7（过于复杂）：噪声积累速度快于信息获取
   - Lambda=2（最优区间）：表达能力与噪声敏感性的最佳平衡

3. **噪声模型对比：**
   - TLS-like 噪声（最差）：Lambda≥3 时性能显著下降
   - 全局热噪声（中等）：适度退化模式
   - 无噪声（最佳）：Lambda≥2 时持续高性能

4. **特征维度爆炸：** 特征数量随 Lambda 指数级增长（$p = \sum_{k=0}^{\Lambda}\binom{d}{k}2^{k}$），导致 Lambda>2 时过拟合

---

## 9. References / 参考文献

**English:**
1. **Du, Y., Huang, Z., & Tao, D. (2022).** "Quantum kernel method with guaranteed convergence." *Nature Machine Intelligence, 4*(12), 1171-1182.
   - Introduced truncated trigonometric quantum kernels
   - Established theoretical convergence guarantees

2. **Huang, H.-Y., Kueng, R., & Preskill, J. (2020).** "Predicting many properties of a quantum system from very few measurements." *Nature Physics, 16*(10), 1050-1057.
   - Foundational work on classical shadows protocol

3. **Preskill, J. (2018).** "Quantum Computing in the NISQ era and beyond." *Quantum, 2*, 79.
   - Defined NISQ-era constraints motivating noise resilience studies

**中文:**
1. **Du, Y., Huang, Z., & Tao, D. (2022).** "具有保证收敛性的量子核方法。" *Nature Machine Intelligence, 4*(12), 1171-1182.
   - 提出截断三角量子核
   - 建立理论收敛保证

2. **Huang, H.-Y., Kueng, R., & Preskill, J. (2020).** "从极少测量预测量子系统的多种性质。" *Nature Physics, 16*(10), 1050-1057.
   - 经典阴影协议的基础工作

3. **Preskill, J. (2018).** "NISQ时代及未来的量子计算。" *Quantum, 2*, 79.
   - 定义NISQ时代约束，激发噪声鲁棒性研究

---

## 10. Citation / 引用

**English:**
If you use this code or findings in your research, please cite:

```bibtex
@misc{quantum-kernel-noise-2025,
  author = {Lks},
  title = {Investigating Noise Resilience in Kernel-Based Quantum Property Learning},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/quantum-kernel-noise-resilience}}
}
```

**中文:**
如果您在研究中使用此代码或发现，请引用：

```bibtex
@misc{quantum-kernel-noise-2025,
  author = {Lks},
  title = {Investigating Noise Resilience in Kernel-Based Quantum Property Learning},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/quantum-kernel-noise-resilience}}
}
```

---

## 11. License / 许可证

**English:**
This project is open-source under the MIT License. See `LICENSE` file for details.

**中文:**
本项目采用MIT许可证开源。详见`LICENSE`文件。

---

## 12. Contact / 联系方式

**English:**
For questions or collaboration inquiries, please open an issue on GitHub or contact the author directly.

**中文:**
如有问题或合作咨询，请在GitHub上提交issue或直接联系作者。
