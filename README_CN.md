# 基于核方法的量子性质学习中的噪声韧性研究

**[English](README_EN.md) | 中文版**

---

**作者:** lks  
**日期:** 2025年11月17日  
**项目灵感:** Yuxuan Du, et al. "Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits" (arXiv:2408.12199v2) 及相关 [GitHub 仓库](https://github.com/yuxuan-du/Efficient_Predicting_Bounded_Gate_QC)

---

## 1. 项目概述

本项目是论文 "Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits" 的一项拓展性研究。原论文提出了一种基于**经典阴影 (Classical Shadows)** 和**截断三角展开 (Truncated Trigonometric Expansions)** 的核方法 (Kernel-based ML model)，可以高效地学习量子电路的线性特性。

本研究从一个核心问题出发:

> **原论文中的"截断"机制，除了平衡计算开销与近似精度之外，能否在实际含噪硬件上表现为一种"频域滤波器"，在有限样本下减弱热弛豫、TLS 等噪声对学习模型的影响？**

为验证这一假设，本项目做出了以下贡献：

1. **复现与重构:** 使用 PennyLane 和 Qiskit 独立实现了原论文中用于 "Pretraining Hamiltonian-variational ansatz" 任务的量子拟设 (ansatz) 电路，并在10-qubit TFIM（横场伊辛模型）上复现了与原论文相近的VQE结果。

2. **拓展与噪声模拟:** 由于算力限制，我们构建了4-qubit的TFIM模拟。为检验我们的核心假设，我们设计了三个数据集：
   - **Noiseless (`shadow`):** 理想的无噪声模拟
   - **Global Noise (`noise_shadow`):** 施加全局热弛豫噪声
   - **TLS-like Noise (`tls_noise_shadow`):** 施加一种在特定参数点增强的、类似TLS（二能级系统）的热弛豫噪声

3. **截断分析:** 我们使用原论文的训练程序，在上述三个噪声数据集上，系统地训练和分析了截断参数 Λ 从 1 到 7 对模型性能的影响。

---

## 2. 核心方法：截断三角核与噪声建模

本研究从一个核心问题出发：

原论文中的"截断"机制，除了平衡计算开销与近似精度之外，能否在真实含噪硬件上扮演某种"频域滤波器"的角色，在有限样本下减弱热弛豫、TLS 等噪声对学习模型的影响？

### 2.1 原论文中的截断三角核

原论文的目标是学习量子电路在可观测量 $O$ 下的期望值

$$f(x,O)=\text{Tr}(\rho(x)O),$$

其中 $x\in[-\pi,\pi]^d$ 是经典输入参数，$\rho(x)$ 是对应的 $N$-qubit 量子态。

论文提出的学习模型是一个基于核的估计器

$$h_s(x,O)=\frac{1}{n}\sum_{i=1}^{n}\kappa_{\Lambda}(x,x^{(i)})g(x^{(i)},O),$$

其中 $g(x^{(i)},O)$ 来自 Pauli-based classical shadow，是 $\text{Tr}(\tilde{\rho}_T(x^{(i)})O)$ 的估计值。

截断三角单项式核（Truncated Trigonometric Monomial Kernel）定义为

$$\kappa_{\Lambda}(x,x^{(i)})=\sum_{\omega:\|\omega\|_0\le\Lambda}2^{\|\omega\|_0}\Phi_{\omega}(x)\Phi_{\omega}(x^{(i)}),$$

其中三角基

$$\Phi_{\omega}(x)=\prod_{j=1}^{d}\begin{cases}1, & \omega_j=0,\\ \cos(x_j), & \omega_j=1,\\ \sin(x_j), & \omega_j=-1,\end{cases}\quad\omega\in\{0,\pm1\}^d.$$

这里 $\omega$ 可以理解为"频率模式"，$\|\omega\|_0$ 是其非零分量个数。截断参数 $\Lambda$ 控制我们在核中只保留 $\|\omega\|_0\le\Lambda$ 的低阶与中阶频率：

- $\Lambda$ 小：特征少、计算开销小，但高频信息被大量丢弃，截断 bias 较大，容易欠拟合；
- $\Lambda$ 大：特征多、表达能力强，但特征维度 $p(\Lambda)=\sum_{k=0}^{\Lambda}\binom{d}{k}2^k$ 迅速增长，如果样本数 $n$ 不随之增长，估计误差会被放大，容易过拟合。

原论文在一定光滑性假设下证明：只要 $n$ 随 $|C(\Lambda)|\log|C(\Lambda)|$ 适当增长，就可以同时控制截断误差与估计误差，使总体预测误差不超过给定 $\varepsilon$。在本工作中，我们刻意固定 $n$，只扫描 $\Lambda$，从而直接看到理论中 bias–variance 权衡的实证表现，并考察不同噪声模型下这种权衡是否会发生系统性变化。

### 2.2 噪声模型及其在核空间中的分解

在主文中，我们的学习模型使用的训练标签为

$$g(x^{(i)},O)=\text{Tr}(\tilde{\rho}_T(x^{(i)})O),$$

其中 $\tilde{\rho}_T(x)$ 是在含噪声的量子器件上通过 Pauli-based classical shadow（$T$ 个 snapshot）得到的状态估计。换句话说，学习器事实上在拟合的是

$$x\longmapsto f(x,O):=\mathbb{E}[g(x,O)]\approx\text{Tr}(\rho_{\text{noisy}}(x)O),$$

这里 $\rho_{\text{noisy}}(x)$ 是某个噪声信道作用在理想态 $\rho_{\text{id}}(x)$ 上所得的真实硬件状态。

本节给出数值模拟中使用的两类噪声模型，并说明它们如何在三角多项式核的频率展开下表现为对 Fourier 系数的修改，以及截断 $\Lambda$ 对高频成分的影响。我们考虑两种噪声：

1. **均匀（homogeneous）的热弛豫噪声**：在每一个量子门之后、对每个被作用的量子比特施加固定的 $(T_1,T_2)$ 热弛豫信道；
2. **类 TLS 的参数依赖热弛豫噪声**：有效的 $(T_1,T_2)$ 随电路参数变化，并随时间缓慢漂移，用来模拟真实硬件中 TLS hotspot 及其 drift 行为。

两类噪声本质上都是作用在理想态 $\rho_{\text{id}}(x)$ 上的 CPTP（完全正且保迹）信道，因此既可以在算符基（Pauli 基）下分解，也可以在输入空间的三角频率基 $\{\Phi_{\omega}\}$ 下分解，我们正是利用这一点来讨论噪声对截断核学习问题的影响。

#### 2.2.1 均匀热弛豫噪声

第一类噪声是在每一个门后、对该门作用的所有比特施加相同的 thermal relaxation 信道。PennyLane 实现如下：

```python
@qml.BooleanFn
def fcond_all_gates(op):
    """对所有量子门施加噪声（排除已有的 channel 和测量）。"""
    return (not isinstance(op, Channel)) and (not isinstance(op, MeasurementProcess))

def thermal_noise(op, **kwargs):
    """在每个门之后添加热弛豫误差。"""
    for wire in op.wires:
        qml.ThermalRelaxationError(0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"], wire)
```

这里 `t1`, `t2`, `tg` 分别为纵向弛豫时间 $T_1$、横向/退相干时间 $T_2$ 以及门时长。直观上，这是在整个电路上叠加一个空间上均匀的振幅阻尼 + 相位阻尼过程，对所有位置、所有量子比特的影响形式一致。

记由这些局部热弛豫组成的整体信道为 $\mathcal{N}_{\text{hom}}$，则含噪态为

$$\rho_{\text{hom}}(x)=\mathcal{N}_{\text{hom}}(\rho_{\text{id}}(x)).$$

对固定可观测量 $O$ 的期望值为

$$f_{\text{hom}}(x,O):=\text{Tr}(O\rho_{\text{hom}}(x))=\text{Tr}(\mathcal{N}_{\text{hom}}^*(O)\rho_{\text{id}}(x)),$$

其中 $\mathcal{N}_{\text{hom}}^*$ 是对偶映射，作用在可观测量上。因此，均匀热弛豫既可以理解为"把状态 $\rho_{\text{id}}(x)$ 变混、Pauli 分量被衰减"，也可以理解为"把测量算符 $O$ 变成等效测量 $\mathcal{N}_{\text{hom}}^*(O)$"。

从 $x$ 的角度看，函数 $f_{\text{hom}}(x,O)$ 的结构仍然比较光滑，频率域上的主要效果是 Fourier 系数整体缩放，尤其是对应高权重 Pauli 串的系数被更强烈地压缩。

#### 2.2.2 类 TLS 的参数依赖热弛豫噪声

第二类噪声旨在更贴近真实超导硬件行为：实验中量子比特的 $T_1,T_2$ 往往随偏置、电路频率等参数变化，在某些 TLS resonance 附近会出现急剧下降，并且这些"热点"还会缓慢漂移。

我们让有效的 $(T_1,T_2)$ 随电路参数 `params` 以及一个随时间演化的 `drift` 变量变化。

**(1) 参数空间中的 TLS hotspot**

在参数空间中指定一个 TLS 中心 `tls_center`，并用高斯型权重函数衡量当前参数配置离该中心的远近：

```python
tls_center = np.zeros(d_param)
tls_sigma  = 1.0

def tls_weight(params):
    """TLS 权重 w(params) ∈ (0, 1]，在 tls_center 附近最大。"""
    diff = params - tls_center
    r2 = np.sum(diff**2)
    return np.exp(-r2 / (2.0 * tls_sigma**2))
```

当 `params` 接近 `tls_center` 时，$w(\text{params})\approx 1$，表示强烈受 TLS 影响；远离该区域时 $w\ll 1$，噪声回到近似均匀热弛豫。

**(2) 有均值回复的漂移（drift）**

为模拟 TLS 参数随时间缓慢漂移，引入一个带均值回复的随机游走：

```python
def update_drift(prev_drift, scale=0.1, beta=0.1):
    """带均值回复的随机游走，防止漂移发散。"""
    return prev_drift * (1 - beta) + np.random.normal(scale=scale)
```

其中 `beta>0` 控制均值回复强度，`scale` 控制随机扰动大小。这样 `drift` 会随时间缓慢变化，但不会无界增大。

**(3) 有效 $T_1/T_2$ 随 $(\text{params},\text{drift})$ 变化**

综合 TLS 权重与 drift，定义有效的 $T_1,T_2$：

```python
def effective_t1_t2(
    params, drift,
    t1_base=T1_BASE, t2_base=T2_BASE,
    max_rel_change=0.5,
):
    w = tls_weight(params)             # hotspot 强度
    local = np.clip(w * drift, -max_rel_change, max_rel_change)
    factor = 1.0 + local               # 相对变化因子
    t1_eff = t1_base * factor
    t2_eff = t2_base * factor
    # 保证正值
    t1_eff = float(max(t1_eff, 1e-4))
    t2_eff = float(max(t2_eff, 1e-4))
    return t1_eff, t2_eff
```

含义是：

- 靠近 TLS 中心时 $w\approx 1$，$T_1,T_2$ 可以在基准值基础上有最多 ±50% 的变化；
- 远离 TLS 时 $w\ll 1$，退相干时间基本回到基准值；
- `drift` 使得 hotspot 的强弱在不同实验轮次之间缓慢变化。

**(4) 构造带 TLS 噪声的 QNode**

有了 $T_{1}^{\text{eff}}(x),T_{2}^{\text{eff}}(x)$ 之后，我们构造参数依赖的噪声模型：

```python
@qml.BooleanFn
def fcond_all_gates(op):
    """对所有量子门施加噪声（排除已有 channel 和测量）。"""
    return (not isinstance(op, Channel)) and (not isinstance(op, MeasurementProcess))

def thermal_noise(op, **kwargs):
    """在每个门之后添加热弛豫误差。"""
    for wire in op.wires:
        qml.ThermalRelaxationError(
            0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"],
            wires=wire,
        )

def make_noisy_qnode(t1_eff, t2_eff):
    """构造带指定 T1/T2 噪声的 QNode。"""
    metadata    = dict(t1=t1_eff, t2=t2_eff, tg=TG_BASE)
    noise_model = qml.NoiseModel({fcond_all_gates: thermal_noise}, **metadata)
    noisy_quantum_circuit = qml.add_noise(quantum_circuit, noise_model)

    dev_noisy = qml.device("default.mixed", wires=num_qubits, shots=num_shots)

    @qml.defer_measurements
    @qml.qnode(dev_noisy, diff_method=None)
    def noisy_shadow_circuit_qnode(params):
        noisy_quantum_circuit(params)
        return qml.shadow_expval(H)

    return noisy_shadow_circuit_qnode
```

数学上，对每个输入 $x$（或对应参数 $\theta(x)$），得到一个依赖于 $x$ 的 CPTP 映射 $\mathcal{N}_{\text{TLS}}(x)$，其 Kraus 算符由 $T_{1}^{\text{eff}}(\theta(x)),T_{2}^{\text{eff}}(\theta(x))$ 决定。含噪状态为

$$\rho_{\text{TLS}}(x):=\mathcal{N}_{\text{TLS}}(x)(\rho_{\text{id}}(x)),$$

对应期望值

$$f_{\text{TLS}}(x,O):=\text{Tr}(O\rho_{\text{TLS}}(x)).$$

与均匀噪声不同，$f_{\text{TLS}}(x,O)$ 在接近 TLS hotspot 的参数区域会出现比较"尖"的变化（因为 $T_1,T_2$ 在参数空间中有局部急剧变化），再叠加 drift 带来的慢时间随机性，使得整体结构更加复杂。

### 2.3 在三角核与不同基下的分解

主文中我们对理想态 $\rho(x)$ 在输入空间上采用三角多项式展开：

$$\rho(x)=\sum_{\omega\in C(d)}\Phi_{\omega}(x)\rho_{\omega},$$

在固定可观测量 $O$ 下得到

$$f_{\text{id}}(x,O):=\text{Tr}(\rho(x)O)=\sum_{\omega\in C(d)}\Phi_{\omega}(x)\alpha_{\omega},\quad\alpha_{\omega}:=\text{Tr}(\rho_{\omega}O).$$

另一方面，每个算符 $\rho_{\omega}$ 可以在 **Pauli 基** 下展开：

$$\rho_{\omega}=2^{-N}\sum_{s}c_{\omega,s}P_s,\quad O=\sum_{s}o_sP_s,$$

其中 $P_s$ 为 $N$ 比特 Pauli 串。于是

$$\alpha_{\omega}=\text{Tr}(\rho_{\omega}O)=2^{-N}\sum_{s}c_{\omega,s}o_s\text{Tr}(P_s^2),$$

描述了 $\rho_{\omega}$ 在 Pauli 基上的系数 $c_{\omega,s}$ 与 $O$ 的重叠。

#### (a) 均匀热噪声对频率分布的影响

对每个 $\rho_{\omega}$ 施加均匀信道 $\mathcal{N}_{\text{hom}}$ 得到

$$\rho_{\omega}^{\text{hom}}:=\mathcal{N}_{\text{hom}}(\rho_{\omega})=2^{-N}\sum_{s}\lambda_s c_{\omega,s}P_s,$$

其中 $|\lambda_s|\le 1$ 是对 Pauli 串 $P_s$ 的衰减因子，通常权重越大的 Pauli 串衰减越剧烈。于是含噪期望为

$$f_{\text{hom}}(x,O)=\sum_{\omega}\Phi_{\omega}(x)\alpha_{\omega}^{\text{hom}},\quad\alpha_{\omega}^{\text{hom}}:=\text{Tr}(\rho_{\omega}^{\text{hom}}O).$$

在三角频率基 $\{\Phi_{\omega}\}$ 下，均匀热噪声不会改变展开形式，只是将系数 $\alpha_{\omega}$ 整体重缩放：

- 低频（$\|\omega\|_0$ 小）的系数变化不大；
- 高频（$\|\omega\|_0$ 大）对应的高权重 Pauli 串，其系数会被显著压缩。

因此 $f_{\text{hom}}(x,O)$ 比理想情况更光滑，$\mathbb{E}_x\|\nabla_x f_{\text{hom}}(x,O)\|_2^2$ 变小，即 Lemma F.1 中的常数 $C$ 实际上会减小，从而在给定截断 $\Lambda$ 下，截断误差上界 $C/\Lambda$ 反而更小。

#### (b) TLS 型噪声对频率分布的影响

TLS 噪声的信道 $\mathcal{N}_{\text{TLS}}(x)$ 显式依赖于 $x$（或 $\theta(x)$），于是

$$\rho_{\text{TLS}}(x)=\mathcal{N}_{\text{TLS}}(x)(\rho_{\text{id}}(x))=\sum_{\omega}\Phi_{\omega}(x)\rho_{\omega}^{\text{TLS}}(x),$$

其中 $\rho_{\omega}^{\text{TLS}}(x)$ 本身也可以带 $x$ 依赖。对应期望为

$$f_{\text{TLS}}(x,O)=\sum_{\omega}\Phi_{\omega}(x)\alpha_{\omega}^{\text{TLS}}(x),\quad\alpha_{\omega}^{\text{TLS}}(x):=\text{Tr}(\rho_{\omega}^{\text{TLS}}(x)O).$$

直观上，可将 $\mathcal{N}_{\text{TLS}}(x)$ 视为"平均均匀噪声 + 局部扰动"：

$$\mathcal{N}_{\text{TLS}}(x)\approx\overline{\mathcal{N}}_{\text{hom}}+\Delta\mathcal{N}_{\text{TLS}}(x),$$

于是

$$f_{\text{TLS}}(x,O)\approx f_{\text{hom}}(x,O)+\Delta f_{\text{TLS}}(x,O).$$

其中 $\Delta f_{\text{TLS}}(x,O)$ 主要集中在 TLS hotspot 附近的参数区域，是在 $x$ 空间的"局部尖峰结构"。按 Fourier 理论，这类局部、变化较快的结构对应于频率空间中更厚的高频尾巴，即

$$\sum_{\|\omega\|_0>\Lambda}2^{-\|\omega\|_0}|\alpha_{\omega}^{\text{TLS}}|^2\text{ 可能明显大于均匀噪声情形}。$$

这意味着：

- $\mathbb{E}_x\|\nabla_x f_{\text{TLS}}(x,O)\|_2^2$（即 Lemma F.1 假设中的 $C$）会变大；
- 在相同截断水平 $\Lambda$ 下，截断误差 $\mathbb{E}_x|f_{\Lambda}(x,O)-f_{\text{TLS}}(x,O)|^2\lesssim C_{\text{TLS}}/\Lambda$ 更难压低；
- 为达到同样的误差容忍度 $\varepsilon$，需要更大的 $\Lambda$，从而根据 Theorem 2 中 $n\sim|C(\Lambda)|\log|C(\Lambda)|/\varepsilon$ 的 scaling，需要更多样本 $n$。

此外，drift 会增加标签 $g(x^{(i)},O)$ 的随机性，在 Lemma F.2 的 Hoeffding 不等式中体现为更大的有效方差（或更大的 $B^2$），同样推高达到给定误差所需的样本数。

### 2.4 对核模型与截断的综合影响

主文中的核函数为

$$\kappa_{\Lambda}(x,x^{(i)})=\sum_{\omega:\|\omega\|_0\le\Lambda}2^{\|\omega\|_0}\Phi_{\omega}(x)\Phi_{\omega}(x^{(i)}),$$

对应学习模型

$$h_s(x,O)=\frac{1}{n}\sum_{i=1}^{n}\kappa_{\Lambda}(x,x^{(i)})g(x^{(i)},O).$$

需要强调的是：**噪声不会改变核函数本身的形式**（$\Phi_{\omega}(x)$ 和 $\kappa_{\Lambda}$ 都是确定的），**噪声只改变了目标函数**

$$x\mapsto f(x,O)=\mathbb{E}[g(x,O)]$$

**的光滑程度和频率分布**。

- 对于 **均匀热噪声**，$f_{\text{hom}}(x,O)$ 的高频成分被压缩，函数更平滑，主要能量集中在 $\|\omega\|_0$ 较小的频率上。此时，适度的频率截断（中等大小的 $\Lambda$）就足以捕获大部分能量，截断误差项 $C/\Lambda$ 较小。

- 对于 **TLS 型噪声**，$f_{\text{TLS}}(x,O)$ 在 TLS hotspot 附近产生局部、快速变化，使高频 Fourier 系数更大，在截断 $\Lambda$ 之后被"砍掉"的那一部分能量会显著增加：

$$\sum_{\|\omega\|_0>\Lambda}2^{-\|\omega\|_0}|\alpha_{\omega}^{\text{TLS}}|^2\text{ 较大}\implies\text{截断误差更大}。$$

为维持同样的误差上界 $\varepsilon$，需要增大 $\Lambda$（保留更多频率分量），也就需要更大的样本数 $n$ 来学习这些额外高维特征。

**综合来看**：

- **均匀热噪声**：从频域角度看主要起到"低通滤波"的作用，使目标函数更光滑、谱能量更集中在低频，对截断 $\Lambda$ 相对友好；
- **TLS 型噪声**：在参数空间局部区域引入尖锐结构，对应频谱中高频能量增加，使得在给定 $\Lambda$ 下的截断更"损伤信息"，理论上需要更大的 $\Lambda$ 和更多样本才能保持同样的可学习性。

后续第三节的 4-qubit 数值实验，将在这个频域图景下解释为何在有限体系尺寸下，TLS 噪声的"高频尾加粗"效应只部分显现。

---

## 3. 数值结果：TLS 噪声、截断与学习目标的关系

在第二节的频率分解框架下，我们面临一个核心问题：给定一个固定的目标函数 $f(x,O)$（如真实硬件的 $f_{\mathrm{noisy}}$），理论上 $\Lambda$ 越大，截断核对该目标的逼近越好。但在实际量子器件上，训练标签来自含噪状态 $\rho_{\mathrm{noisy}}(x)$，而研究者通常更关心理想态 $\rho_{\mathrm{id}}(x)$ 对应的 $f_{\mathrm{id}}(x,O) = \mathrm{Tr}(\rho_{\mathrm{id}}(x)O)$。这引出一个关键问题：**在有限 $\Lambda$ 和有限样本数 $n$ 下训练出的模型，究竟更接近 $f_{\mathrm{noisy}}$ 还是 $f_{\mathrm{id}}$？** 对于显式依赖 $x$ 且在 hotspot 附近剧烈变化的 TLS 噪声，这个问题尤为微妙。

从理论上讲，若将有噪硬件视为"真实世界"，大 $\Lambda$ 在逼近该目标时更为忠实。然而实践中，研究者往往更关注潜在的理想态结构。此时需要区分信号部分（低频段中 $f_{\mathrm{id}}$ 和 $f_{\mathrm{noisy}}$ 共有的结构）与噪声部分（TLS 在高频尾部额外填充的成分）。更大的 $\Lambda$ 会学习噪声的高频结构，使模型逼近 $f_{\mathrm{noisy}}$；而较小的 $\Lambda$ 通过截断高 $\|\omega\|_0$ 频率，仅保留低阶结构，相当于对"无噪信号+噪声"进行低通滤波，剩余部分更接近 $f_{\mathrm{id}}$ 的低频成分。这正是本项目强调的**"截断作为频域隐式去噪"**的核心含义。

在当前 4-qubit TFIM 实验中，这一图像仅显现雏形。在低截断区间（$\Lambda=1\sim3$），三条曲线（`shadow`、`noise_shadow`、`tls_noise_shadow`）几乎重合（MSE 约 $0.03\sim0.04$，$R^2$ 约 $0.89\sim0.92$），表明 TLS 噪声的高频结构未能完全被推至高阶模式，部分仍泄漏到低阶频率中。因此低截断在去除部分噪声的同时，也损失了理想态与平滑噪声共享的信息，"偏向理想态"的效果不显著。当 $\Lambda$ 增至 $5,6,7$ 时，所有数据集严重过拟合（MSE 飙升、$R^2$ 变负），TLS 曲线表现略差，提示其高频噪声使大 $\Lambda$ 模型更偏向有噪硬件。但此时已进入样本数远小于特征维度的高方差区，难以区分"忠实学习 $f_{\mathrm{TLS}}$"与"纯统计过拟合"的贡献。

总结而言，在 4-qubit + 当前 TLS 参数设置下，低截断区间的"低通滤波"抑制作用存在但较弱，高截断区间则伴随严重过拟合。我们预期在更高比特数、更高参数维度及更尖锐 TLS 条件下，将清晰观察到"截断学习理想态 vs. 有噪态"的分叉现象：对同一组含噪标签，低 $\Lambda$ 训练的模型更接近 $f_{\mathrm{id}}(x,O)$（仅保留低频、受 TLS 污染轻微的部分），而高 $\Lambda$ 模型则更接近 $f_{\mathrm{TLS}}(x,O)$（能分辨 hotspot 和 drift 细节，但需更多样本和更强正则化）。本次 4-qubit 结果表明，TLS 噪声向高频尾填充结构、使大 $\Lambda$ 模型偏向有噪态的趋势已经显现，但由于系统规模限制，"高维尾部"尚未充分发展，低截断对理想态的偏向仅呈现温和的一阶效应，而非理论预期的显著分叉。

---

## 4. 仓库结构

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

## 5. 安装

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

## 6. 使用说明

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

## 7. 实验结果

所有实验结果和可视化保存在 `results/` 目录中：
- **shadow_results/**: 无噪声经典阴影测量（Lambda 1-7）
- **noise_shadow_results/**: 全局热噪声结果（固定 T1/T2，Lambda 1-7）
- **tls_noise_shadow_results/**: TLS 波动噪声结果（Lambda 1-7）

每个子目录包含：
- Adam 优化结果
- ML 模型拟合性能指标
- 训练模型检查点（.joblib 文件）

---

## 8. 引用

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

## 9. 许可证

本项目采用MIT许可证开源。详见`LICENSE`文件。

---

## 10. 联系方式

如有问题或合作咨询，请在GitHub上提交issue或直接联系作者。
