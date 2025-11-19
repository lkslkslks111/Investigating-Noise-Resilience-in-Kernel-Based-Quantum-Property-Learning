# Investigating Noise Resilience in Kernel-Based Quantum Property Learning

**[中文版](README_CN.md) | English**

---

**Author:** lks
**Date:** November 17, 2025  
**Inspired by:** Yuxuan Du, et al. "Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits" (arXiv:2408.12199v2) and the associated [GitHub Repository](https://github.com/yuxuan-du/Efficient_Predicting_Bounded_Gate_QC)

---

## 1. Project Overview

This project is an extension study of the paper "Efficient Learning for Linear Properties of Bounded-Gate Quantum Circuits". The original paper proposed a kernel-based ML model using **Classical Shadows** and **Truncated Trigonometric Expansions** to efficiently learn linear properties of quantum circuits.

This research begins with a core question:

> **Can the "truncation" mechanism in the original paper, beyond balancing computational cost with approximation accuracy, act as a "frequency-domain filter" on real noisy hardware to mitigate the impact of thermal relaxation, TLS, and other noise sources on the learning model under limited sampling?**

To verify this hypothesis, this project makes the following contributions:

1. **Reproduction & Reconstruction:** Independently implemented the quantum ansatz circuits from the original paper using PennyLane and Qiskit for the "Pretraining Hamiltonian-variational ansatz" task, reproducing VQE results similar to the original paper on a 10-qubit TFIM (Transverse Field Ising Model).

2. **Extension & Noise Simulation:** Due to computational limitations, we constructed 4-qubit TFIM simulations. To test our core hypothesis, we designed three datasets:
   - **Noiseless (`shadow`):** Ideal noise-free simulation
   - **Global Noise (`noise_shadow`):** Applied global thermal relaxation noise
   - **TLS-like Noise (`tls_noise_shadow`):** Applied a TLS (Two-Level System)-like thermal relaxation noise that intensifies at specific parameter points

3. **Truncation Analysis:** Using the original paper's training procedure, we systematically trained and analyzed the impact of truncation parameter Λ from 1 to 7 on model performance across the three noise datasets.

---

## 2. Core Method: Truncated Trigonometric Kernel and Noise Modeling

This research begins with a core question:

Can the "truncation" mechanism in the original paper, beyond balancing computational cost with approximation accuracy, act as a "frequency-domain filter" on real noisy hardware to mitigate the impact of thermal relaxation, TLS, and other noise sources on the learning model under limited sampling?

### 2.1 Truncated Trigonometric Kernel from the Original Paper

The original paper aims to learn the expectation value of a quantum circuit under observable $O$:

$$f(x,O)=\text{Tr}(\rho(x)O),$$

where $x\in[-\pi,\pi]^d$ are classical input parameters, and $\rho(x)$ is the corresponding $N$-qubit quantum state.

The learning model proposed in the paper is a kernel-based estimator:

$$h_s(x,O)=\frac{1}{n}\sum_{i=1}^{n}\kappa_{\Lambda}(x,x^{(i)})g(x^{(i)},O),$$

where $g(x^{(i)},O)$ comes from Pauli-based classical shadow, and is an estimate of $\text{Tr}(\tilde{\rho}_T(x^{(i)})O)$.

The Truncated Trigonometric Monomial Kernel is defined as:

$$\kappa_{\Lambda}(x,x^{(i)})=\sum_{\omega:\|\omega\|_0\le\Lambda}2^{\|\omega\|_0}\Phi_{\omega}(x)\Phi_{\omega}(x^{(i)}),$$

where the trigonometric basis is:

$$\Phi_{\omega}(x)=\prod_{j=1}^{d}\begin{cases}1, & \omega_j=0,\\ \cos(x_j), & \omega_j=1,\\ \sin(x_j), & \omega_j=-1,\end{cases}\quad\omega\in\{0,\pm1\}^d.$$

Here $\omega$ can be understood as a "frequency mode", and $\|\omega\|_0$ is the number of its non-zero components. The truncation parameter $\Lambda$ controls that we only keep frequency components with $\|\omega\|_0\le\Lambda$ in the kernel:

- **Small $\Lambda$:** Fewer features, lower computational cost, but high-frequency information is largely discarded, leading to larger truncation bias and potential underfitting;
- **Large $\Lambda$:** More features, stronger expressivity, but the feature dimension $p(\Lambda)=\sum_{k=0}^{\Lambda}\binom{d}{k}2^k$ grows rapidly. If sample size $n$ does not grow accordingly, estimation error will be amplified, leading to potential overfitting.

The original paper proves under certain smoothness assumptions: as long as $n$ grows appropriately with $|C(\Lambda)|\log|C(\Lambda)|$, both truncation error and estimation error can be controlled simultaneously, keeping the total prediction error below a given $\varepsilon$. In this work, we deliberately fix $n$ and only scan $\Lambda$, to directly observe the empirical manifestation of the bias-variance tradeoff in the theory, and examine whether this tradeoff changes systematically under different noise models.

### 2.2 Noise Models and Their Decomposition in Kernel Space

In the main text, the training labels used by our learning model are:

$$g(x^{(i)},O)=\text{Tr}(\tilde{\rho}_T(x^{(i)})O),$$

where $\tilde{\rho}_T(x)$ is the state estimate obtained through Pauli-based classical shadow ($T$ snapshots) on a noisy quantum device. In other words, the learner is actually fitting:

$$x\longmapsto f(x,O):=\mathbb{E}[g(x,O)]\approx\text{Tr}(\rho_{\text{noisy}}(x)O),$$

where $\rho_{\text{noisy}}(x)$ is the real hardware state obtained by applying some noise channel to the ideal state $\rho_{\text{id}}(x)$.

This section describes the two types of noise models used in numerical simulations, and explains how they manifest as modifications to Fourier coefficients under the frequency expansion of the trigonometric polynomial kernel, as well as the impact of truncation $\Lambda$ on high-frequency components. We consider two types of noise:

1. **Homogeneous thermal relaxation noise:** After each quantum gate, apply a fixed $(T_1,T_2)$ thermal relaxation channel to each qubit acted upon;
2. **TLS-like parameter-dependent thermal relaxation noise:** The effective $(T_1,T_2)$ varies with circuit parameters and drifts slowly over time, to simulate TLS hotspots and their drift behavior in real hardware.

Both types of noise are essentially CPTP (completely positive and trace-preserving) channels acting on the ideal state $\rho_{\text{id}}(x)$, so they can be decomposed both in the operator basis (Pauli basis) and in the trigonometric frequency basis $\{\Phi_{\omega}\}$ of the input space. We use this to discuss the impact of noise on the truncated kernel learning problem.

#### 2.2.1 Homogeneous Thermal Relaxation Noise

The first type of noise applies the same thermal relaxation channel to all qubits after each gate. PennyLane implementation:

```python
@qml.BooleanFn
def fcond_all_gates(op):
    """Apply noise to all quantum gates (excluding existing channels and measurements)."""
    return (not isinstance(op, Channel)) and (not isinstance(op, MeasurementProcess))

def thermal_noise(op, **kwargs):
    """Add thermal relaxation error after each gate."""
    for wire in op.wires:
        qml.ThermalRelaxationError(0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"], wire)
```

Here `t1`, `t2`, `tg` are the longitudinal relaxation time $T_1$, transverse/dephasing time $T_2$, and gate duration, respectively. Intuitively, this superimposes a spatially uniform amplitude damping + phase damping process over the entire circuit, affecting all positions and qubits uniformly.

Let the overall channel composed of these local thermal relaxations be denoted as $\mathcal{N}_{\text{hom}}$, then the noisy state is:

$$\rho_{\text{hom}}(x)=\mathcal{N}_{\text{hom}}(\rho_{\text{id}}(x)).$$

For a fixed observable $O$, the expectation value is:

$$f_{\text{hom}}(x,O):=\text{Tr}(O\rho_{\text{hom}}(x))=\text{Tr}(\mathcal{N}_{\text{hom}}^*(O)\rho_{\text{id}}(x)),$$

where $\mathcal{N}_{\text{hom}}^*$ is the dual map acting on observables. Therefore, homogeneous thermal relaxation can be understood both as "making the state $\rho_{\text{id}}(x)$ mixed, with Pauli components attenuated" and as "turning the measurement operator $O$ into an effective measurement $\mathcal{N}_{\text{hom}}^*(O)$".

From the perspective of $x$, the structure of the function $f_{\text{hom}}(x,O)$ remains relatively smooth, with the main effect in the frequency domain being an overall rescaling of Fourier coefficients, especially stronger compression of coefficients corresponding to high-weight Pauli strings.

#### 2.2.2 TLS-like Parameter-Dependent Thermal Relaxation Noise

The second type of noise aims to more closely mimic real superconducting hardware behavior: in experiments, the $T_1, T_2$ of qubits often vary with bias, circuit frequency, and other parameters, showing sharp drops near certain TLS resonances, and these "hotspots" also drift slowly.

We let the effective $(T_1,T_2)$ vary with circuit parameters `params` and a time-evolving `drift` variable.

**(1) TLS hotspot in parameter space**

Specify a TLS center `tls_center` in parameter space, and use a Gaussian-type weight function to measure how close the current parameter configuration is to this center:

```python
tls_center = np.zeros(d_param)
tls_sigma  = 1.0

def tls_weight(params):
    """TLS weight w(params) ∈ (0, 1], maximum near tls_center."""
    diff = params - tls_center
    r2 = np.sum(diff**2)
    return np.exp(-r2 / (2.0 * tls_sigma**2))
```

When `params` is close to `tls_center`, $w(\text{params})\approx 1$, indicating strong TLS influence; when far from this region, $w\ll 1$, and noise returns to approximately homogeneous thermal relaxation.

**(2) Mean-reverting drift**

To simulate the slow drift of TLS parameters over time, introduce a mean-reverting random walk:

```python
def update_drift(prev_drift, scale=0.1, beta=0.1):
    """Mean-reverting random walk to prevent drift divergence."""
    return prev_drift * (1 - beta) + np.random.normal(scale=scale)
```

where `beta>0` controls mean-reversion strength, and `scale` controls random perturbation size. This way, `drift` changes slowly over time but does not grow unboundedly.

**(3) Effective $T_1/T_2$ varying with $(\text{params},\text{drift})$**

Combining TLS weight and drift, define effective $T_1,T_2$:

```python
def effective_t1_t2(
    params, drift,
    t1_base=T1_BASE, t2_base=T2_BASE,
    max_rel_change=0.5,
):
    w = tls_weight(params)             # hotspot strength
    local = np.clip(w * drift, -max_rel_change, max_rel_change)
    factor = 1.0 + local               # relative change factor
    t1_eff = t1_base * factor
    t2_eff = t2_base * factor
    # ensure positive values
    t1_eff = float(max(t1_eff, 1e-4))
    t2_eff = float(max(t2_eff, 1e-4))
    return t1_eff, t2_eff
```

Meaning:

- Near the TLS center when $w\approx 1$, $T_1,T_2$ can vary by up to ±50% from baseline values;
- Far from TLS when $w\ll 1$, decoherence times return basically to baseline values;
- `drift` causes the hotspot strength to vary slowly between different experimental runs.

**(4) Constructing QNode with TLS noise**

With $T_{1}^{\text{eff}}(x),T_{2}^{\text{eff}}(x)$, we construct a parameter-dependent noise model:

```python
@qml.BooleanFn
def fcond_all_gates(op):
    """Apply noise to all quantum gates (excluding existing channels and measurements)."""
    return (not isinstance(op, Channel)) and (not isinstance(op, MeasurementProcess))

def thermal_noise(op, **kwargs):
    """Add thermal relaxation error after each gate."""
    for wire in op.wires:
        qml.ThermalRelaxationError(
            0.1, kwargs["t1"], kwargs["t2"], kwargs["tg"],
            wires=wire,
        )

def make_noisy_qnode(t1_eff, t2_eff):
    """Construct QNode with specified T1/T2 noise."""
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

Mathematically, for each input $x$ (or corresponding parameter $\theta(x)$), we obtain an $x$-dependent CPTP map $\mathcal{N}_{\text{TLS}}(x)$, whose Kraus operators are determined by $T_{1}^{\text{eff}}(\theta(x)),T_{2}^{\text{eff}}(\theta(x))$. The noisy state is:

$$\rho_{\text{TLS}}(x):=\mathcal{N}_{\text{TLS}}(x)(\rho_{\text{id}}(x)),$$

with corresponding expectation value:

$$f_{\text{TLS}}(x,O):=\text{Tr}(O\rho_{\text{TLS}}(x)).$$

Unlike homogeneous noise, $f_{\text{TLS}}(x,O)$ exhibits relatively "sharp" changes in parameter regions near the TLS hotspot (because $T_1,T_2$ have local sharp variations in parameter space), combined with the slow-time randomness brought by drift, making the overall structure more complex.

### 2.3 Decomposition in Trigonometric Kernel and Different Bases

In the main text, we adopt a trigonometric polynomial expansion for the ideal state $\rho(x)$ in the input space:

$$\rho(x)=\sum_{\omega\in C(d)}\Phi_{\omega}(x)\rho_{\omega},$$

Under a fixed observable $O$, we obtain:

$$f_{\text{id}}(x,O):=\text{Tr}(\rho(x)O)=\sum_{\omega\in C(d)}\Phi_{\omega}(x)\alpha_{\omega},\quad\alpha_{\omega}:=\text{Tr}(\rho_{\omega}O).$$

On the other hand, each operator $\rho_{\omega}$ can be expanded in the **Pauli basis**:

$$\rho_{\omega}=2^{-N}\sum_{s}c_{\omega,s}P_s,\quad O=\sum_{s}o_sP_s,$$

where $P_s$ is an $N$-qubit Pauli string. Thus:

$$\alpha_{\omega}=\text{Tr}(\rho_{\omega}O)=2^{-N}\sum_{s}c_{\omega,s}o_s\text{Tr}(P_s^2),$$

describing the overlap between the coefficients $c_{\omega,s}$ of $\rho_{\omega}$ in the Pauli basis and $O$.

#### (a) Impact of Homogeneous Thermal Noise on Frequency Distribution

Applying the homogeneous channel $\mathcal{N}_{\text{hom}}$ to each $\rho_{\omega}$ yields:

$$\rho_{\omega}^{\text{hom}}:=\mathcal{N}_{\text{hom}}(\rho_{\omega})=2^{-N}\sum_{s}\lambda_s c_{\omega,s}P_s,$$

where $|\lambda_s|\le 1$ is the attenuation factor for Pauli string $P_s$, typically with higher-weight Pauli strings attenuated more severely. Thus the noisy expectation is:

$$f_{\text{hom}}(x,O)=\sum_{\omega}\Phi_{\omega}(x)\alpha_{\omega}^{\text{hom}},\quad\alpha_{\omega}^{\text{hom}}:=\text{Tr}(\rho_{\omega}^{\text{hom}}O).$$

Under the trigonometric frequency basis $\{\Phi_{\omega}\}$, homogeneous thermal noise does not change the expansion form, but only rescales the coefficients $\alpha_{\omega}$ overall:

- Low-frequency (small $\|\omega\|_0$) coefficients change little;
- High-frequency (large $\|\omega\|_0$) coefficients corresponding to high-weight Pauli strings are significantly compressed.

Therefore $f_{\text{hom}}(x,O)$ is smoother than the ideal case, $\mathbb{E}_x\|\nabla_x f_{\text{hom}}(x,O)\|_2^2$ becomes smaller, i.e., the constant $C$ in Lemma F.1 actually decreases, so that under a given truncation $\Lambda$, the truncation error upper bound $C/\Lambda$ becomes even smaller.

#### (b) Impact of TLS-type Noise on Frequency Distribution

The TLS noise channel $\mathcal{N}_{\text{TLS}}(x)$ explicitly depends on $x$ (or $\theta(x)$), so:

$$\rho_{\text{TLS}}(x)=\mathcal{N}_{\text{TLS}}(x)(\rho_{\text{id}}(x))=\sum_{\omega}\Phi_{\omega}(x)\rho_{\omega}^{\text{TLS}}(x),$$

where $\rho_{\omega}^{\text{TLS}}(x)$ itself can also have $x$ dependence. The corresponding expectation is:

$$f_{\text{TLS}}(x,O)=\sum_{\omega}\Phi_{\omega}(x)\alpha_{\omega}^{\text{TLS}}(x),\quad\alpha_{\omega}^{\text{TLS}}(x):=\text{Tr}(\rho_{\omega}^{\text{TLS}}(x)O).$$

Intuitively, we can view $\mathcal{N}_{\text{TLS}}(x)$ as "average homogeneous noise + local perturbation":

$$\mathcal{N}_{\text{TLS}}(x)\approx\overline{\mathcal{N}}_{\text{hom}}+\Delta\mathcal{N}_{\text{TLS}}(x),$$

thus:

$$f_{\text{TLS}}(x,O)\approx f_{\text{hom}}(x,O)+\Delta f_{\text{TLS}}(x,O).$$

where $\Delta f_{\text{TLS}}(x,O)$ is mainly concentrated in parameter regions near the TLS hotspot, forming "local spike structures" in $x$ space. According to Fourier theory, such local, rapidly-varying structures correspond to a thicker high-frequency tail in frequency space:

$$\sum_{\|\omega\|_0>\Lambda}2^{-\|\omega\|_0}|\alpha_{\omega}^{\text{TLS}}|^2\text{ may be significantly larger than in the homogeneous noise case}.$$

This means:

- $\mathbb{E}_x\|\nabla_x f_{\text{TLS}}(x,O)\|_2^2$ (i.e., the constant $C$ in Lemma F.1 assumption) becomes larger;
- At the same truncation level $\Lambda$, the truncation error $\mathbb{E}_x|f_{\Lambda}(x,O)-f_{\text{TLS}}(x,O)|^2\lesssim C_{\text{TLS}}/\Lambda$ is harder to suppress;
- To achieve the same error tolerance $\varepsilon$, a larger $\Lambda$ is needed, and thus according to the scaling $n\sim|C(\Lambda)|\log|C(\Lambda)|/\varepsilon$ in Theorem 2, more samples $n$ are required.

Additionally, drift increases the randomness of labels $g(x^{(i)},O)$, manifested in Lemma F.2's Hoeffding inequality as a larger effective variance (or larger $B^2$), also pushing up the sample size needed to achieve a given error.

### 2.4 Combined Impact on Kernel Model and Truncation

The kernel function in the main text is:

$$\kappa_{\Lambda}(x,x^{(i)})=\sum_{\omega:\|\omega\|_0\le\Lambda}2^{\|\omega\|_0}\Phi_{\omega}(x)\Phi_{\omega}(x^{(i)}),$$

corresponding to the learning model:

$$h_s(x,O)=\frac{1}{n}\sum_{i=1}^{n}\kappa_{\Lambda}(x,x^{(i)})g(x^{(i)},O).$$

It is important to emphasize: **noise does not change the form of the kernel function itself** ($\Phi_{\omega}(x)$ and $\kappa_{\Lambda}$ are deterministic), **noise only changes the target function**:

$$x\mapsto f(x,O)=\mathbb{E}[g(x,O)]$$

**in terms of smoothness and frequency distribution**.

- For **homogeneous thermal noise**, the high-frequency components of $f_{\text{hom}}(x,O)$ are compressed, the function is smoother, and the main energy is concentrated in frequencies with smaller $\|\omega\|_0$. In this case, moderate frequency truncation (medium-sized $\Lambda$) is sufficient to capture most of the energy, and the truncation error term $C/\Lambda$ is small.

- For **TLS-type noise**, $f_{\text{TLS}}(x,O)$ produces local, rapid changes near the TLS hotspot, making high-frequency Fourier coefficients larger. The energy "cut off" after truncation $\Lambda$ increases significantly:

$$\sum_{\|\omega\|_0>\Lambda}2^{-\|\omega\|_0}|\alpha_{\omega}^{\text{TLS}}|^2\text{ is large}\implies\text{truncation error is large}.$$

To maintain the same error upper bound $\varepsilon$, we need to increase $\Lambda$ (retain more frequency components), which also requires a larger sample size $n$ to learn these additional high-dimensional features.

**In summary**:

- **Homogeneous thermal noise**: From a frequency domain perspective, it mainly acts as a "low-pass filter", making the target function smoother and concentrating spectral energy in low frequencies, which is relatively friendly to truncation $\Lambda$;
- **TLS-type noise**: Introduces sharp structures in local regions of parameter space, corresponding to increased high-frequency energy in the spectrum, making truncation at a given $\Lambda$ more "lossy" in terms of information. Theoretically, a larger $\Lambda$ and more samples are needed to maintain the same learnability.

The subsequent Section 3 4-qubit numerical experiments will explain, under this frequency domain picture, why in finite system sizes, the "high-frequency tail thickening" effect of TLS noise only partially manifests.

---

## 3. Numerical Results: TLS Noise, Truncation, and Learning Targets

Within the frequency-domain decomposition framework of Section 2, we confront a central question: given a fixed target function $f(x,O)$ (such as the true hardware's $f_{\mathrm{noisy}}$), theoretically larger $\Lambda$ provides better approximation to that target. However, on actual quantum devices, training labels derive from noisy states $\rho_{\mathrm{noisy}}(x)$, while researchers typically care more about the ideal state $\rho_{\mathrm{id}}(x)$ and its associated $f_{\mathrm{id}}(x,O) = \mathrm{Tr}(\rho_{\mathrm{id}}(x)O)$. This raises a fundamental question: **at finite $\Lambda$ and finite sample size $n$, does the trained model more closely approximate $f_{\mathrm{noisy}}$ or $f_{\mathrm{id}}$?** For TLS noise, which explicitly depends on $x$ and varies dramatically near hotspots, this question is particularly subtle.

Theoretically, if one regards the noisy hardware as the "true world," larger $\Lambda$ is more faithful in approximating that target. In practice, however, researchers often focus on the underlying ideal-state structure. Here we must distinguish the signal component (structure shared by $f_{\mathrm{id}}$ and $f_{\mathrm{noisy}}$ at low frequencies) from the noise component (TLS-induced high-frequency tail contributions). Larger $\Lambda$ learns the noise's high-frequency structure, making the model approximate $f_{\mathrm{noisy}}$; smaller $\Lambda$, by truncating high-$\|\omega\|_0$ frequencies and retaining only low-order structure, acts as a low-pass filter on "noiseless signal + noise," with the remainder closer to $f_{\mathrm{id}}$'s low-frequency content. This embodies the core concept of **"truncation as implicit frequency-domain denoising"** emphasized in this project.

In the current 4-qubit TFIM experiment, this picture only emerges in nascent form. In the low-truncation regime ($\Lambda=1\sim3$), the three curves (`shadow`, `noise_shadow`, `tls_noise_shadow`) nearly coincide (MSE $\approx 0.03\sim0.04$, $R^2 \approx 0.89\sim0.92$), indicating that TLS noise's high-frequency structure has not been fully relegated to high-order modes, with portions still leaking into low-order frequencies. Consequently, low truncation removes some noise while also sacrificing information shared by the ideal state and smooth noise, rendering the "bias toward ideal state" effect weak. When $\Lambda$ increases to $5,6,7$, all datasets severely overfit (MSE surges, $R^2$ turns negative), with the TLS curve performing slightly worse, suggesting its high-frequency noise biases large-$\Lambda$ models toward noisy hardware. However, at this point we have entered a high-variance regime where sample size is far smaller than feature dimensionality, making it difficult to disentangle "faithfully learning $f_{\mathrm{TLS}}$" from "pure statistical overfitting."

In summary, under the 4-qubit + current TLS parameter setting, the "low-pass filtering" suppression effect in the low-truncation regime exists but remains weak, while the high-truncation regime is dominated by severe overfitting. We anticipate that in higher-qubit, higher-parameter-dimension, and sharper-TLS conditions, a clear bifurcation between "truncation learning ideal vs. noisy state" will emerge: for the same set of noisy labels, low-$\Lambda$ trained models will more closely approximate $f_{\mathrm{id}}(x,O)$ (retaining only low-frequency, lightly TLS-contaminated components), while high-$\Lambda$ models will better approximate $f_{\mathrm{TLS}}(x,O)$ (resolving hotspot and drift details, albeit requiring more samples and stronger regularization). The current 4-qubit results demonstrate that the trend of TLS noise populating high-frequency tails and biasing large-$\Lambda$ models toward noisy states is already evident, but due to system-size limitations, the "high-dimensional tail" has not fully developed, leaving the low-truncation bias toward the ideal state as merely a mild first-order effect rather than the striking bifurcation predicted by theory.

---

## 4. Repository Structure

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

---

## 5. Installation

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

---

## 6. Usage

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

## 7. Experimental Results

All experimental results and visualizations are saved in the `results/` directory:
- **shadow_results/**: Noiseless classical shadow measurements (Lambda 1-7)
- **noise_shadow_results/**: Global thermal noise results (fixed T1/T2, Lambda 1-7)
- **tls_noise_shadow_results/**: TLS fluctuating noise results (Lambda 1-7)

Each subdirectory contains:
- Adam optimization results
- ML model fitting performance metrics
- Trained model checkpoints (.joblib files)

---

## 8. Citation

If you use this code or findings in your research, please cite:

```bibtex
@misc{quantum-kernel-noise-2025,
  author = {Lks},
  title = {Investigating Noise Resilience in Kernel-Based Quantum Property Learning},
  year = {2025},
  howpublished = {\url{https://github.com/yourusername/quantum-kernel-noise-resilience}}
}
```

---

## 9. License

This project is open-source under the MIT License. See `LICENSE` file for details.

---

## 10. Contact

For questions or collaboration inquiries, please open an issue on GitHub or contact the author directly.
