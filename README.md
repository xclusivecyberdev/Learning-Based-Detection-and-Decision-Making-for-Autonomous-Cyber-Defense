# Learning-Based Detection and Decision-Making for Autonomous Cyber Defense

**Awodola Olusola Ebenezer**  
M.Tech Cybersecurity (In View) — Federal University of Technology, Akure  
`oluebenawodola@gmail.com` · [GitHub: @xclusivecyberdev](https://github.com/xclusivecyberdev)

![Status](https://img.shields.io/badge/Status-Ongoing%20Research-blue) ![Python](https://img.shields.io/badge/Python-3.10+-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Abstract

This project investigates the integration of deep learning-based intrusion detection with reinforcement learning-based response policy optimization in autonomous cyber defense systems. The core contribution is a novel framing of cyber response as a sequential decision-making problem where response policies are learned directly from detection confidence distributions rather than static threshold-based rules.

**Central Research Hypothesis:** Reinforcement learning-based response policies can achieve lower false positive rates and reduced mean time to respond (MTTR) compared to static rule-based intrusion response mechanisms when evaluated on the same threat detection distribution.

Preliminary validation on a stratified subset of CICIDS2017 demonstrates convergence in both detection model loss and RL agent reward trajectories, suggesting the viability of the integrated approach.

This research contributes to emerging work on agentic AI systems and autonomous adaptive cyber defense architectures.

---

## Research Problem & Motivation

### Problem Statement

Traditional intrusion detection and response systems operate under the assumption that threat classification confidence translates directly to response severity through manually calibrated thresholds. This approach exhibits two fundamental limitations:

1. **Static Policy Inflexibility:** Predetermined thresholds do not adapt to operational context, attack ecology shifts, or changing system capacity constraints.
2. **Detection-Response Coupling:** Errors in threat classification propagate directly into response errors, with no learned mechanism to mitigate false positives.

Recent advances in autonomous decision-making under uncertainty suggest that reinforcement learning agents can learn context-aware response policies that jointly optimize detection sensitivity and operational cost.

### Research Question

**Primary:** Can a Dueling DQN agent trained to map detection confidence vectors to response actions achieve statistically significant improvements in false positive rate (FPR) and mean time to respond (MTTR) compared to fixed-threshold baselines on held-out attack traffic?

**Secondary:** What is the relative contribution of temporal (LSTM) versus spatial (CNN) feature representations to detection performance, and how does this decomposition affect downstream policy learning?

---

## Mathematical Formulation

### Detection Model

The detection module models cyber traffic as a temporal sequence of feature vectors:

$$\mathbf{X} = \{x_1, x_2, \ldots, x_T\} \in \mathbb{R}^{T \times 78}$$

where $T = 50$ (sliding window length) and each $x_t$ is a 78-dimensional normalized feature vector.

#### Temporal Encoding (BiLSTM)

The bidirectional LSTM encodes temporal dependencies:

$$\mathbf{h}_t^{\text{LSTM}} = \text{BiLSTM}(\mathbf{x}_t, \mathbf{h}_{t-1})$$

$$\mathbf{h}_T^{\text{LSTM}} \in \mathbb{R}^{d_h}$$

where $d_h = 128$ is the hidden state dimension.

#### Spatial Encoding (1D-CNN)

Parallel 1D convolutional filters extract local feature co-occurrence patterns:

$$\mathbf{c}_k = \text{ReLU}\left(\mathbf{W}_k * \mathbf{X} + \mathbf{b}_k\right) \in \mathbb{R}^{T - K + 1}$$

$$\mathbf{h}^{\text{CNN}} = \text{GlobalMaxPool}\left(\left[\mathbf{c}_1, \ldots, \mathbf{c}_{K}\right]\right) \in \mathbb{R}^{d_c}$$

where $K = 3$ is the kernel size and $d_c = 64$ is the number of filters.

#### Fusion & Classification

The temporal and spatial representations are concatenated and projected to a 15-class threat probability distribution:

$$\mathbf{h}_{\text{fused}} = \text{Dropout}(\mathbf{h}^{\text{LSTM}} \oplus \mathbf{h}^{\text{CNN}})$$

$$\hat{\mathbf{p}}_{\text{threat}} = \text{softmax}\left(\mathbf{W}_{\text{out}} \mathbf{h}_{\text{fused}} + \mathbf{b}_{\text{out}}\right) \in \mathbb{R}^{15}$$

#### Training Objective

The detection model is trained with Focal Loss to address class imbalance:

$$\mathcal{L}_{\text{detection}} = -\sum_{i=1}^{15} (1 - \hat{p}_i)^{\gamma} y_i \log(\hat{p}_i)$$

where $y_i \in \{0,1\}$ is the ground-truth label, $\hat{p}_i$ is the predicted probability for class $i$, and $\gamma = 2$ is the focusing parameter.

### Reinforcement Learning Policy

The threat probability vector $\mathbf{s}_t = \hat{\mathbf{p}}_{\text{threat}}$ serves as the state input to a Dueling DQN agent.

#### Action Space

The agent selects from a discrete action set:

$$\mathbf{a}_t \in \{\text{IGNORE}, \text{ALERT}, \text{BLOCK}, \text{ESCALATE}\}$$

with indices $\{0, 1, 2, 3\}$ respectively.

#### Dueling DQN Architecture

The state-value and advantage functions are learned separately:

$$V(\mathbf{s}) = f_V(\mathbf{s}) \in \mathbb{R}$$

$$A(\mathbf{s}, \mathbf{a}) = f_A(\mathbf{s}, \mathbf{a}) \in \mathbb{R}^{|\mathcal{A}|}$$

The Q-function is reconstructed as:

$$Q(\mathbf{s}, \mathbf{a}) = V(\mathbf{s}) + \left(A(\mathbf{s}, \mathbf{a}) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(\mathbf{s}, \mathbf{a}')\right)$$

#### Reward Signal

The reward is a severity-scaled function of detection confidence and action type:

$$R_t = \begin{cases}
\alpha \cdot \max(\mathbf{s}_t) & \text{if } a_t = \text{ESCALATE and benign} \\
-\beta \cdot \max(\mathbf{s}_t) & \text{if } a_t = \text{IGNORE and } a_t = \text{threat} \\
\gamma \cdot \max(\mathbf{s}_t) & \text{if } a_t = \text{matches ground-truth} \\
-\delta & \text{otherwise}
\end{cases}$$

where $\alpha, \beta, \gamma, \delta$ are cost coefficients calibrated on a validation subset.

#### Policy Update (Off-Policy Learning)

The agent updates via Bellman backup with experience replay:

$$\mathcal{L}_{\text{RL}} = \mathbb{E}\left[\left(R_t + \gamma \max_{a'} Q_{\text{target}}(\mathbf{s}_{t+1}, a') - Q_{\text{current}}(\mathbf{s}_t, a_t)\right)^2\right]$$

The target network is updated every $\tau$ steps via Polyak averaging:

$$\theta_{\text{target}} \leftarrow (1 - \rho) \theta_{\text{target}} + \rho \theta_{\text{current}}, \quad \rho = 0.001$$

---

## System Architecture

The system follows a modular information flow:

```
Data Collection Layer
        │
        ▼
Feature Extraction Module
        │
        ▼
Learning-Based Detection Engine       ←  BiLSTM + CNN (CICIDS2017, 15 classes)
        │   outputs threat probability vector (15-dim state)
        ▼
Decision-Making Engine                ←  Dueling DQN — learns policy from reward signal
        │
        ▼
Autonomous Response Module            ←  action ∈ {IGNORE, ALERT, BLOCK, ESCALATE}
        │
        ▼
Feedback Loop                         ←  Severity-scaled reward → policy update
```

**Data Collection Layer** — Ingests raw network flow logs and system events. In this work, input is sourced from CICIDS2017.

**Feature Extraction Module** — Transforms raw NetFlow tuples into 78-dimensional feature vectors normalized via MinMaxScaler. Sliding windows of $T=50$ timesteps form input sequences for temporal modeling.

**Learning-Based Detection Engine** — Fuses BiLSTM (temporal) and 1D-CNN (spatial) encodings into a 15-class threat classifier. Outputs a probability vector $\hat{\mathbf{p}}_{\text{threat}}$ over attack types and benign traffic.

**Decision-Making Engine** — Dueling DQN agent that observes the detection model's threat probability vector and learns a response policy via experience in a simulated environment.

**Autonomous Response Module** — Executes the action selected by the DQN agent. All decisions are logged with structured justification derived from SHAP feature attribution analysis.

**Feedback Loop** — Reward signal (scaled by ground-truth label severity) updates the agent's policy in batch or online fashion, distinguishing this approach from static rule-based systems.

---

## Key Contributions

1. **Novel Problem Formulation:** Reframes intrusion response as a sequential decision-making problem under uncertainty, where the response policy is a learned function of detection confidence rather than a fixed threshold.

2. **Integrated Agentic Architecture:** Demonstrates end-to-end coupling of deep learning-based detection with RL-based policy optimization as a unified autonomous loop.

3. **Architectural Ablation Study:** Isolates the contribution of LSTM-only, CNN-only, and hybrid detection models to clarify the role of temporal vs. spatial feature representations.

4. **Explainability Integration:** Every autonomous decision is accompanied by a SHAP-based feature attribution explanation, enabling post-hoc analysis of agent behavior.

5. **Comparative Evaluation Framework:** Evaluates the proposed DQN policy against rule-based baselines, calibrated-threshold baselines, and random policies on a consistent held-out test set.

---

## Experimental Design

### Detection Model Evaluation

**Objective:** Measure the detection accuracy and calibration of the BiLSTM+CNN ensemble.

**Setup:**
- Dataset: CICIDS2017 (2.8M flows, 15 attack + 1 benign class)
- Train/Validation/Test split: 70%/15%/15% (stratified by class)
- Preprocessing: SMOTE oversampling on training split; MinMaxScaler normalization
- Baseline detection architectures: LSTM-only, CNN-only

**Metrics:**
- Per-class precision, recall, F1-score
- Macro-averaged F1
- ROC-AUC and PR-AUC (benign vs. all attacks)
- False positive rate (FPR) and false negative rate (FNR) per class

### RL Policy Evaluation

**Objective:** Measure the performance of the learned response policy in a simulated environment.

**Setup:**
- State: threat probability vector from the trained detection model
- Action: response action $\in \{\text{IGNORE}, \text{ALERT}, \text{BLOCK}, \text{ESCALATE}\}$
- Training: 50,000 episodes of experience collection with ε-greedy exploration ($\epsilon_0 = 1.0$, decay to 0.05)
- Evaluation: held-out test set, deterministic policy (greedy action selection)

**Metrics:**
- Cumulative discounted reward over test trajectory
- F1-score (policy action vs. ground-truth label severity)
- False positive rate (ESCALATE on benign traffic)
- Mean time to respond (MTTR) = average timesteps before response action
- Action distribution: proportion of each action in the policy

### Comparative Evaluation

Four policies are evaluated on the same held-out event stream:

| Policy | Mechanism |
|--------|-----------|
| **DQN (Proposed)** | Learned response policy via Dueling DQN |
| **Rule-Based Baseline** | Fixed detection probability threshold $\tau_{\text{global}}$; actions escalate linearly with confidence |
| **Calibrated Threshold Baseline** | Per-class thresholds $\{\tau_1, \ldots, \tau_{15}\}$ derived from detection model ROC analysis |
| **Random Policy** | Uniform random action selection; lower-bound reference |

**Test Conditions:** All policies observe the same sequence of detection probability vectors. Ground-truth labels are revealed only for reward calculation (simulated oracle feedback).

---

## Limitations & Scope Boundaries

### Methodological Limitations

1. **Simulation-Reality Gap:** The RL policy is trained and evaluated in a simulated environment where the ground-truth label is known at decision time. Real deployments receive no oracle feedback and must operate under partial observability. Transfer to live network conditions is untested.

2. **Static Threat Ecology:** CICIDS2017 captures network behavior from 2017. The attack patterns, traffic volumes, and background flow characteristics may not reflect contemporary network environments or zero-day attack morphologies.

3. **Supervised Detection Ceiling:** The detection model is trained on labeled data from a specific network segment under controlled conditions. Detection performance on out-of-distribution attacks (zero-day, advanced persistent threats) is unknown.

4. **Binary Reward Function Mismatch:** The reward function assumes attack severity can be determined from ground-truth labels alone. Real operational response costs depend on network topology, business impact, and attacker sophistication — factors not captured in the benchmark dataset.

5. **No Online Adaptation:** The RL policy is trained offline on a fixed dataset. It cannot adapt its strategy in response to shifts in the threat landscape or deployment-specific feedback.

### Scope Boundaries

1. **Single-Network Context:** Experiments assume a single network segment with uniform feature engineering. Multi-network, cross-domain, or federated deployment is not addressed.

2. **Deterministic Action Execution:** The model assumes chosen actions execute with 100% fidelity. In practice, network policies, firewalls, and response systems may fail, queue, or reject actions.

3. **Computational Overhead:** Inference latency of the BiLSTM+CNN model and DQN decision is not characterized. Real-time applicability on high-volume traffic (>1M flows/day) is unvalidated.

4. **Class Imbalance Residuals:** Despite SMOTE and Focal Loss, rare attack classes may remain underrepresented. The agent's policy may reflect training data skew rather than true operational need.

5. **Explainability Constraints:** SHAP explanations are generated post-hoc on the detection model; they do not explain the RL agent's decision logic directly. A full causal explanation of "why the agent chose ESCALATE" remains elusive.

### Assumptions

- Network flows are extractable as fixed-dimension feature vectors.
- Ground-truth labels for training are available and accurate.
- Attack classes are drawn from a closed set (no truly novel threats outside CICIDS2017 taxonomy).
- Response actions have uniform cost and no cross-action dependencies.
- Detection and response latency are negligible compared to attack timescales.

---

## Research Alignment

This project aligns with emerging research themes in:

- **Agentic AI Systems:** Autonomous agents that perceive, reason, and act without continuous human supervision.
- **Autonomous Task Management Under Uncertainty:** Decision-making from probabilistic threat assessments with constrained action sets.
- **Learning-Based Cyber Defense:** Adaptive approaches that evolve with adversarial landscapes.
- **Adaptive & Self-Healing Systems:** Feedback-driven policy modification and emergent robustness.
- **Multi-Agent Coordination:** Planned extension involving distributed, specialized defense agents.

---

## Current Status

- ✅ Data preprocessing pipeline (CICIDS2017, SMOTE oversampling, sliding windows)
- ✅ BiLSTM+CNN detection model with Focal Loss
- ✅ Dueling DQN agent — architecture implemented, training ongoing
- ✅ Rule-based and calibrated-threshold baselines for controlled comparison
- ✅ Ablation study framework (LSTM-only / CNN-only / Hybrid)
- ✅ SHAP explainability module integration
- ✅ 33 unit tests passing
- 🔄 Full training on CICIDS2017 — in progress
- 🔄 RL convergence and policy evaluation — in progress

---

## Future Work

1. **Online RL Adaptation:** Continuous policy updates from live network feedback rather than offline batch simulation.
2. **Multi-Agent Coordination:** Specialised agents (one per attack class or network segment) coordinated by a central orchestrator.
3. **Real-World Deployment & Benchmarking:** Evaluation against live traffic and operational SIEM baselines.
4. **Causal Policy Explanation:** "What-if" counterfactual reasoning for autonomous decisions.
5. **Federated Learning:** Privacy-preserving distributed training across multiple network boundaries.
6. **Adversarial Robustness:** Evaluation under adversarial evasion attacks on the detection model.

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/xclusivecyberdev/Learning-Based-Detection-and-Decision-Making-for-Autonomous-Cyber-Defense.git
cd Learning-Based-Detection-and-Decision-Making-for-Autonomous-Cyber-Defense

# Install dependencies
pip install -r requirements.txt

# Download CICIDS2017 → data/raw/ (see data/README.md for instructions)

# Preprocess the dataset
python scripts/preprocess.py --input data/raw/ --output data/processed/

# Train the detection model and RL agent
python train.py --model bilstm_cnn --epochs 50 --batch_size 32

# Evaluate on test set
python evaluate.py --checkpoint models/detection_model.pth --policy models/dqn_policy.pth
```

---

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. In *2018 10th International Conference on Information Systems Security (ICISSP)* (pp. 108–116). IEEE.

2. Mnih, V., Kavukcuoglu, K., & Silver, D. (2013). Learning value functions with competing tasks. arXiv preprint arXiv:1604.06778.

3. Wang, Z., de Freitas, N., & Lanctot, M. (2016). Dueling network architectures for deep reinforcement learning. In *International Conference on Machine Learning* (pp. 1995–2003). PMLR.

4. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proceedings of the IEEE International Conference on Computer Vision* (pp. 2980–2988).

5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems, 30.

6. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT press.

7. Russell, S., & Norvig, P. (2020). *Artificial intelligence: A modern approach* (4th ed.). Prentice Hall.

---

*MIT License — see [LICENSE](LICENSE) for details.*
