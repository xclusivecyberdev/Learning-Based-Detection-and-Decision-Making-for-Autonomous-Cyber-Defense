# Learning-Based Detection and Decision-Making for Autonomous Cyber Defense

**Awodola Olusola Ebenezer**  
M.Tech Cybersecurity (In View) — Federal University of Technology, Akure  
`oluebenawodola@gmail.com` · [GitHub: @xclusivecyberdev](https://github.com/xclusivecyberdev)

![Status](https://img.shields.io/badge/Status-Ongoing%20Research-blue) ![Python](https://img.shields.io/badge/Python-3.10+-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Abstract

This work proposes and evaluates an integrated framework for autonomous cyber defense that explicitly couples learning-based threat detection with reinforcement learning-based response policy optimization. The core contribution is a novel formulation of cyber response as a **Markov Decision Process** where the agent learns optimal actions directly from probabilistic threat assessments, rather than applying static confidence thresholds.

**Central Research Hypothesis:** Reinforcement learning-based response policies conditioned on detection confidence distributions achieve statistically significant reductions in false positive rates and mean time to respond (MTTR) compared to static rule-based or calibrated-threshold alternatives on the same threat detection substrate.

This work fills a critical gap in the autonomous cyber defense literature: **existing threat detection systems optimize classification accuracy without modeling response costs, while existing cyber response frameworks assume deterministic threat labels.** By formulating the joint detection–response problem as a sequential decision process under uncertainty, we demonstrate that the agent can learn to hedge detection model uncertainty through strategic action selection.

Preliminary validation on CICIDS2017 shows convergence in detection model loss and RL agent reward trajectories, with 33 unit tests passing and full dataset training ongoing.

---

## Related Work & Positioning

### The Detection-Response Gap

Current cyber defense research follows two largely separate trajectories:

**Detection-centric systems** (e.g., AutoML-IDS frameworks, neural IDS models):
- ✅ Optimize detection accuracy, calibration, and latency
- ❌ Assume binary or threshold-based response: *if confidence > τ, alert*
- ❌ Do not model response costs, false positive impact, or operational constraints
- ❌ Cannot learn to mitigate detection model uncertainty through action selection

**Response-centric systems** (e.g., SIEM orchestration, incident response automation):
- ✅ Optimize action sequencing and escalation chains
- ❌ Treat threat classification as deterministic input
- ❌ Rely on manual rule authoring or static policies
- ❌ Do not adapt response strategies to evolving threat distributions

### This Work's Position

This research bridges the gap by proposing that **response policy learning must be conditioned on the detection model's uncertainty distribution**, not its hard classification. Specifically:

1. **Detection-Aware Policy:** The RL agent observes the full 15-dimensional threat probability vector $\hat{\mathbf{p}}_{\text{threat}}$, not just $\arg\max(\hat{\mathbf{p}}_{\text{threat}})$.

2. **Explicit Uncertainty Modeling:** Actions like IGNORE or ALERT are not mere thresholds, but learned hedges against detection uncertainty. An IGNORE action on a borderline detection (entropy near max) differs strategically from IGNORE on a high-confidence benign classification.

3. **End-to-End Optimization:** The detection and response models are not independently optimized. The response policy learns which types of detection errors matter most operationally, creating feedback pressure for the detection model to specialize.

### What's Novel Here

Among the literature in autonomous cyber defense, threat detection with machine learning, and RL-based decision-making:

- **Few works explicitly learn response policies from detection confidence distributions.** Most assume detected threats are given and model response sequencing.
- **Existing cyber RL work** (e.g., autonomous honeypots, defensive deception) does not condition policy on probabilistic threat assessments; they operate in fully-observable or adversarial game-theoretic settings.
- **AutoML-based IDS frameworks** optimize detection pipelines end-to-end but do not model response or operational cost.

**This work is among the first to explicitly formulate the joint detection–response problem as a single Markov Decision Process where the agent learns to map probabilistic threat outputs to strategic response actions, with structured explainability.**

---

## Research Problem & Theoretical Framing

### Problem Formulation

We formulate cyber defense as a **sequential decision process under uncertainty**:

$$\mathcal{M} = (\mathcal{S}, \mathcal{A}, p(s' | s, a), R(s, a), \gamma, \rho_0)$$

where:
- $\mathcal{S} = \mathbb{R}^{15}$ is the state space (threat probability vectors from detection model)
- $\mathcal{A} = \{\text{IGNORE}, \text{ALERT}, \text{BLOCK}, \text{ESCALATE}\}$ is the action space
- $p(s' | s, a)$ models state transitions (threat events and their progression)
- $R(s, a) \in \mathbb{R}$ is a severity-scaled reward function
- $\gamma \in (0, 1)$ is the discount factor
- $\rho_0$ is the initial state distribution (network traffic)

The goal is to learn an optimal policy $\pi^* : \mathcal{S} \to \mathcal{A}$ that maximizes expected cumulative discounted reward:

$$J(\pi) = \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t R(s_t, \pi(s_t)) \bigg| s_0 \sim \rho_0\right]$$

### Gap in Existing Approaches

**Static Threshold Policies** treat detection output as a scalar confidence $c = \max(\hat{\mathbf{p}}_{\text{threat}})$ and apply a fixed rule:

$$a_{\text{static}} = \begin{cases}
\text{ALERT} & \text{if } c > \tau \\
\text{IGNORE} & \text{otherwise}
\end{cases}$$

**Limitations:**
1. Single threshold ignores the shape of the probability distribution (e.g., a uniform distribution over 5 attack classes has high entropy but may trigger the same response as a confident single-class prediction).
2. No adaptation to operational context: $\tau$ is fixed globally, not conditioned on network state, system load, or recent false positive rates.
3. Detection errors propagate directly: if the model is 70% accurate, 30% of responses are misdirected, and the static policy cannot learn to hedge.

**This work's approach:** Learn $\pi^*(s) = \pi^*({\mathbf{p}})$ such that the agent exploits the full probability distribution to make decisions robust to detection uncertainty.

---

## Mathematical Formulation

### Detection Model: Learning Threat Probability Distributions

The detection module models cyber traffic as temporal sequences of feature vectors and outputs a calibrated threat probability distribution (not a hard classification):

$$\mathbf{X} = \{x_1, x_2, \ldots, x_T\} \in \mathbb{R}^{T \times 78}$$

where $T = 50$ timesteps and each $x_t$ is an 78-dimensional normalized flow feature vector.

#### Temporal Encoding (BiLSTM)

Bidirectional LSTM captures temporal dependencies in attack progression:

$$\overrightarrow{\mathbf{h}}_t = \text{LSTM}_{\text{fwd}}(\mathbf{x}_t, \overrightarrow{\mathbf{h}}_{t-1})$$

$$\overleftarrow{\mathbf{h}}_t = \text{LSTM}_{\text{bwd}}(\mathbf{x}_t, \overleftarrow{\mathbf{h}}_{t-1})$$

$$\mathbf{h}_t^{\text{LSTM}} = [\overrightarrow{\mathbf{h}}_t \oplus \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{d_h}$$

Final hidden state: $\mathbf{h}_T^{\text{LSTM}} \in \mathbb{R}^{256}$ (concatenated forward + backward).

#### Spatial Encoding (1D-CNN)

Parallel convolution filters detect local attack signatures and feature co-occurrences:

$$\mathbf{c}_k = \text{ReLU}\left(\mathbf{W}_k * \mathbf{X} + \mathbf{b}_k\right)_{j} = \max\left(0, \sum_{i=1}^{78} W_{k,i} X_{i, j:j+K-1} + b_k\right)$$

$$\mathbf{h}^{\text{CNN}} = \text{GlobalMaxPool}\left([\mathbf{c}_1, \ldots, \mathbf{c}_{K}]\right) \in \mathbb{R}^{64}$$

where $K = 3$ is the kernel size (capturing 3-timestep patterns).

#### Fusion & Probability Calibration

Representations are concatenated and projected through dense layers with dropout:

$$\mathbf{h}_{\text{fused}} = \text{ReLU}(\mathbf{W}_1 [\mathbf{h}^{\text{LSTM}} \oplus \mathbf{h}^{\text{CNN}}] + \mathbf{b}_1) \in \mathbb{R}^{128}$$

$$\mathbf{h}_{\text{fused}} = \text{Dropout}(\mathbf{h}_{\text{fused}}, p=0.3)$$

Final softmax output (calibrated probability distribution over 15 attack classes + benign):

$$\hat{\mathbf{p}}_{\text{threat}} = \text{softmax}\left(\mathbf{W}_{\text{out}} \mathbf{h}_{\text{fused}} + \mathbf{b}_{\text{out}}\right) \in \Delta^{15}$$

where $\Delta^{15}$ is the 15-dimensional probability simplex.

#### Training with Focal Loss

To handle class imbalance (benign >> rare attacks), we use Focal Loss:

$$\mathcal{L}_{\text{detection}} = -\sum_{i=1}^{15} (1 - \hat{p}_i)^{\gamma} y_i \log(\hat{p}_i)$$

with $\gamma = 2$ (focusing parameter). This down-weights easy negatives and emphasizes hard-to-classify samples, forcing the model to learn confident, well-calibrated probability estimates.

---

### Response Policy: Learning from Detection Uncertainty

The threat probability vector $\mathbf{s}_t = \hat{\mathbf{p}}_{\text{threat}}$ becomes the state for a Dueling DQN agent:

#### Markov Decision Process Setup

- **State space:** $\mathcal{S} = \{\hat{\mathbf{p}}_{\text{threat}} : \mathbf{p} \in \Delta^{15}\}$
- **Action space:** $\mathcal{A} = \{0, 1, 2, 3\}$ representing $\{\text{IGNORE}, \text{ALERT}, \text{BLOCK}, \text{ESCALATE}\}$
- **State transitions:** Deterministic in simulation; $s_{t+1} = f_{\text{detect}}(\text{next traffic chunk})$
- **Reward:** Severity-scaled, reflecting whether action matches ground-truth threat level

#### Dueling DQN Architecture

The value and advantage functions are learned separately to stabilize training:

$$V(\mathbf{s}) = \mathbf{w}_V^T \phi_V(\mathbf{s})$$

$$A(\mathbf{s}, a) = \mathbf{w}_A^T \phi_A(\mathbf{s}, a)$$

where $\phi_V, \phi_A$ are learned feature representations. The Q-function is reconstructed to prevent overestimation:

$$Q(\mathbf{s}, a) = V(\mathbf{s}) + \left(A(\mathbf{s}, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(\mathbf{s}, a')\right)$$

This decomposition allows the agent to learn state values independent of action advantages, improving learning stability.

#### Reward Function (Severity-Scaled)

The reward balances three objectives: detection sensitivity, false positive cost, and response latency:

$$R(s_t, a_t) = \begin{cases}
+10 & \text{if } y_t = \text{attack} \land a_t \in \{\text{BLOCK}, \text{ESCALATE}\} \land \max(s_t) > 0.7 \\
-5 & \text{if } y_t = \text{benign} \land a_t \in \{\text{BLOCK}, \text{ESCALATE}\} \quad \text{(false positive)} \\
+2 & \text{if } y_t = \text{attack} \land a_t = \text{ALERT} \\
-8 & \text{if } y_t = \text{attack} \land a_t = \text{IGNORE} \quad \text{(missed attack)} \\
+1 & \text{if } y_t = \text{benign} \land a_t = \text{IGNORE} \\
-1 & \text{otherwise}
\end{cases}$$

The agent thus learns: *aggressive response (ESCALATE) is rewarded on true attacks but penalized on false positives. Calibrating this tradeoff through learned action selection is the core contribution.*

#### Policy Update: Off-Policy Learning with Experience Replay

The agent updates using the Bellman equation with a target network for stability:

$$\mathcal{L}_{\text{RL}}(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{B}}\left[\left(r + \gamma \max_{a'} Q_{\text{target}}(s', a'; \theta^-) - Q(s, a; \theta)\right)^2\right]$$

where $\mathcal{B}$ is a replay buffer of (state, action, reward, next-state) tuples, and $\theta^-$ are target network parameters.

The target network is updated via Polyak averaging every $\tau$ steps:

$$\theta^- \leftarrow (1 - \rho) \theta^- + \rho \theta, \quad \rho = 0.001, \quad \tau = 1000$$

This slow update reduces correlation between bootstrapped Q-values and ground-truth targets.

---

## System Architecture

The integrated framework follows a unified information flow:

```
Network Traffic
        │
        ▼
Sliding Window Buffer (T=50 timesteps)
        │
        ▼
Feature Extraction (78-dim → normalized)
        │
        ▼
┌──────────────────────────────────────────┐
│   Detection Model: BiLSTM + 1D-CNN       │
│   Output: Threat Probability Vector      │
│   p̂_threat ∈ ℝ^15  (calibrated)         │
└──────────────────────────────────────────┘
        │  (state s_t)
        ▼
┌──────────────────────────────────────────┐
│   Decision-Making Engine: Dueling DQN   │
│   Input: Threat Probability Vector       │
│   Output: Action a_t ∈ {0,1,2,3}        │
└──────────────────────────────────────────┘
        │
        ▼
Autonomous Response Module
  - IGNORE: log and monitor
  - ALERT: notify security team
  - BLOCK: network-level isolation
  - ESCALATE: incident escalation + forensics
        │
        ▼
Oracle Feedback (in simulation)
  - Ground-truth label y_t
  - Reward signal R(s_t, a_t, y_t)
        │
        ▼
Policy Update (off-policy RL)
  - Buffer experience in replay memory
  - Batch Bellman update every N steps
  - Target network Polyak averaging
```

**Key Insight:** The detection and response models form a **closed information loop**. Detection uncertainty (entropy in $\hat{\mathbf{p}}_{\text{threat}}$) directly affects action selection, and action consequences (reward signal) provide feedback that optimizes both the policy AND informs detection model interpretability.

---

## Key Contributions

1. **Novel Problem Formulation:** First to explicitly formulate joint detection–response as a single Markov Decision Process where response policies are learned as functions of probabilistic threat assessments. This fundamentally differs from pipelined approaches that optimize detection and response separately.

2. **Explicit Uncertainty Modeling:** The response agent observes full threat probability distributions, not hard classifications. This allows the agent to learn principled actions under epistemic uncertainty (e.g., ALERT on high-entropy detections).

3. **Integrated Agentic Architecture:** Demonstrates end-to-end tight coupling of deep learning-based detection with RL-based policy optimization as a single closed-loop autonomous system, rather than loosely coupled modules.

4. **Architectural Ablation Study:** Isolates the contribution of LSTM-only, CNN-only, and hybrid detection models. This characterizes whether threat detection benefits more from temporal or spatial feature representations—a question often left unanswered in neural IDS literature.

5. **Structured Explainability:** Every autonomous decision is accompanied by SHAP-based feature attribution explaining the detection model's confidence, enabling post-hoc analysis of policy behavior and detection model correctness.

6. **Comparative Evaluation:** Evaluates the learned policy against three baselines (rule-based, calibrated-threshold, random) on a consistent test set, with transparent metrics for false positive rates, MTTR, and reward accumulation.

---

## Experimental Design

### Detection Model Evaluation

**Objective:** Measure how well the BiLSTM+CNN ensemble learns well-calibrated threat probability distributions.

**Setup:**
- Dataset: CICIDS2017 (2.8M flows, 15 attack types + benign)
- Train/Validation/Test split: 70%/15%/15% (stratified by class)
- Preprocessing: SMOTE oversampling on training split; MinMaxScaler normalization
- Baselines: LSTM-only, CNN-only, LSTM+CNN with late fusion

**Primary Metrics:**
- **Macro F1-score** (primary): Balances precision/recall across imbalanced classes
- **ROC-AUC** (benign vs. all attacks): Measures discrimination ability
- **ECE (Expected Calibration Error)**: Measures confidence calibration; lower is better
- **Per-class FPR & FNR:** Identifies which attack types the model struggles with

### RL Policy Evaluation

**Objective:** Measure the learned response policy's ability to balance true positive rates and false positive costs.

**Setup:**
- State input: threat probability vector from trained detection model
- Training: 50,000 episodes with ε-greedy exploration ($\epsilon$ decayed from 1.0 to 0.05 over 40k episodes)
- Evaluation: deterministic greedy policy on held-out test set
- Oracle feedback: ground-truth labels reveal true attack/benign status (simulated oracle)

**Primary Metrics:**
- **Cumulative discounted reward:** Higher is better; measures value of learned policy
- **F1-score (policy vs. severity):** Treats action selection as classification (IGNORE=negative, ALERT=borderline, ESCALATE=positive)
- **False Positive Rate (FPR):** Percentage of benign traffic where policy selects BLOCK or ESCALATE
- **Mean Time To Respond (MTTR):** Average timesteps before policy takes action (lower is better)
- **Action distribution:** Proportion of each action; reveals learned strategy

### Comparative Evaluation

Four policies evaluated on the same test event stream:

| Policy | Mechanism | Adaptation |
|--------|-----------|-----------|
| **DQN (Proposed)** | Learned mapping: $\hat{\mathbf{p}}_{\text{threat}} \to a$ via Dueling DQN | Full adaptation to probability distribution |
| **Rule-Based Baseline** | If $\max(\hat{\mathbf{p}}_{\text{threat}}) > \tau_{\text{global}}$, escalate; else ignore. $\tau$ fixed globally. | No adaptation; static threshold |
| **Calibrated Threshold Baseline** | Per-class thresholds $\{\tau_1, \ldots, \tau_{15}\}$ derived from ROC analysis on validation set | Adapted per attack class, but static otherwise |
| **Random Policy** | Uniform random action selection | No adaptation; theoretical lower bound |

**Test Conditions:** All policies observe identical detection probability vectors. Ground-truth labels are revealed only for reward calculation; policies operate with no oracle feedback during action selection.

---

## Limitations & Scope Boundaries

### Methodological Limitations

1. **Simulation-Reality Gap (Critical):** The RL policy is trained and evaluated in a simulated environment where the ground-truth attack label is known at decision time. Real deployments operate under partial observability: the true label is unknown until post-incident analysis or operational feedback arrives much later. **Transferability to live networks is unvalidated.**

2. **Static Threat Ecology:** CICIDS2017 captures network behavior from 2017. Contemporary attacks (ransomware chains, supply-chain compromises, zero-days) follow different patterns. **Generalization to 2024+ threat landscapes is unknown.**

3. **Supervised Detection Ceiling:** The detection model is trained on labeled data from a specific network segment under controlled lab conditions. **Performance on out-of-distribution or zero-day attacks with no training analogues is likely poor.**

4. **Reward Function Mismatch:** The reward function assumes attack severity is known from labels alone. **Real operational costs depend on network topology, business impact, attacker sophistication, and cascading failures—none captured in the benchmark.**

5. **No Online Adaptation:** The RL policy is trained offline on a fixed dataset. **It cannot adapt in response to drift in the threat landscape, deployment-specific feedback, or concept drift in traffic patterns.**

6. **Single-Agent Assumption:** The work assumes a single centralized agent making decisions. **Multi-agent, hierarchical, or distributed response architectures are not explored.**

### Scope Boundaries

1. **Single-Network Context:** Experiments assume a monolithic network segment with uniform feature engineering. **Multi-network, cross-domain, federated, or edge-based deployment is not addressed.**

2. **Deterministic Action Execution:** The model assumes chosen actions execute with 100% fidelity. **Real network policies, firewalls, and response systems can fail, queue, or reject actions; policy robustness to action failure is unknown.**

3. **Computational Overhead:** Inference latency of the BiLSTM+CNN model and DQN decision is not characterized. **Real-time applicability on high-volume traffic (>1M flows/second) or mobile edge devices is unvalidated.**

4. **Class Imbalance Residuals:** Despite SMOTE and Focal Loss, rare attack classes (e.g., Infiltration, 3.5% of data) remain underrepresented. **The agent's policy may reflect training data skew rather than true operational prioritization.**

5. **Explainability Gaps:** SHAP explanations are generated post-hoc on the detection model only. **The RL agent's decision logic—*why* it selected ESCALATE over ALERT—remains a black box. Causal counterfactual explanations ("what would change the decision?") are not provided.**

### Explicit Assumptions

- Network flows are extractable as fixed-dimension feature vectors; flow reconstruction is perfect.
- Ground-truth labels for training are accurate and representative of deployment conditions.
- Attack classes form a closed taxonomy (no entirely novel threats outside CICIDS2017 classes).
- Response actions have **independent costs** and no cascading side-effects (e.g., BLOCK does not propagate across network layers).
- Detection and response latency are negligible compared to attack dwell times.
- Reward signal is known immediately after each action (oracle feedback).

---

## Research Alignment

This work contributes to emerging research in:

- **Agentic AI & Autonomous Systems:** Agents that perceive, reason, and act without continuous human oversight.
- **Sequential Decision-Making Under Uncertainty:** Formalizing cyber defense as a Markov Decision Process rather than a rule engine.
- **Learning-Based Cyber Defense:** Adaptive security that evolves with adversarial landscapes, not static signatures.
- **Intelligent Explainability:** Coupling deep learning interpretability (SHAP) with reinforcement learning policy analysis.

---

## Current Status

- ✅ Data preprocessing pipeline (CICIDS2017, SMOTE, sliding windows, normalization)
- ✅ BiLSTM+CNN detection model with Focal Loss training
- ✅ Dueling DQN agent architecture and training loop
- ✅ Rule-based and calibrated-threshold baselines implemented
- ✅ Ablation study framework (LSTM-only / CNN-only / Hybrid)
- ✅ SHAP explainability module integrated
- ✅ 33 unit tests passing
- 🔄 Full-scale training on CICIDS2017 — in progress
- 🔄 RL convergence and policy evaluation — in progress
- 🔄 Detection model calibration analysis — in progress

---

## Future Work

1. **Online RL Adaptation:** Extend the offline framework to support continuous policy updates from live network feedback, enabling the agent to adapt to threat drift.

2. **Multi-Agent Cyber Defense:** Specialised agents (one per attack class, network segment, or defense layer) coordinated by a central orchestrator or hierarchical policy.

3. **Real-World Deployment & Transfer Learning:** Evaluate against live traffic and operational SIEM baselines; study domain adaptation from CICIDS2017 to live networks.

4. **Causal Policy Explanations:** Implement counterfactual reasoning ("what change in detection output would flip the decision?") for full transparency.

5. **Federated Learning:** Privacy-preserving distributed training where multiple organizations train a shared policy without sharing raw network data.

6. **Adversarial Robustness:** Evaluate policy and detection model under adversarial evasion attacks; study robustness-accuracy tradeoffs.

7. **Action-Level Constraints:** Model hard constraints (e.g., "never BLOCK without ALERT"; rate limits on escalation) as part of the MDP formulation.

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/xclusivecyberdev/Learning-Based-Detection-and-Decision-Making-for-Autonomous-Cyber-Defense.git
cd Learning-Based-Detection-and-Decision-Making-for-Autonomous-Cyber-Defense

# Install dependencies
pip install -r requirements.txt

# Download CICIDS2017 dataset
# See data/README.md for download instructions

# Preprocess the dataset
python scripts/preprocess.py --input data/raw/ --output data/processed/

# Train the detection model and RL agent
python train.py --model bilstm_cnn --epochs 50 --batch_size 32

# Evaluate on test set
python evaluate.py --checkpoint models/detection_model.pth --policy models/dqn_policy.pth

# Visualize policy and SHAP explanations
python visualize.py --policy models/dqn_policy.pth --data data/processed/X_test.npy
```

---

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. In *Proc. 10th International Conference on Information Systems Security (ICISSP)* (pp. 108–116). IEEE.

2. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement learning: An introduction* (2nd ed.). MIT Press.

3. Mnih, V., Kavukcuoglu, K., & Silver, D. (2013). Playing Atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.

4. Wang, Z., de Freitas, N., & Lanctot, M. (2016). Dueling network architectures for deep reinforcement learning. In *International Conference on Machine Learning* (pp. 1995–2003). PMLR.

5. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. In *Proc. IEEE International Conference on Computer Vision* (pp. 2980–2988).

6. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

7. Russell, S., & Norvig, P. (2020). *Artificial intelligence: A modern approach* (4th ed.). Prentice Hall.

8. Goodfellow, I., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. In *International Conference on Learning Representations (ICLR)*.

9. Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. In *IEEE Symposium on Security and Privacy (SP)* (pp. 39–57). IEEE.

---

*MIT License — see [LICENSE](LICENSE) for details.*
