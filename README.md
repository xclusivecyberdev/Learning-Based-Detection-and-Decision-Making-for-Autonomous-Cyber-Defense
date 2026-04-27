# Learning-Based Detection and Decision-Making for Autonomous Cyber Defense

**Awodola Olusola Ebenezer**
M.Tech Cybersecurity (In View) — Federal University of Technology, Akure
`oluebenawodola@gmail.com` · [GitHub: @xclusivecyberdev](https://github.com/xclusivecyberdev)

![Status](https://img.shields.io/badge/Status-Ongoing%20Research-blue) ![Python](https://img.shields.io/badge/Python-3.10+-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Abstract

This project explores the design of an intelligent cybersecurity system capable of integrating learning-based threat detection with adaptive decision-making mechanisms. The system aims to move beyond static, rule-based defenses by introducing agentic behaviour — where a system can perceive, reason, and act autonomously in response to evolving cyber threats.

**This work reframes cyber defense as a sequential decision-making problem, where response policies are learned directly from detection uncertainty distributions rather than manually defined thresholds.** The system models cyber defense as an autonomous task management problem, where the agent must prioritise, defer, or escalate actions under uncertainty and system constraints — framing the problem in terms directly aligned with agentic AI research.

**Central Hypothesis:** *This work hypothesises that reinforcement learning-based response policies can reduce false positive rates and response latency compared to static rule-based intrusion response systems, because a learned policy adapts its action selection to the confidence distribution of the detection model rather than applying fixed thresholds uniformly across all threat types.*

Preliminary experiments on a stratified subset of the dataset show consistent validation loss reduction in the detection model and expected reward improvement trends in the RL agent, indicating stable training dynamics.

This work contributes toward the development of autonomous cyber defense agents aligned with emerging research in agentic AI and self-adaptive systems.

---

## Research Motivation

Traditional cybersecurity systems rely heavily on predefined rules and human intervention, making them less effective against rapidly evolving and unknown threats. Recent advances in artificial intelligence suggest the need for systems that can dynamically adapt, learn from data, and make context-aware decisions.

This project investigates how learning-based detection models can be coupled with adaptive response policies to enable autonomous, real-time cyber defense. The core research question is:

> *Does replacing a static threshold response policy with a reinforcement learning-based decision agent produce measurably lower false positive rates and reduced mean time to respond on the same detection model outputs?*

---

## System Architecture

The system is designed as a modular intelligent pipeline:

```
Data Collection Layer
        │
        ▼
Feature Extraction Module
        │
        ▼
Learning-Based Detection Engine       ←  BiLSTM + CNN (CICIDS2017, 15 classes)
        │   outputs threat probability vector (18-dim state)
        ▼
Decision-Making Engine                ←  Dueling DQN — learns policy from reward signal
        │
        ▼
Autonomous Response Module            ←  IGNORE | ALERT | BLOCK | ESCALATE
        │
        ▼
Feedback Loop                         ←  Reward (severity-scaled) → policy update
```

**Data Collection Layer** — captures network flow logs, system events, and traffic behaviour.

**Feature Extraction Module** — transforms raw flows into 78-dimensional normalised feature vectors; sliding windows of 50 timesteps are constructed for sequential modelling.

**Learning-Based Detection Engine** — a BiLSTM encoder captures temporal dependencies in flow sequences; a 1D-CNN extracts spatial feature co-occurrences. Both representations are fused into a 15-class threat classifier trained with Focal Loss to address class imbalance.

**Decision-Making Engine** — a Dueling DQN agent observes the detection model's output probability vector as its state, and learns a response policy through experience in a reward-shaped simulated environment. Crucially, the policy is *learned* — not coded — which allows it to adapt its behaviour to the detector's confidence distribution across threat types.

**Autonomous Response Module** — executes the chosen action and logs a structured, human-readable justification for every decision.

**Feedback Loop** — a severity-scaled reward signal updates the agent's policy continuously, distinguishing the system from static alternatives.

---

## Key Contributions

- **Novel framing**: Reframes intrusion response as a sequential decision-making problem under uncertainty, where the action policy is a learned function of detection confidence — not a fixed threshold
- Integration of deep learning-based detection with a reinforcement learning response policy as a unified agentic loop
- Ablation study comparing LSTM-only, CNN-only, and hybrid detection architectures to isolate each component's contribution
- Structured explainability: every autonomous decision is accompanied by a traceable, natural-language rationale derived from SHAP feature attribution
- Real-time threat simulation using the CICIDS2017 benchmark (2.8M flow records, 15 attack classes)

---

## Proposed Experiment

**Detection model evaluation** — BiLSTM+CNN trained on CICIDS2017 (70/15/15 stratified split). Metrics: accuracy, macro F1, ROC-AUC, false positive rate per class. Ablation variants: LSTM-only, CNN-only, Hybrid.

**Agent policy evaluation** — the trained detection model generates threat probability vectors fed to the DQN agent as state. Metrics: F1, false positive rate, mean time to respond (MTTR), cumulative episode reward.

**Comparative evaluation** — four policies evaluated on the same held-out event stream:

| Policy | Description |
|--------|-------------|
| DQN (proposed) | Learned response policy |
| Rule-based baseline | Fixed confidence thresholds |
| Calibrated threshold baseline | Thresholds derived from supervised probability outputs of the detection model |
| Random policy | Lower-bound reference |

Results will be reported here as experiments complete on the full dataset.

---

## Critical Analysis

While the system introduces a promising framework for autonomous cyber defense, several limitations warrant acknowledgement.

The key limitation is that the response policy is learned in a simulated environment rather than a live operational setting, which may affect real-world transferability. The reward function, while carefully designed, approximates the cost structure of real SOC decisions — discrepancies between simulated and real reward signals could produce policies that do not generalise to production environments.

The reliance on supervised detection models may constrain adaptability against zero-day attacks with no training analogues. The CICIDS2017 dataset reflects 2017-era network conditions; evaluating on more recent benchmarks is a necessary next step.

Future work should address online adaptation — allowing the RL policy to update from live feedback — and explore multi-agent coordination models where specialised agents collaborate across network, endpoint, and identity domains. Incorporating counterfactual explainability would further enhance trust in autonomous decision-making in high-stakes operational contexts.

---

## Research Alignment

This project aligns with ongoing research in:

- **Agentic AI systems** — autonomous agents that perceive, reason, and act without continuous human direction
- **Autonomous task management under uncertainty** — the agent must prioritise, defer, or escalate actions given probabilistic threat assessments
- **Intelligent cyber defense** — learning-based approaches that adapt to adversarial evolution
- **Adaptive and self-healing systems** — feedback-driven behaviour modification
- **Multi-agent coordination** — a planned extension of this architecture

---

## Current Status

- ✅ Data preprocessing pipeline (CICIDS2017, SMOTE, sliding windows)
- ✅ BiLSTM+CNN detection model with Focal Loss
- ✅ Dueling DQN agent — initial implementation complete, training ongoing
- ✅ Rule-based and calibrated threshold baselines for controlled comparison
- ✅ Ablation study framework (LSTM-only / CNN-only / Hybrid)
- ✅ SHAP explainability integration
- ✅ 33 unit tests passing
- 🔄 Full training on CICIDS2017 — in progress
- 🔄 RL convergence experiments — in progress

**Feedback, review, and collaboration are welcomed.** Reach out at `oluebenawodola@gmail.com`.

---

## Future Work

- **Online RL adaptation** — continuous policy updates from live feedback rather than batch simulation
- **Multi-agent cyber defense architecture** — specialised agents coordinated by a central orchestrator
- **Real-world deployment and benchmarking** — evaluation against live traffic and operational SIEM baselines
- **Counterfactual explainability** — "what would have changed the decision" reasoning for each autonomous action
- **Federated learning** — privacy-preserving training across distributed network nodes

---

## Quick Start

```bash
git clone https://github.com/xclusivecyberdev/agentic-threat-detection.git
cd agentic-threat-detection
pip install -r requirements.txt

# Download CICIDS2017 → data/raw/  (see data/README.md)
python scripts/preprocess.py --input data/raw/ --output data/processed/
python train.py
python run_agent.py --mode simulate --data data/sample/sample_traffic.csv
```

---

## References

1. Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward generating a new intrusion detection dataset and intrusion traffic characterization. *ICISSP 2018*.
2. Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*, 529–533.
3. Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML 2016*.
4. Lin, T.-Y., et al. (2017). Focal loss for dense object detection. *ICCV 2017*.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017*.
6. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). MIT Press.
7. AgenticCyber (2024). GenAI-powered multi-agent system for multimodal threat detection. *arXiv:2512.06396*.

---

*MIT License — see [LICENSE](LICENSE) for details.*
