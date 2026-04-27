# Learning-Based Detection and Decision-Making for Autonomous Cyber Defense

**Awodola Olusola Ebenezer**
M.Tech Cybersecurity (In View) — Federal University of Technology, Akure
`oluebenawodola@gmail.com` · [GitHub: @xclusivecyberdev](https://github.com/xclusivecyberdev)

![Status](https://img.shields.io/badge/Status-Ongoing%20Research-blue) ![Python](https://img.shields.io/badge/Python-3.10+-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

---

## Abstract

This project explores the design of an intelligent cybersecurity system capable of integrating learning-based threat detection with adaptive decision-making mechanisms. The system aims to move beyond static, rule-based defenses by introducing agentic behaviour — where a system can perceive, reason, and act autonomously in response to evolving cyber threats.

**Central Hypothesis:** *Reinforcement learning-based response policies can outperform static rule-based approaches in reducing false positives and improving response latency in cyber threat mitigation, because a learned policy adapts continuously to the confidence distribution of the detection model — whereas fixed threshold rules cannot.*

Preliminary experiments indicate stable convergence behaviour of the reinforcement learning agent in simulated environments, and consistent validation loss reduction in the detection model across early training epochs — providing initial evidence that the coupled architecture behaves as theoretically expected.

This work contributes toward the development of autonomous cyber defense agents aligned with emerging research in agentic AI and self-adaptive systems.

---

## Research Motivation

Traditional cybersecurity systems rely heavily on predefined rules and human intervention, making them less effective against rapidly evolving and unknown threats. Recent advances in artificial intelligence suggest the need for systems that can dynamically adapt, learn from data, and make context-aware decisions.

This project investigates how learning-based models can be integrated with decision-making frameworks to enable autonomous, real-time cyber defense capabilities. The core research question is:

> *Does integrating a deep learning-based anomaly detector with a reinforcement learning response agent produce measurably superior outcomes — specifically lower false positive rates and reduced mean time to respond — compared to static threshold policies?*

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
Learning-Based Detection Engine       ←  BiLSTM + CNN (CICIDS2017)
        │
        ▼
Decision-Making Engine                ←  Dueling DQN Agent
        │
        ▼
Autonomous Response Module            ←  IGNORE | ALERT | BLOCK | ESCALATE
        │
        ▼
Feedback Loop                         ←  Reward signal → policy update
```

**Data Collection Layer** — captures network flow logs, system events, and traffic behaviour.

**Feature Extraction Module** — transforms raw flows into 78-dimensional normalised feature vectors using the CICIDS2017 schema; sliding windows of 50 timesteps are created for sequential modelling.

**Learning-Based Detection Engine** — a BiLSTM encoder captures temporal dependencies in flow sequences, while a 1D-CNN extracts spatial feature co-occurrences. The two representations are fused into a 15-class threat classifier trained with Focal Loss to address class imbalance.

**Decision-Making Engine** — a Dueling DQN agent observes the detection model's output probability vector alongside system context features, and learns an optimal response policy through interaction with a simulated cyber defense environment.

**Autonomous Response Module** — executes the chosen action (ignore, alert, firewall block, or escalate to human analyst) and logs a structured, human-readable justification for every decision.

**Feedback Loop** — the reward signal (shaped by threat severity and action correctness) continuously updates the agent's policy, enabling adaptation over time.

---

## Key Contributions

- Integration of learning-based detection with a reinforcement learning decision-making policy
- Prototype design for an autonomous cyber defense agent following the Perceive → Reason → Act → Reflect cycle
- Exploration of agentic AI principles applied to cybersecurity — treating response as a *learned* behaviour rather than a hard-coded ruleset
- Ablation study comparing LSTM-only, CNN-only, and hybrid detection architectures
- Structured explainability layer: every autonomous decision is accompanied by a traceable, natural-language rationale
- Real-time threat analysis and response simulation using the CICIDS2017 benchmark

---

## Proposed Experiment

To evaluate system effectiveness:

**Detection model evaluation** — train BiLSTM+CNN on CICIDS2017 (70% train / 15% val / 15% test, stratified split). Measure accuracy, macro F1, ROC-AUC, and false positive rate per class. Compare against LSTM-only and CNN-only ablation variants.

**Agent policy evaluation** — using the trained detection model to generate threat probability vectors, train the Dueling DQN agent on a simulated event stream. Measure:
- Detection F1 and false positive rate
- Mean Time to Respond (MTTR)
- Cumulative episode reward

**Comparison** — evaluate three policies on the same held-out event stream:

| Policy | Description |
|--------|-------------|
| DQN (proposed) | Learned response policy |
| Rule-based baseline | Fixed confidence thresholds |
| Random policy | Lower bound reference |

Results will be reported here as experiments complete on the full dataset.

---

## Critical Analysis

While the system introduces a promising framework for autonomous cyber defense, several limitations remain. The reliance on supervised learning models may constrain adaptability in highly dynamic threat environments, particularly when encountering zero-day attacks. Furthermore, the current decision-making mechanism — though learned rather than rule-based — is trained on a single dataset and may not generalise without retraining.

The CICIDS2017 dataset, while a widely used benchmark, reflects 2017-era network conditions. Evaluating generalisability on more recent datasets (e.g., CICIDS2023) is a necessary next step.

Future work should deepen the reinforcement learning component to enable continuous online adaptation, explore multi-agent coordination models for collaborative defense strategies, and incorporate explainable AI techniques to enhance transparency and trust in autonomous decision-making.

---

## Research Alignment

This project aligns with ongoing research in:

- **Agentic AI systems** — autonomous agents that perceive, reason, and act without continuous human direction
- **Autonomous task execution** — closing the loop from detection to response without manual triage
- **Intelligent cyber defense** — learning-based approaches that adapt to adversarial evolution
- **Adaptive and self-healing systems** — feedback-driven behaviour modification
- **Multi-agent coordination** — a direction for future extension of this work

---

## Current Status

This project is in active development. All core components are implemented and tested:

- ✅ Data preprocessing pipeline (CICIDS2017, SMOTE, sliding windows)
- ✅ BiLSTM+CNN detection model with Focal Loss
- ✅ Dueling DQN agent — initial implementation complete, training ongoing
- ✅ Rule-based baseline policy for controlled comparison
- ✅ Ablation study framework
- ✅ SHAP explainability integration
- ✅ 33 unit tests passing
- 🔄 Full training on CICIDS2017 — in progress
- 🔄 RL convergence experiments — in progress

**Feedback, review, and collaboration are welcomed.** If you are a researcher working in agentic AI, autonomous systems, or cyber defense, I would be glad to discuss the methodology or share preliminary results. Please reach out at `oluebenawodola@gmail.com`.

---

## Future Work

- **Reinforcement learning-based adaptive defense** — extend the DQN agent toward online learning with continuous policy updates as new threats are observed
- **Multi-agent cyber defense architecture** — specialised agents (network, endpoint, identity) coordinated by a central orchestrator
- **Real-world deployment and benchmarking** — evaluation on live network traffic and comparison against operational SIEM systems
- **Explainable AI for decision transparency** — counterfactual explanations and confidence-calibrated justifications for every autonomous action
- **Federated learning** — privacy-preserving training across distributed network nodes without centralising sensitive log data

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
