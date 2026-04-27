"""
dashboard/app.py
Streamlit dashboard for real-time threat monitoring and agent decision review.

Launch:
    streamlit run dashboard/app.py
"""

import streamlit as st
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

st.set_page_config(
    page_title="Agentic Threat Detection System",
    page_icon="🛡️",
    layout="wide",
)

# ── Helpers ────────────────────────────────────────────────────────────────
@st.cache_data
def load_incidents(log_dir: str = "results/incident_logs") -> pd.DataFrame:
    records = []
    if os.path.exists(log_dir):
        for fname in sorted(os.listdir(log_dir)):
            if fname.endswith(".jsonl"):
                with open(os.path.join(log_dir, fname)) as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
    return pd.DataFrame(records) if records else pd.DataFrame()


@st.cache_data
def load_metrics() -> dict:
    paths = [
        "results/metrics.json",
        "results/agent_comparison.json",
        "results/ablation.json",
    ]
    data = {}
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                data[os.path.basename(p).replace(".json", "")] = json.load(f)
    return data


def action_color(action: str) -> str:
    return {
        "IGNORE":   "#6B7280",
        "ALERT":    "#F59E0B",
        "BLOCK":    "#EF4444",
        "ESCALATE": "#8B5CF6",
    }.get(action, "#374151")


# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/System-Active-brightgreen", width=150)
st.sidebar.title("🛡️ ATS Control Panel")
view = st.sidebar.radio(
    "Navigation",
    ["Overview", "Incident Feed", "Model Performance", "Agent Comparison",
     "Ablation Study", "Training Curves"],
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Agentic Threat Detection System")
st.sidebar.markdown("**Author:** Awodola Olusola Ebenezer")
st.sidebar.markdown("**Status:** 🔬 Research Development Phase")

# ── Main Content ───────────────────────────────────────────────────────────
st.title("🛡️ Agentic Threat Detection System")
st.caption("Deep Learning + Reinforcement Learning for Autonomous Cyber Defense")

metrics_data = load_metrics()
incidents_df = load_incidents()

# ── Overview ───────────────────────────────────────────────────────────────
if view == "Overview":
    col1, col2, col3, col4 = st.columns(4)

    det_metrics = metrics_data.get("metrics", {})
    col1.metric("Detection Accuracy",
                f"{det_metrics.get('accuracy', 0):.2%}" if det_metrics else "Pending",
                help="Test set accuracy of BiLSTM+CNN detection model")
    col1.caption("⚠️ Awaiting training run")

    col2.metric("Macro F1 Score",
                f"{det_metrics.get('macro_f1', 0):.4f}" if det_metrics else "Pending")
    col2.caption("⚠️ Awaiting training run")

    col3.metric("Mean FPR",
                f"{det_metrics.get('fpr', 0):.4f}" if det_metrics else "Pending")
    col3.caption("⚠️ Awaiting training run")

    total_incidents = len(incidents_df) if not incidents_df.empty else 0
    col4.metric("Total Incidents Logged", total_incidents)

    st.divider()
    st.subheader("Research Summary")
    st.info(
        "**Research question:** Does integrating a deep learning-based anomaly detector "
        "(BiLSTM+CNN) with a reinforcement learning response agent (Dueling DQN) reduce "
        "Mean Time to Respond (MTTR) and False Positive Rate (FPR) compared to static "
        "rule-based thresholds in an autonomous cyber defense context?\n\n"
        "**Dataset:** CICIDS2017 (2.8M flow records, 15 classes)\n\n"
        "**Status:** Project is in active development. Results will be updated as "
        "experiments are completed."
    )

    st.subheader("System Architecture")
    st.code("""
    Network Flows → Preprocessing → BiLSTM+CNN Detection Model → Threat Probabilities (15 classes)
                                                                          ↓
                                                           DQN Agent State (18-dim)
                                                                          ↓
                                              Perceive → Reason → Act → Reflect
                                                                          ↓
                                        IGNORE | ALERT | BLOCK | ESCALATE
    """, language="text")

# ── Incident Feed ──────────────────────────────────────────────────────────
elif view == "Incident Feed":
    st.subheader("📋 Recent Agent Decisions")
    if incidents_df.empty:
        st.warning(
            "No incidents logged yet. Run the agent in simulation mode:\n\n"
            "```bash\npython run_agent.py --mode simulate --data data/sample/sample_traffic.csv\n```"
        )
    else:
        # Filter controls
        col1, col2 = st.columns(2)
        action_filter = col1.multiselect(
            "Filter by action", ["IGNORE", "ALERT", "BLOCK", "ESCALATE"],
            default=["ALERT", "BLOCK", "ESCALATE"]
        )
        conf_threshold = col2.slider("Min confidence", 0.0, 1.0, 0.5, 0.05)

        filtered = incidents_df[
            incidents_df["action"].isin(action_filter) &
            (incidents_df["confidence"] >= conf_threshold)
        ]

        st.dataframe(
            filtered[["timestamp", "predicted_threat", "action",
                       "confidence", "outcome", "response_time_ms"]].tail(100),
            use_container_width=True,
        )

        # Action distribution
        st.subheader("Action Distribution")
        action_counts = incidents_df["action"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(action_counts.index, action_counts.values,
                      color=["#6B7280", "#F59E0B", "#EF4444", "#8B5CF6"])
        ax.set_ylabel("Count")
        ax.set_title("Agent Action Distribution")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)

# ── Model Performance ──────────────────────────────────────────────────────
elif view == "Model Performance":
    st.subheader("📊 Detection Model — Test Set Results")
    det_metrics = metrics_data.get("metrics", {})

    if not det_metrics:
        st.warning(
            "No results yet. Train the model first:\n\n"
            "```bash\npython train.py\npython scripts/evaluate.py\n```"
        )
        st.info(
            "**Why no results?**\n\n"
            "This project requires the CICIDS2017 dataset (~1.2GB compressed). "
            "Results will be computed from real experiments and updated here. "
            "We explicitly avoid hardcoding synthetic numbers — all metrics "
            "in this dashboard reflect actual model training outputs."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{det_metrics['accuracy']:.4f}")
        col2.metric("Macro F1",  f"{det_metrics['macro_f1']:.4f}")
        col3.metric("ROC-AUC",   f"{det_metrics['roc_auc']:.4f}")
        col4.metric("Mean FPR",  f"{det_metrics['fpr']:.4f}")

        if "classification_report_text" in det_metrics:
            st.text("Full Classification Report:")
            st.code(det_metrics["classification_report_text"])

    # Show figures if available
    figures_dir = "results/figures"
    for fig_name, title in [
        ("confusion_matrix.png", "Confusion Matrix"),
        ("roc_curves.png", "ROC Curves"),
    ]:
        fig_path = os.path.join(figures_dir, fig_name)
        if os.path.exists(fig_path):
            st.subheader(title)
            st.image(fig_path)
        else:
            st.info(f"{title}: Run `python scripts/evaluate.py` to generate this figure.")

# ── Agent Comparison ───────────────────────────────────────────────────────
elif view == "Agent Comparison":
    st.subheader("🤖 DQN Agent vs Rule-Based Baseline vs Random Policy")
    agent_data = metrics_data.get("agent_comparison", {})

    if not agent_data:
        st.warning(
            "No agent comparison results yet. Run:\n\n"
            "```bash\n"
            "python run_agent.py --mode train_rl\n"
            "python scripts/evaluate.py\n"
            "```"
        )
        st.info(
            "**Research hypothesis being tested:**\n\n"
            "H1: The DQN-based response policy achieves higher F1 and lower FPR "
            "than the rule-based threshold policy.\n\n"
            "H2: The DQN policy achieves lower MTTR than manual escalation baselines.\n\n"
            "Results will populate here once RL training is complete."
        )
    else:
        rows = []
        for policy, res in agent_data.items():
            if "status" not in res:
                rows.append({
                    "Policy": policy,
                    "F1": res.get("f1", 0),
                    "FPR": res.get("fpr", 0),
                    "Precision": res.get("precision", 0),
                    "Recall": res.get("recall", 0),
                    "MTTR (ms)": res.get("mean_response_time_ms", 0),
                    "Total Reward": res.get("total_reward", 0),
                })
        if rows:
            df = pd.DataFrame(rows).set_index("Policy")
            st.dataframe(df.style.highlight_max(subset=["F1", "Precision", "Recall"])
                           .highlight_min(subset=["FPR", "MTTR (ms)"]),
                         use_container_width=True)

# ── Ablation Study ─────────────────────────────────────────────────────────
elif view == "Ablation Study":
    st.subheader("🔬 Ablation Study: Architecture Variants")
    ablation_data = metrics_data.get("ablation", {})

    if not ablation_data:
        st.warning(
            "No ablation results yet. Run:\n\n"
            "```bash\n"
            "python train.py --variant lstm_only\n"
            "python train.py --variant cnn_only\n"
            "python train.py --variant hybrid\n"
            "python scripts/evaluate.py --ablation\n"
            "```"
        )
        st.info(
            "**Ablation design:**\n\n"
            "Three variants are trained identically (same data, seed, epochs) "
            "and compared on the held-out test set.\n\n"
            "- **lstm_only**: BiLSTM encoder → classification head (no CNN)\n"
            "- **cnn_only**: CNN classifier → classification head (no LSTM)\n"
            "- **hybrid**: BiLSTM + CNN → fused head ← *primary model*\n\n"
            "This tests hypothesis H0: Hybrid F1 > max(LSTM F1, CNN F1)"
        )
    else:
        rows = []
        for variant, res in ablation_data.items():
            if "status" not in res:
                rows.append({
                    "Variant": variant,
                    "Accuracy": res.get("accuracy", 0),
                    "Macro F1": res.get("macro_f1", 0),
                    "ROC-AUC": res.get("roc_auc", 0),
                    "FPR": res.get("fpr", 0),
                })
        if rows:
            df = pd.DataFrame(rows).set_index("Variant")
            st.dataframe(df.style.highlight_max(subset=["Accuracy", "Macro F1", "ROC-AUC"])
                           .highlight_min(subset=["FPR"]),
                         use_container_width=True)

# ── Training Curves ────────────────────────────────────────────────────────
elif view == "Training Curves":
    st.subheader("📈 Training History")
    curves_path = "results/figures/training_curves.png"
    if os.path.exists(curves_path):
        st.image(curves_path)
    else:
        st.warning("Training curves not available. Run: `python train.py`")

    history_path = "results/logs/history.json"
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        col1, col2 = st.columns(2)
        col1.line_chart(
            pd.DataFrame({
                "train_loss": history["train_loss"],
                "val_loss": history["val_loss"],
            })
        )
        col2.line_chart(pd.DataFrame({"val_accuracy": history["val_acc"]}))
