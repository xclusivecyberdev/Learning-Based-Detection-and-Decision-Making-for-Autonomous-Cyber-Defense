"""
dashboard/components.py
Reusable Streamlit UI components.
"""

import streamlit as st


def alert_card(action: str, threat: str, confidence: float, reasoning: str):
    color_map = {
        "IGNORE":   "#6B7280",
        "ALERT":    "#F59E0B",
        "BLOCK":    "#EF4444",
        "ESCALATE": "#8B5CF6",
    }
    color = color_map.get(action, "#374151")
    st.markdown(
        f"""
        <div style="border-left: 4px solid {color}; padding: 10px 16px;
                    background: #1F2937; border-radius: 6px; margin-bottom: 8px;">
            <b style="color:{color}">{action}</b> &nbsp;|&nbsp;
            <span style="color:#D1D5DB">{threat}</span> &nbsp;|&nbsp;
            <span style="color:#9CA3AF">conf={confidence:.3f}</span>
            <br/>
            <small style="color:#6B7280">{reasoning}</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


def threat_gauge(confidence: float, label: str = "Threat Confidence"):
    """Simple CSS-based confidence bar."""
    pct = int(confidence * 100)
    color = "#EF4444" if pct > 80 else "#F59E0B" if pct > 50 else "#1D9E75"
    st.markdown(
        f"""
        <div style="margin-bottom:8px">
            <small>{label}: {pct}%</small>
            <div style="background:#374151; border-radius:4px; height:8px;">
                <div style="background:{color}; width:{pct}%; border-radius:4px; height:8px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
