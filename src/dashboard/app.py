"""
Streamlit Dashboard — Module 13.

Panels:
  1. Overview (stat cards + system health)
  2. Model Status (current champion, version history)
  3. Drift Events (timeline + score chart)
  4. Retraining History (jobs + promotion decisions)
  5. Audit Log (filterable, read-only)

Auth: username/password login form that calls POST /auth/login.
      JWT stored in st.session_state.  All API calls include the token.
"""
from __future__ import annotations

import os
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Self-Healing ML Pipeline",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Auth helpers ───────────────────────────────────────────────────────────────

def api(method: str, path: str, **kwargs) -> requests.Response:
    token = st.session_state.get("token")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return requests.request(
        method, f"{API_URL}{path}", headers=headers, timeout=10, **kwargs
    )


def login_form():
    st.title("🔄 Self-Healing ML Pipeline")
    st.subheader("Sign in")
    with st.form("login"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")
    if submitted:
        try:
            r = requests.post(
                f"{API_URL}/auth/login",
                data={"username": username, "password": password},
                timeout=5,
            )
            if r.status_code == 200:
                body = r.json()
                st.session_state["token"] = body["access_token"]
                st.session_state["username"] = body["username"]
                st.session_state["role"] = body["role"]
                st.rerun()
            else:
                st.error("Invalid credentials")
        except Exception as e:
            st.error(f"Cannot reach API: {e}")


def logout():
    for k in ("token", "username", "role"):
        st.session_state.pop(k, None)
    st.rerun()


# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    st.sidebar.title("🔄 ML Pipeline")
    st.sidebar.markdown(f"**User:** {st.session_state.get('username', '—')}")
    st.sidebar.markdown(f"**Role:** {st.session_state.get('role', '—')}")

    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Drift Events", "Retraining History", "Audit Log"],
    )

    st.sidebar.divider()

    # Health check
    try:
        h = api("GET", "/health").json()
        status = "🟢 Online" if h.get("status") == "ok" else "🔴 Degraded"
        model_loaded = h.get("model_loaded", False)
        st.sidebar.markdown(f"**API:** {status}")
        st.sidebar.markdown(f"**Model loaded:** {'✅' if model_loaded else '❌'}")
        st.sidebar.markdown(f"**Version:** {h.get('model_version', '—')}")
    except Exception:
        st.sidebar.markdown("**API:** 🔴 Unreachable")

    if st.sidebar.button("Sign out"):
        logout()

    return page


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_overview():
    st.title("📊 Overview")

    # Model status card
    try:
        ms = api("GET", "/model/status").json()
        col1, col2, col3 = st.columns(3)
        col1.metric("Champion Version", ms.get("version", "—"))
        col2.metric("Metric", ms.get("metric_name", "—"))
        col3.metric("Score", f"{ms.get('metric_value', 0):.4f}" if ms.get("metric_value") else "—")
        st.caption(f"Created: {ms.get('created_at', '—')}")
    except Exception as e:
        st.warning(f"Could not load model status: {e}")

    st.divider()

    # Recent drift events chart
    st.subheader("Drift Score — Last 30 Events")
    try:
        events = api("GET", "/drift/history?limit=30").json()
        if events:
            df = pd.DataFrame(events)
            df["detected_at"] = pd.to_datetime(df["detected_at"])
            fig = px.line(
                df.sort_values("detected_at"),
                x="detected_at", y="score",
                color_discrete_sequence=["#4f72ea"],
                title="",
            )
            fig.add_hline(y=0.5, line_dash="dash", line_color="orange",
                          annotation_text="threshold")
            fig.update_layout(height=300, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No drift events yet.")
    except Exception as e:
        st.warning(f"Could not load drift events: {e}")

    # Recent retraining jobs
    st.subheader("Recent Retraining Jobs")
    try:
        jobs = api("GET", "/retraining/history?limit=10").json()
        if jobs:
            st.dataframe(
                pd.DataFrame(jobs)[["id", "status", "started_at", "triggered_by"]],
                use_container_width=True,
            )
        else:
            st.info("No retraining jobs yet.")
    except Exception as e:
        st.warning(f"Could not load jobs: {e}")


def page_drift_events():
    st.title("🌊 Drift Events")
    try:
        data = api("GET", "/drift/history?limit=100").json()
        if not data:
            st.info("No drift events recorded yet.")
            return
        df = pd.DataFrame(data)
        df["detected_at"] = pd.to_datetime(df["detected_at"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Events", len(df))
        col2.metric("Drift Detected", int(df["is_drift"].sum()))
        col3.metric("Retrains Triggered", int(df["triggered_retrain"].sum()))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["detected_at"], y=df["score"], mode="lines+markers",
            name="Composite Score", line=dict(color="#4f72ea"),
            marker=dict(
                color=df["is_drift"].map({True: "#ef4444", False: "#22c55e"}),
                size=8,
            )
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color="orange")
        fig.update_layout(
            height=350, xaxis_title="Time", yaxis_title="Score",
            title="Drift Score Timeline (red = drift detected)",
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            df[["id", "detected_at", "window_id", "method", "score",
                "threshold", "is_drift", "triggered_retrain"]],
            use_container_width=True,
        )
    except Exception as e:
        st.error(f"Error loading drift events: {e}")


def page_retraining():
    st.title("⚙️ Retraining History")

    is_admin = st.session_state.get("role") in ("admin", "ml_engineer")
    if is_admin:
        st.subheader("Manual Trigger")
        if st.button("🚀 Force Retrain Now", type="primary"):
            try:
                r = api("POST", "/retrain/trigger")
                if r.status_code == 200:
                    st.success(f"Retraining job created: {r.json()['job_id']}")
                else:
                    st.error(r.json().get("detail", "Error"))
            except Exception as e:
                st.error(str(e))

        if st.session_state.get("role") == "admin":
            st.subheader("Rollback")
            if st.button("⏪ Rollback to Previous Champion", type="secondary"):
                try:
                    r = api("POST", "/model/rollback")
                    if r.status_code == 200:
                        st.success(r.json().get("message"))
                    else:
                        st.error(r.json().get("detail", "Error"))
                except Exception as e:
                    st.error(str(e))

        st.divider()

    try:
        jobs = api("GET", "/retraining/history?limit=50").json()
        if jobs:
            df = pd.DataFrame(jobs)
            status_colors = {"success": "🟢", "failed": "🔴", "running": "🟡",
                             "pending": "⚪"}
            df["status_icon"] = df["status"].map(
                lambda s: f"{status_colors.get(s, '?')} {s}"
            )
            st.dataframe(
                df[["id", "status_icon", "started_at", "finished_at",
                    "triggered_by", "candidate_model_id"]],
                use_container_width=True,
            )

            # Job status distribution
            counts = df["status"].value_counts().reset_index()
            counts.columns = ["status", "count"]
            fig = px.bar(counts, x="status", y="count",
                         color="status",
                         color_discrete_map={"success": "#22c55e",
                                             "failed": "#ef4444",
                                             "running": "#f59e0b",
                                             "pending": "#94a3b8"})
            fig.update_layout(height=250, showlegend=False,
                               margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No retraining jobs yet.")
    except Exception as e:
        st.error(f"Error: {e}")


def page_audit_log():
    st.title("📋 Audit Log")

    if st.session_state.get("role") != "admin":
        st.warning("Admin access required to view the audit log.")
        return

    col1, col2 = st.columns([3, 1])
    action_filter = col1.text_input("Filter by action (e.g. DRIFT_DETECTED)")
    limit = col2.number_input("Limit", min_value=10, max_value=500, value=100)

    params = f"?limit={limit}"
    if action_filter:
        params += f"&action={action_filter.strip().upper()}"

    try:
        logs = api("GET", f"/audit-log{params}").json()
        if logs:
            df = pd.DataFrame(logs)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.metric("Entries shown", len(df))
            st.dataframe(
                df[["id", "timestamp", "actor", "action",
                    "entity_type", "entity_id"]],
                use_container_width=True,
            )
        else:
            st.info("No audit log entries found.")
    except Exception as e:
        st.error(f"Error loading audit log: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if "token" not in st.session_state:
        login_form()
        return

    page = sidebar()

    if page == "Overview":
        page_overview()
    elif page == "Drift Events":
        page_drift_events()
    elif page == "Retraining History":
        page_retraining()
    elif page == "Audit Log":
        page_audit_log()


if __name__ == "__main__":
    main()
