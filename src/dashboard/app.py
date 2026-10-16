"""
app.py — Construction Risk Dashboard (Streamlit)

Sections:
  1. Project overview — tasks, phases, Gantt baseline
  2. Monte Carlo results — duration/cost distributions, P10/P50/P90
  3. Criticality analysis — which tasks drive the schedule
  4. Cost risk — VaR, cost overrun distribution
  5. Sensitivity — tornado chart
  6. Stress test — what-if analysis (worst-case selected tasks)

Run:
    streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.project_generator import generate_project
from src.risk.monte_carlo import MonteCarloEngine
from src.risk.sensitivity import SensitivityAnalyzer
from src.schedule.cpm import CPMScheduler
from src.schedule.pert import PERTEstimate

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Risk Dashboard",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@300;400;500;600&family=Courier+Prime:wght@400;700&display=swap');
html,[class*="css"]{font-family:'Libre Franklin',sans-serif}
.stApp{background:#f7f5f0}
h1,h2,h3{font-family:'Libre Franklin',sans-serif;font-weight:600}
.kpi{background:#fff;border:1px solid #e0dcd4;border-radius:4px;padding:1.1rem;text-align:center;
     border-top:4px solid #1e3a5f}
.kpi.red{border-top-color:#dc2626}.kpi.amber{border-top-color:#d97706}
.kpi.green{border-top-color:#16a34a}.kpi.blue{border-top-color:#1e3a5f}
.kpi-label{font-size:.63rem;color:#6b7280;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem}
.kpi-value{font-family:'Courier Prime',monospace;font-size:1.9rem;font-weight:700;color:#1f2937;line-height:1}
.kpi-sub{font-size:.68rem;color:#9ca3af;margin-top:.2rem}
.section-hdr{font-size:.8rem;font-weight:600;letter-spacing:.06em;text-transform:uppercase;
  color:#374151;border-bottom:2px solid #e5e7eb;padding-bottom:.4rem;margin-bottom:1rem}
</style>
""", unsafe_allow_html=True)

THEME = dict(paper_bgcolor="#fff", plot_bgcolor="#fff",
             font=dict(color="#374151", family="Libre Franklin"),
             xaxis=dict(gridcolor="#f3f4f6"), yaxis=dict(gridcolor="#f3f4f6"))

PHASE_COLORS = {
    "Préparation": "#3b82f6", "VRD": "#10b981", "Gros Oeuvre": "#f59e0b",
    "Second Oeuvre": "#8b5cf6", "Finitions": "#ec4899",
    "Réception": "#6b7280", "Études": "#06b6d4",
    "Terrassement": "#84cc16", "Réseaux": "#f97316",
    "Chaussée": "#a855f7", "OA": "#ef4444", "Extérieurs": "#14b8a6",
}

def kpi(label, value, sub="", color="blue"):
    st.markdown(f"""
    <div class="kpi {color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ Construction Risk")
    st.divider()
    project_type = st.selectbox("Project type",
        ["residential","commercial","infrastructure"],
        format_func=lambda x: {"residential":"Résidentiel R+3",
                               "commercial":"Bureaux R+2",
                               "infrastructure":"VRD / Infrastructure"}[x])
    n_sim = st.select_slider("Monte Carlo iterations",
        options=[500,1000,2000,5000], value=1000)
    seed = st.number_input("Random seed", value=42, min_value=0, step=1)
    run_btn = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    st.divider()
    st.markdown("### What-if: Delay Selected Tasks")
    delay_pct = st.slider("Duration multiplier on selected tasks (%)", 100, 200, 130)
    st.caption("Simulates weather, supply chain, or labour disruptions")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("# 🏗️ Construction Project Risk Dashboard")
st.markdown("*Monte Carlo · CPM · Cost VaR · Criticality Analysis*")
st.divider()

if not run_btn:
    st.info("Select a project type and click **Run Simulation** to start.")
    st.stop()

# Generate & simulate
with st.spinner("Generating project..."):
    project = generate_project(project_type)
    task_list = [
        {"id": t.id, "name": t.name, "phase": t.phase, "trade": t.trade,
         "predecessors": t.predecessors,
         "duration_optimistic": t.duration_optimistic,
         "duration_most_likely": t.duration_most_likely,
         "duration_pessimistic": t.duration_pessimistic,
         "cost_optimistic": t.cost_optimistic,
         "cost_most_likely": t.cost_most_likely,
         "cost_pessimistic": t.cost_pessimistic}
        for t in project.tasks
    ]
    project_dict = {
        "name": project.name, "project_type": project.project_type,
        "tasks": task_list,
        "deadline_days": project.deadline_days,
        "budget": project.budget,
    }

with st.spinner(f"Running {n_sim:,} Monte Carlo simulations..."):
    engine = MonteCarloEngine(project_dict, n_simulations=n_sim, random_seed=int(seed))
    # Pre-sample arrays for sensitivity
    task_ids = [t.id for t in project.tasks]
    task_names = {t.id: t.name for t in project.tasks}
    rng2 = np.random.default_rng(int(seed))
    dur_samples  = np.array([engine.duration_perts[tid].sample(n_sim, rng=rng2)
                              for tid in task_ids]).T
    cost_samples = np.array([engine.cost_perts[tid].sample(n_sim, rng=rng2)
                              for tid in task_ids]).T
    result = engine.run()

# CPM baseline
scheduler = CPMScheduler(task_list)
cpm = scheduler.compute()

# Sensitivity
sensitivity = SensitivityAnalyzer(
    dur_samples, cost_samples,
    result.durations, result.costs,
    task_ids, task_names, result.criticality
)
dur_sens  = sensitivity.duration_sensitivity()
cost_sens = sensitivity.cost_sensitivity()

# ─── KPIs ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">PROJECT RISK SUMMARY</div>', unsafe_allow_html=True)

bd = result.baseline_duration
bc = result.baseline_cost
p50d = result.duration_percentile(0.50)
p90d = result.duration_percentile(0.90)
p90c = result.cost_percentile(0.90)
var95c = result.cost_var(0.95)
pot = result.prob_on_time()

c1,c2,c3,c4,c5,c6 = st.columns(6)
with c1: kpi("Baseline Duration", f"{bd:.0f}d", "CPM PERT means", "blue")
with c2: kpi("P50 Completion", f"{p50d:.0f}d", f"+{p50d-bd:.0f}d vs baseline", "amber")
with c3: kpi("P90 Completion", f"{p90d:.0f}d", f"+{p90d-bd:.0f}d vs baseline", "red")
with c4: kpi("P(On Time)", f"{pot*100:.0f}%" if not np.isnan(pot) else "N/A",
             f"Deadline: {project.deadline_days}d", "green" if pot>0.5 else "red")
with c5: kpi("Baseline Cost", f"€{bc/1e3:.0f}k", "PERT means sum", "blue")
with c6: kpi("Cost VaR 95%", f"+€{var95c/1e3:.0f}k",
             f"+{var95c/bc*100:.1f}% overrun", "red")

st.divider()

# ─── Gantt + Schedule distribution ───────────────────────────────────────────
st.markdown('<div class="section-hdr">BASELINE SCHEDULE (CPM)</div>', unsafe_allow_html=True)

gantt_data = scheduler.gantt_data()
fig_gantt = go.Figure()
for row in gantt_data:
    color = PHASE_COLORS.get(row["phase"], "#94a3b8")
    border = "#ef4444" if row["critical"] else color
    fig_gantt.add_trace(go.Bar(
        x=[row["finish"] - row["start"]], y=[f"{row['id']} — {row['name'][:28]}"],
        base=row["start"], orientation="h",
        marker=dict(color=color, opacity=.8,
                    line=dict(color=border, width=2 if row["critical"] else 0.5)),
        name=row["phase"], showlegend=False,
        hovertemplate=f"{row['name']}<br>Start: {row['start']:.1f}d → End: {row['finish']:.1f}d<br>"
                      f"Duration: {row['duration']:.1f}d | Float: {row['float']:.1f}d<extra></extra>",
    ))
if project.deadline_days:
    fig_gantt.add_vline(x=project.deadline_days, line=dict(color="#ef4444", dash="dash", width=2),
                        annotation_text="Deadline", annotation_font=dict(color="#ef4444"))
fig_gantt.add_vline(x=bd, line=dict(color="#374151", dash="dot", width=1.5),
                    annotation_text="Baseline")
fig_gantt.update_layout(
    title="Gantt Chart — Critical tasks highlighted in red",
    xaxis_title="Project Days", height=max(350, len(gantt_data)*22),
    bargap=0.15, **THEME
)
st.plotly_chart(fig_gantt, use_container_width=True)

st.divider()

# ─── MC Results ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">MONTE CARLO RESULTS</div>', unsafe_allow_html=True)

col_dur, col_cost = st.columns(2)
with col_dur:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=result.durations, nbinsx=60,
        marker_color="#1e3a5f", opacity=0.75, name="Simulated duration"))
    for pct, color, label in [(50,"#3b82f6","P50"),(80,"#f59e0b","P80"),(90,"#ef4444","P90")]:
        v = result.duration_percentile(pct/100)
        fig.add_vline(x=v, line=dict(color=color, dash="dash", width=1.8),
                      annotation_text=f"{label}: {v:.0f}d",
                      annotation_font=dict(color=color))
    fig.add_vline(x=bd, line=dict(color="#374151", width=2),
                  annotation_text=f"Baseline: {bd:.0f}d")
    if project.deadline_days:
        fig.add_vline(x=project.deadline_days,
                      line=dict(color="#ef4444", dash="dot", width=2),
                      annotation_text="Deadline")
    fig.update_layout(title="Project Duration Distribution", xaxis_title="Days",
                      height=320, **THEME)
    st.plotly_chart(fig, use_container_width=True)

with col_cost:
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(x=result.costs/1e3, nbinsx=60,
        marker_color="#065f46", opacity=0.75, name="Simulated cost"))
    for pct, color, label in [(50,"#3b82f6","P50"),(80,"#f59e0b","P80"),(90,"#ef4444","P90")]:
        v = result.cost_percentile(pct/100)
        fig2.add_vline(x=v/1e3, line=dict(color=color, dash="dash", width=1.8),
                       annotation_text=f"{label}: €{v/1e3:.0f}k",
                       annotation_font=dict(color=color))
    fig2.add_vline(x=bc/1e3, line=dict(color="#374151", width=2),
                   annotation_text=f"Baseline: €{bc/1e3:.0f}k")
    if project.budget:
        fig2.add_vline(x=project.budget/1e3,
                       line=dict(color="#ef4444", dash="dot", width=2),
                       annotation_text="Budget")
    fig2.update_layout(title="Total Cost Distribution (€k)", xaxis_title="Cost (€k)",
                       height=320, **THEME)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ─── Criticality ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">CRITICALITY INDEX</div>', unsafe_allow_html=True)

crit_df = pd.DataFrame([
    {"id": tid, "name": task_names.get(tid, tid), "criticality": ci}
    for tid, ci in sorted(result.criticality.items(), key=lambda x: -x[1])
]).head(12)

fig_crit = go.Figure(go.Bar(
    x=crit_df["criticality"]*100,
    y=crit_df["name"],
    orientation="h",
    marker_color=[f"rgba(239,68,68,{max(0.2, c)})" for c in crit_df["criticality"]],
    text=[f"{c*100:.0f}%" for c in crit_df["criticality"]],
    textposition="outside",
))
fig_crit.add_vline(x=50, line=dict(color="#6b7280", dash="dot"),
                   annotation_text="50% threshold")
fig_crit.update_layout(
    title="Criticality Index — P(task on critical path)",
    xaxis_title="Criticality (%)", xaxis_range=[0,115],
    height=max(300, len(crit_df)*30), **THEME,
)
st.plotly_chart(fig_crit, use_container_width=True)

st.divider()

# ─── Sensitivity ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">SENSITIVITY ANALYSIS — TORNADO CHART</div>',
            unsafe_allow_html=True)

col_ts, col_cs = st.columns(2)
with col_ts:
    dur_df = dur_sens.to_dataframe().head(10)
    fig_t = go.Figure(go.Bar(
        x=dur_df["correlation"],
        y=dur_df["task_name"],
        orientation="h",
        marker_color=[("#ef4444" if c>0 else "#3b82f6") for c in dur_df["correlation"]],
    ))
    fig_t.update_layout(title="Schedule Sensitivity (Spearman ρ)",
                         xaxis_title="Correlation with project duration",
                         xaxis_range=[-1,1], height=360, **THEME)
    st.plotly_chart(fig_t, use_container_width=True)

with col_cs:
    cost_df = cost_sens.to_dataframe().head(10)
    fig_c = go.Figure(go.Bar(
        x=cost_df["correlation"],
        y=cost_df["task_name"],
        orientation="h",
        marker_color=[("#065f46" if c>0 else "#1e3a5f") for c in cost_df["correlation"]],
    ))
    fig_c.update_layout(title="Cost Sensitivity (Spearman ρ)",
                         xaxis_title="Correlation with total cost",
                         xaxis_range=[-1,1], height=360, **THEME)
    st.plotly_chart(fig_c, use_container_width=True)

st.divider()

# ─── Full task table ──────────────────────────────────────────────────────────
with st.expander("📋 Full Task Schedule & Risk Register"):
    rows = []
    for t in project.tasks:
        from src.schedule.pert import PERTEstimate
        dp = PERTEstimate(t.duration_optimistic, t.duration_most_likely, t.duration_pessimistic)
        cp = PERTEstimate(t.cost_optimistic, t.cost_most_likely, t.cost_pessimistic)
        rows.append({
            "ID":           t.id,
            "Task":         t.name,
            "Phase":        t.phase,
            "Risk":         t.risk_level,
            "Dur. O/M/P":   f"{t.duration_optimistic}/{t.duration_most_likely}/{t.duration_pessimistic}d",
            "Dur. PERT":    f"{dp.mean:.1f}d",
            "Cost PERT":    f"€{cp.mean:,.0f}",
            "Criticality":  f"{result.criticality.get(t.id, 0)*100:.0f}%",
            "Float (CPM)":  f"{cpm.tasks.get(t.id).total_float:.1f}d"
                            if t.id in cpm.tasks else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=400)

st.divider()
st.markdown(
    "<div style='text-align:center;font-size:.72rem;color:#9ca3af'>"
    "Construction Risk Dashboard · Alaa Eddine TAHIR · Arts et Métiers ParisTech"
    "</div>", unsafe_allow_html=True
)
