"""Reproduce the README results and figure.

    python scripts/make_readme_figures.py

Runs 10,000 Monte Carlo simulations on each sample project (fixed seed) and
writes assets/schedule_risk.png and assets/sample_results.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.project_generator import generate_project  # noqa: E402
from src.risk.monte_carlo import MonteCarloEngine  # noqa: E402

N_SIM = 10_000


def simulate(kind: str):
    project = generate_project(kind)
    engine = MonteCarloEngine(project.to_dict(), n_simulations=N_SIM, random_seed=42)
    return project, engine.run()


def main() -> dict:
    out: dict[str, dict] = {}
    runs = {}
    for kind in ("residential", "commercial", "infrastructure"):
        project, res = simulate(kind)
        runs[kind] = (project, res)
        out[kind] = {
            "tasks": len(project.tasks),
            "plan_days": project.deadline_days,
            "p50_days": round(res.duration_percentile(0.5), 1),
            "p90_days": round(res.duration_percentile(0.9), 1),
            "prob_on_time": round(res.prob_on_time(), 3),
            "plan_budget_eur": project.budget,
            "p90_cost_eur": round(res.cost_percentile(0.9), -2),
            "prob_within_budget": round(res.prob_within_budget(), 3),
        }

    project, res = runs["residential"]
    names = {t.id: t.name for t in project.tasks}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.9), gridspec_kw={"width_ratios": [1.3, 1]})
    a1.hist(res.durations, bins=60, color="#1f6feb", alpha=0.75)
    top = a1.get_ylim()[1]
    for val, label, col, ha, y in [(project.deadline_days, "plan (most likely)", "#cf222e", "right", 0.97),
                                   (res.duration_percentile(0.5), "P50", "#24292f", "left", 0.80),
                                   (res.duration_percentile(0.9), "P90", "#9a6700", "left", 0.97)]:
        a1.axvline(val, ls="--", lw=1.2, color=col)
        pad = -0.6 if ha == "right" else 0.6
        a1.text(val + pad, top * y, f"{label}\n{val:.0f} d", fontsize=8, color=col, va="top", ha=ha)
    a1.set_xlabel("Project duration (working days)")
    a1.set_ylabel("Simulations")
    a1.set_title(f"Residential building: {res.prob_on_time():.0%} chance of meeting the plan", fontsize=10)

    crit = sorted(res.criticality.items(), key=lambda kv: kv[1])
    crit = [(k, v) for k, v in crit if 0.02 < v < 0.999][-8:]
    if crit:
        a2.barh([names[k][:28] for k, _ in crit], [v * 100 for _, v in crit], color="#8250df", alpha=0.8, height=0.5)
    a2.set_xlabel("Criticality index (% of simulations on the critical path)")
    a2.set_title("Tasks whose criticality depends on chance", fontsize=10)
    a2.tick_params(axis="y", labelsize=8)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (ROOT / "assets").mkdir(exist_ok=True)
    fig.savefig(ROOT / "assets" / "schedule_risk.png", dpi=150)
    (ROOT / "assets" / "sample_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
