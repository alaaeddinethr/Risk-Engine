"""
monte_carlo.py
--------------
Monte Carlo simulation engine for construction project risk.

For each simulation:
  1. Sample task durations from Beta-PERT distributions
  2. Sample task costs from Beta-PERT distributions
  3. Run CPM to compute project duration and identify critical path
  4. Record: total duration, total cost, critical tasks

Outputs:
  - Distribution of project completion dates (P10/P50/P90/P95)
  - Distribution of final project cost
  - Criticality index per task (P(task is on critical path))
  - Probability of meeting deadline / budget

Reference:
    Vose, D. (2008). Risk Analysis: A Quantitative Guide. Wiley.
    PMI Practice Standard for Scheduling, 3rd Ed.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.schedule.cpm import CPMScheduler
from src.schedule.pert import PERTEstimate, pert_from_task

# ─── Simulation Result ────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Full Monte Carlo simulation output."""
    n_simulations: int
    durations: np.ndarray      # shape: (n_sim,) — project duration per sim
    costs: np.ndarray          # shape: (n_sim,) — total cost per sim
    criticality: dict[str, float]  # task_id → P(on critical path)
    baseline_duration: float
    baseline_cost: float
    deadline: float | None = None
    budget: float | None = None

    # Percentiles
    def duration_percentile(self, p: float) -> float:
        return float(np.percentile(self.durations, p * 100))

    def cost_percentile(self, p: float) -> float:
        return float(np.percentile(self.costs, p * 100))

    def prob_on_time(self) -> float:
        if self.deadline is None:
            return float("nan")
        return float((self.durations <= self.deadline).mean())

    def prob_within_budget(self) -> float:
        if self.budget is None:
            return float("nan")
        return float((self.costs <= self.budget).mean())

    def duration_var(self, confidence: float = 0.95) -> float:
        """Schedule Risk = P95 − baseline (days at risk)."""
        return self.duration_percentile(confidence) - self.baseline_duration

    def cost_var(self, confidence: float = 0.95) -> float:
        """Cost Overrun VaR = P95 cost − baseline cost."""
        return self.cost_percentile(confidence) - self.baseline_cost

    def top_critical_tasks(self, top_n: int = 5) -> list[tuple[str, float]]:
        """Return top N tasks by criticality index, descending."""
        return sorted(self.criticality.items(), key=lambda x: -x[1])[:top_n]

    def summary(self) -> dict:
        return {
            "n_simulations":           self.n_simulations,
            "baseline_duration_days":  round(self.baseline_duration, 1),
            "baseline_cost_eur":       round(self.baseline_cost, 0),
            "p50_duration":            round(self.duration_percentile(0.50), 1),
            "p80_duration":            round(self.duration_percentile(0.80), 1),
            "p90_duration":            round(self.duration_percentile(0.90), 1),
            "p95_duration":            round(self.duration_percentile(0.95), 1),
            "schedule_var_95_days":    round(self.duration_var(0.95), 1),
            "p50_cost":                round(self.cost_percentile(0.50), 0),
            "p80_cost":                round(self.cost_percentile(0.80), 0),
            "p90_cost":                round(self.cost_percentile(0.90), 0),
            "p95_cost":                round(self.cost_percentile(0.95), 0),
            "cost_var_95_eur":         round(self.cost_var(0.95), 0),
            "cost_var_95_pct":         round(self.cost_var(0.95) / max(self.baseline_cost, 1) * 100, 1),
            "prob_on_time":            round(self.prob_on_time(), 3) if self.deadline else None,
            "prob_within_budget":      round(self.prob_within_budget(), 3) if self.budget else None,
        }

    def print_report(self) -> None:
        s = self.summary()
        bd = s["baseline_duration_days"]
        bc = s["baseline_cost_eur"]
        print(f"\n{'═'*58}")
        print(f"  MONTE CARLO RISK REPORT — {self.n_simulations:,} simulations")
        print(f"{'═'*58}")
        print("  SCHEDULE RISK")
        print(f"  {'─'*50}")
        print(f"  Baseline:   {bd:>7.1f} days")
        print(f"  P50:        {s['p50_duration']:>7.1f} days  (+{s['p50_duration']-bd:.1f}d)")
        print(f"  P80:        {s['p80_duration']:>7.1f} days  (+{s['p80_duration']-bd:.1f}d)")
        print(f"  P90:        {s['p90_duration']:>7.1f} days  (+{s['p90_duration']-bd:.1f}d)")
        print(f"  P95:        {s['p95_duration']:>7.1f} days  (+{s['p95_duration']-bd:.1f}d)")
        if self.deadline:
            print(f"  P(on time): {s['prob_on_time']*100:>7.1f}%  (deadline: {self.deadline:.0f}d)")
        print("\n  COST RISK")
        print(f"  {'─'*50}")
        print(f"  Baseline:   €{bc:>12,.0f}")
        print(f"  P50:        €{s['p50_cost']:>12,.0f}  (+€{s['p50_cost']-bc:,.0f})")
        print(f"  P90:        €{s['p90_cost']:>12,.0f}  (+€{s['p90_cost']-bc:,.0f})")
        print(f"  Cost VaR95: €{s['cost_var_95_eur']:>12,.0f}  (+{s['cost_var_95_pct']:.1f}%)")
        if self.budget:
            print(f"  P(in budget): {s['prob_within_budget']*100:.1f}%  (budget: €{self.budget:,.0f})")
        print("\n  TOP CRITICAL TASKS (criticality index)")
        print(f"  {'─'*50}")
        for tid, ci in self.top_critical_tasks(5):
            bar = "█" * int(ci * 25)
            print(f"  {tid:<5} {ci*100:6.1f}%  {bar}")
        print(f"{'═'*58}")


# ─── Monte Carlo Engine ───────────────────────────────────────────────────────

class MonteCarloEngine:
    """
    Runs Monte Carlo simulation on a construction project.

    Args:
        project_dict: Project data dict (from project_generator or JSON)
        n_simulations: Number of simulation iterations
        random_seed: Reproducibility seed
    """

    def __init__(
        self,
        project_dict: dict,
        n_simulations: int = 10_000,
        random_seed: int = 42,
    ):
        self.project = project_dict
        self.tasks = project_dict["tasks"]
        self.n_sim = n_simulations
        self.rng = np.random.default_rng(random_seed)

        # Pre-build PERT distributions for each task
        self.duration_perts: dict[str, PERTEstimate] = {}
        self.cost_perts: dict[str, PERTEstimate] = {}

        for t in self.tasks:
            self.duration_perts[t["id"]] = pert_from_task(t, "duration")
            self.cost_perts[t["id"]]     = pert_from_task(t, "cost")

        # Baseline values: critical path with PERT mean durations (a plain sum
        # would add up parallel tasks), and total of PERT mean costs.
        self.baseline_duration, _ = self._run_single(
            {tid: p.mean for tid, p in self.duration_perts.items()})
        self.baseline_cost = sum(p.mean for p in self.cost_perts.values())

    def _run_single(self, sampled_durations: dict[str, float]) -> tuple[float, list[str]]:
        """
        Run one CPM computation with sampled durations.
        Returns (project_duration, critical_path_task_ids).
        """
        task_dicts = [
            {
                "id": t["id"],
                "name": t["name"],
                "phase": t.get("phase", ""),
                "trade": t.get("trade", ""),
                "predecessors": t.get("predecessors", []),
                "duration_most_likely": sampled_durations[t["id"]],
            }
            for t in self.tasks
        ]
        scheduler = CPMScheduler(task_dicts, durations=sampled_durations)
        result = scheduler.compute()
        return result.project_duration, result.critical_path

    def run(self) -> SimulationResult:
        """
        Execute full Monte Carlo simulation.

        Returns:
            SimulationResult with all outputs.
        """
        task_ids = [t["id"] for t in self.tasks]
        n = self.n_sim

        # Pre-sample all durations and costs at once (vectorized)
        duration_samples = np.zeros((n, len(task_ids)))
        cost_samples     = np.zeros((n, len(task_ids)))

        for i, tid in enumerate(task_ids):
            duration_samples[:, i] = self.duration_perts[tid].sample(n, rng=self.rng)
            cost_samples[:, i]     = self.cost_perts[tid].sample(n, rng=self.rng)

        # Total costs per simulation (simple sum — no CPM needed for cost)
        total_costs = cost_samples.sum(axis=1)

        # CPM for each simulation (schedule depends on dependency graph)
        project_durations = np.zeros(n)
        criticality_counts = {tid: 0 for tid in task_ids}

        print(f"🔄 Running {n:,} CPM simulations...")
        batch = max(1, n // 10)
        for i in range(n):
            sampled_dur = {task_ids[j]: float(duration_samples[i, j])
                           for j in range(len(task_ids))}
            proj_dur, crit_path = self._run_single(sampled_dur)
            project_durations[i] = proj_dur
            for tid in crit_path:
                criticality_counts[tid] += 1
            if (i + 1) % batch == 0:
                print(f"  {i+1:,}/{n:,} done ({(i+1)/n*100:.0f}%)")

        criticality_index = {
            tid: count / n for tid, count in criticality_counts.items()
        }

        print("✅ Simulation complete.")
        return SimulationResult(
            n_simulations=n,
            durations=project_durations,
            costs=total_costs,
            criticality=criticality_index,
            baseline_duration=self.baseline_duration,
            baseline_cost=self.baseline_cost,
            deadline=self.project.get("deadline_days"),
            budget=self.project.get("budget"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="data/project.json")
    parser.add_argument("--simulations", type=int, default=1000)
    args = parser.parse_args()

    if not Path(args.project).exists():
        print("Generating project first...")
        import subprocess
        subprocess.run(["python", "src/data/project_generator.py",
                        "--output", args.project])

    with open(args.project, encoding="utf-8") as f:
        project = json.load(f)

    engine = MonteCarloEngine(project, n_simulations=args.simulations)
    result = engine.run()
    result.print_report()
