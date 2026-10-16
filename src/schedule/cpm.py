"""
cpm.py
------
Critical Path Method (CPM) for construction project scheduling.

Computes:
  - Early Start (ES) / Early Finish (EF) for each task — forward pass
  - Late Start (LS) / Late Finish (LF) for each task — backward pass
  - Total Float (TF) = LS - ES = LF - EF
  - Critical Path: tasks with TF = 0
  - Project duration = EF of final task

Can run on:
  - Deterministic durations (PERT means) → classic CPM
  - Sampled durations → used inside Monte Carlo loop

Reference:
    PMI PMBOK Guide, 6th Ed., Section 6.5 (Develop Schedule).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

# ─── Task schedule result ─────────────────────────────────────────────────────

@dataclass
class TaskSchedule:
    task_id: str
    duration: float
    es: float   # Early Start
    ef: float   # Early Finish
    ls: float   # Late Start
    lf: float   # Late Finish

    @property
    def total_float(self) -> float:
        return self.ls - self.es

    @property
    def is_critical(self) -> bool:
        return abs(self.total_float) < 1e-6


@dataclass
class CPMResult:
    """Full CPM schedule result."""
    project_duration: float
    tasks: dict[str, TaskSchedule]
    critical_path: list[str]

    @property
    def critical_task_ids(self) -> set[str]:
        return set(self.critical_path)

    def __str__(self) -> str:
        lines = [
            f"Project Duration: {self.project_duration:.1f} days",
            f"Critical Path: {' → '.join(self.critical_path)}",
            f"\n{'Task':<6} {'ES':>5} {'EF':>5} {'LS':>5} {'LF':>5} {'Float':>6} {'Crit'}",
            "─" * 40,
        ]
        for tid, s in sorted(self.tasks.items(), key=lambda x: x[1].es):
            lines.append(
                f"{tid:<6} {s.es:>5.1f} {s.ef:>5.1f} {s.ls:>5.1f} "
                f"{s.lf:>5.1f} {s.total_float:>6.1f}  {'★' if s.is_critical else ''}"
            )
        return "\n".join(lines)


# ─── CPM Engine ───────────────────────────────────────────────────────────────

class CPMScheduler:
    """
    Computes CPM schedule from a task list with dependency graph.

    Args:
        tasks: List of task dicts with keys: id, predecessors, and a duration value
        duration_key: key in each task dict to use as duration
                      (e.g. 'duration_most_likely' or a sampled value)
    """

    def __init__(
        self,
        tasks: list[dict],
        durations: dict[str, float] | None = None,
    ):
        self.tasks = {t["id"]: t for t in tasks}
        # Allow injecting sampled durations (for Monte Carlo)
        self.durations = durations or {
            t["id"]: t["duration_most_likely"] for t in tasks
        }
        self._build_graph()

    def _build_graph(self) -> None:
        """Build directed acyclic graph from task dependencies."""
        self.G = nx.DiGraph()
        for tid, task in self.tasks.items():
            self.G.add_node(tid, duration=self.durations.get(tid, 0))
            for pred in task.get("predecessors", []):
                self.G.add_edge(pred, tid)

        if not nx.is_directed_acyclic_graph(self.G):
            cycles = list(nx.simple_cycles(self.G))
            raise ValueError(f"Dependency graph contains cycles: {cycles}")

    def _forward_pass(self) -> dict[str, tuple[float, float]]:
        """Compute Early Start (ES) and Early Finish (EF) for each task."""
        es: dict[str, float] = {}
        ef: dict[str, float] = {}

        for tid in nx.topological_sort(self.G):
            dur = self.durations.get(tid, 0)
            preds = list(self.G.predecessors(tid))
            es[tid] = max((ef[p] for p in preds), default=0.0)
            ef[tid] = es[tid] + dur

        return es, ef

    def _backward_pass(
        self,
        es: dict[str, float],
        ef: dict[str, float],
        project_duration: float,
    ) -> dict[str, tuple[float, float]]:
        """Compute Late Start (LS) and Late Finish (LF) for each task."""
        ls: dict[str, float] = {}
        lf: dict[str, float] = {}

        for tid in reversed(list(nx.topological_sort(self.G))):
            dur = self.durations.get(tid, 0)
            succs = list(self.G.successors(tid))
            lf[tid] = min((ls[s] for s in succs), default=project_duration)
            ls[tid] = lf[tid] - dur

        return ls, lf

    def compute(self) -> CPMResult:
        """
        Run the full forward + backward pass.

        Returns:
            CPMResult with all task schedules and critical path.
        """
        es, ef = self._forward_pass()

        # Project duration = latest EF
        project_duration = max(ef.values()) if ef else 0.0

        ls, lf = self._backward_pass(es, ef, project_duration)

        # Build result
        schedules = {}
        for tid in self.tasks:
            schedules[tid] = TaskSchedule(
                task_id=tid,
                duration=self.durations.get(tid, 0),
                es=es[tid],
                ef=ef[tid],
                ls=ls[tid],
                lf=lf[tid],
            )

        # Critical path: chain of critical tasks from start to end
        critical_ids = {tid for tid, s in schedules.items() if s.is_critical}
        critical_path = self._extract_critical_path(critical_ids)

        return CPMResult(
            project_duration=project_duration,
            tasks=schedules,
            critical_path=critical_path,
        )

    def _extract_critical_path(self, critical_ids: set[str]) -> list[str]:
        """
        Extract the longest critical path from start to end.
        Returns ordered list of task IDs.
        """
        # Build subgraph of critical tasks
        critical_subgraph = self.G.subgraph(critical_ids)

        # Find source and sink nodes in the critical subgraph
        sources = [n for n in critical_subgraph.nodes if critical_subgraph.in_degree(n) == 0]
        sinks   = [n for n in critical_subgraph.nodes if critical_subgraph.out_degree(n) == 0]

        if not sources or not sinks:
            return sorted(critical_ids)

        # Find the longest path in the critical subgraph
        try:
            # Use longest path algorithm
            path = nx.dag_longest_path(critical_subgraph, weight=None)
            return path
        except Exception:
            return sorted(critical_ids)

    def gantt_data(self) -> list[dict]:
        """Return schedule data in a format suitable for Gantt charts."""
        result = self.compute()
        rows = []
        for tid, s in sorted(result.tasks.items(), key=lambda x: x[1].es):
            task = self.tasks[tid]
            rows.append({
                "id":       tid,
                "name":     task.get("name", tid),
                "phase":    task.get("phase", ""),
                "trade":    task.get("trade", ""),
                "start":    s.es,
                "finish":   s.ef,
                "duration": s.duration,
                "float":    s.total_float,
                "critical": s.is_critical,
            })
        return rows


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data.project_generator import generate_project

    project = generate_project("residential")
    tasks = [
        {"id": t.id, "name": t.name, "phase": t.phase, "trade": t.trade,
         "predecessors": t.predecessors,
         "duration_most_likely": t.duration_most_likely}
        for t in project.tasks
    ]
    scheduler = CPMScheduler(tasks)
    result = scheduler.compute()
    print(result)
