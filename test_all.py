"""
Test suite — PERT, CPM, Monte Carlo
"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schedule.pert import PERTEstimate, pert_from_task
from src.schedule.cpm import CPMScheduler, CPMResult
from src.risk.monte_carlo import MonteCarloEngine
from src.data.project_generator import generate_project, Task


# ═══ PERT TESTS ══════════════════════════════════════════════════════════════

class TestPERTEstimate:

    def test_mean_formula(self):
        p = PERTEstimate(10, 15, 25)
        expected = (10 + 4*15 + 25) / 6
        assert abs(p.mean - expected) < 1e-10

    def test_std_formula(self):
        p = PERTEstimate(10, 15, 25)
        expected = (25 - 10) / 6
        assert abs(p.std - expected) < 1e-10

    def test_mean_between_o_and_p(self):
        for o,m,p in [(5,10,20),(1,1,10),(3,8,9)]:
            pert = PERTEstimate(o,m,p)
            assert o <= pert.mean <= p

    def test_invalid_order_raises(self):
        with pytest.raises(ValueError):
            PERTEstimate(20, 10, 5)

    def test_degenerate_raises(self):
        with pytest.raises(ValueError):
            PERTEstimate(10, 10, 10)

    def test_sample_shape(self):
        p = PERTEstimate(5, 10, 20)
        samples = p.sample(1000, rng=np.random.default_rng(0))
        assert samples.shape == (1000,)

    def test_samples_within_bounds(self):
        p = PERTEstimate(5, 10, 20)
        samples = p.sample(5000, rng=np.random.default_rng(1))
        assert samples.min() >= 5 - 0.001
        assert samples.max() <= 20 + 0.001

    def test_sample_mean_close_to_pert_mean(self):
        p = PERTEstimate(10, 15, 25)
        samples = p.sample(50000, rng=np.random.default_rng(2))
        assert abs(samples.mean() - p.mean) < 0.1

    def test_cdf_at_o_is_zero(self):
        p = PERTEstimate(5, 10, 20)
        assert p.cdf(5) == 0.0

    def test_cdf_at_p_is_one(self):
        p = PERTEstimate(5, 10, 20)
        assert p.cdf(20) == 1.0

    def test_cdf_monotone(self):
        p = PERTEstimate(5, 10, 20)
        xs = np.linspace(5, 20, 20)
        cdfs = [p.cdf(x) for x in xs]
        assert all(a <= b for a,b in zip(cdfs, cdfs[1:]))

    def test_percentile_p50_near_mode_for_symmetric(self):
        p = PERTEstimate(5, 10, 15)   # symmetric: O+P=2*M
        # For symmetric PERT, median ≈ mode
        assert abs(p.percentile(0.50) - 10) < 1.5

    def test_percentile_p10_above_o(self):
        p = PERTEstimate(5, 10, 20)
        assert p.percentile(0.10) > 5

    def test_percentile_p90_below_p(self):
        p = PERTEstimate(5, 10, 20)
        assert p.percentile(0.90) < 20

    def test_reproducibility(self):
        p = PERTEstimate(5, 10, 20)
        s1 = p.sample(100, rng=np.random.default_rng(99))
        s2 = p.sample(100, rng=np.random.default_rng(99))
        assert np.allclose(s1, s2)

    def test_pert_from_task_duration(self):
        task = {"duration_optimistic": 5, "duration_most_likely": 10, "duration_pessimistic": 20}
        p = pert_from_task(task, "duration")
        assert abs(p.mean - PERTEstimate(5,10,20).mean) < 1e-10

    def test_pert_from_task_cost(self):
        task = {"cost_optimistic": 1000, "cost_most_likely": 1500, "cost_pessimistic": 2500}
        p = pert_from_task(task, "cost")
        assert abs(p.mean - PERTEstimate(1000,1500,2500).mean) < 1e-10

    def test_skewed_pessimistic(self):
        """Pessimistically skewed estimate should have mean > mode."""
        p = PERTEstimate(5, 8, 25)
        assert p.mean > p.most_likely


# ═══ CPM TESTS ═══════════════════════════════════════════════════════════════

def simple_tasks(durations=None):
    """Simple 4-task sequence: T1 → T2 → T3, T2 → T4"""
    tasks = [
        {"id":"T1","name":"Task 1","phase":"A","trade":"X","predecessors":[],"duration_most_likely":10},
        {"id":"T2","name":"Task 2","phase":"A","trade":"X","predecessors":["T1"],"duration_most_likely":5},
        {"id":"T3","name":"Task 3","phase":"A","trade":"X","predecessors":["T2"],"duration_most_likely":8},
        {"id":"T4","name":"Task 4","phase":"A","trade":"X","predecessors":["T2"],"duration_most_likely":3},
    ]
    d = durations or {t["id"]: t["duration_most_likely"] for t in tasks}
    return tasks, d


class TestCPMScheduler:

    def test_project_duration_sequential(self):
        """T1(10) → T2(5) → T3(8): duration = 23"""
        tasks = [
            {"id":"A","name":"A","phase":"X","trade":"X","predecessors":[],"duration_most_likely":10},
            {"id":"B","name":"B","phase":"X","trade":"X","predecessors":["A"],"duration_most_likely":5},
            {"id":"C","name":"C","phase":"X","trade":"X","predecessors":["B"],"duration_most_likely":8},
        ]
        s = CPMScheduler(tasks)
        result = s.compute()
        assert abs(result.project_duration - 23) < 1e-6

    def test_parallel_tasks_take_longer_path(self):
        """T1 → T2(5) and T1 → T3(10): project = T1 + T3"""
        tasks = [
            {"id":"T1","name":"T1","phase":"X","trade":"X","predecessors":[],"duration_most_likely":5},
            {"id":"T2","name":"T2","phase":"X","trade":"X","predecessors":["T1"],"duration_most_likely":3},
            {"id":"T3","name":"T3","phase":"X","trade":"X","predecessors":["T1"],"duration_most_likely":8},
        ]
        s = CPMScheduler(tasks)
        result = s.compute()
        assert abs(result.project_duration - 13) < 1e-6

    def test_critical_path_has_zero_float(self):
        tasks, d = simple_tasks()
        s = CPMScheduler(tasks, durations=d)
        result = s.compute()
        for tid in result.critical_path:
            assert result.tasks[tid].is_critical

    def test_non_critical_tasks_have_positive_float(self):
        tasks, d = simple_tasks()
        s = CPMScheduler(tasks, durations=d)
        result = s.compute()
        non_critical = {tid for tid,s_ in result.tasks.items() if not s_.is_critical}
        for tid in non_critical:
            assert result.tasks[tid].total_float > 0

    def test_es_ef_consistency(self):
        tasks, d = simple_tasks()
        result = CPMScheduler(tasks, durations=d).compute()
        for tid, s in result.tasks.items():
            assert abs(s.ef - s.es - s.duration) < 1e-6

    def test_ls_lf_consistency(self):
        tasks, d = simple_tasks()
        result = CPMScheduler(tasks, durations=d).compute()
        for tid, s in result.tasks.items():
            assert abs(s.lf - s.ls - s.duration) < 1e-6

    def test_returns_cpm_result(self):
        tasks, d = simple_tasks()
        result = CPMScheduler(tasks, durations=d).compute()
        assert isinstance(result, CPMResult)

    def test_all_tasks_in_result(self):
        tasks, d = simple_tasks()
        result = CPMScheduler(tasks, durations=d).compute()
        assert set(result.tasks.keys()) == {"T1","T2","T3","T4"}

    def test_custom_durations(self):
        tasks, _ = simple_tasks()
        custom = {"T1":20,"T2":10,"T3":15,"T4":3}
        result = CPMScheduler(tasks, durations=custom).compute()
        assert abs(result.project_duration - 45) < 1e-6

    def test_gantt_data_nonempty(self):
        tasks, d = simple_tasks()
        gantt = CPMScheduler(tasks, durations=d).gantt_data()
        assert len(gantt) == 4

    def test_residential_project_computes(self):
        project = generate_project("residential")
        task_dicts = [
            {"id": t.id, "name": t.name, "phase": t.phase, "trade": t.trade,
             "predecessors": t.predecessors, "duration_most_likely": t.duration_most_likely}
            for t in project.tasks
        ]
        result = CPMScheduler(task_dicts).compute()
        assert result.project_duration > 0
        assert len(result.critical_path) > 0


# ═══ MONTE CARLO TESTS ═══════════════════════════════════════════════════════

def make_mc_engine(project_type="residential", n_sim=100):
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
    return MonteCarloEngine(
        {"tasks": task_list, "deadline_days": project.deadline_days,
         "budget": project.budget, "name": project.name},
        n_simulations=n_sim, random_seed=42
    )


class TestMonteCarloEngine:

    def setup_method(self):
        self.engine = make_mc_engine(n_sim=200)
        self.result = self.engine.run()

    def test_durations_shape(self):
        assert self.result.durations.shape == (200,)

    def test_costs_shape(self):
        assert self.result.costs.shape == (200,)

    def test_all_durations_positive(self):
        assert (self.result.durations > 0).all()

    def test_all_costs_positive(self):
        assert (self.result.costs > 0).all()

    def test_p50_gt_p10(self):
        assert self.result.duration_percentile(0.50) >= self.result.duration_percentile(0.10)

    def test_p90_gt_p50(self):
        assert self.result.duration_percentile(0.90) >= self.result.duration_percentile(0.50)

    def test_p95_gt_p50(self):
        """P95 duration must be greater than P50 duration."""
        assert self.result.duration_percentile(0.95) >= self.result.duration_percentile(0.50)

    def test_criticality_sums_to_positive(self):
        total = sum(self.result.criticality.values())
        assert total > 0

    def test_criticality_in_0_1(self):
        for tid, ci in self.result.criticality.items():
            assert 0.0 <= ci <= 1.0, f"{tid}: criticality={ci}"

    def test_cost_var_positive(self):
        assert self.result.cost_var(0.95) > 0

    def test_reproducibility(self):
        r1 = make_mc_engine(n_sim=50).run()
        r2 = make_mc_engine(n_sim=50).run()
        assert np.allclose(r1.durations, r2.durations)

    def test_summary_keys(self):
        s = self.result.summary()
        for key in ["p50_duration","p90_duration","p95_duration",
                    "cost_var_95_eur","cost_var_95_pct"]:
            assert key in s

    def test_top_critical_tasks_sorted(self):
        tops = self.result.top_critical_tasks(3)
        vals = [v for _, v in tops]
        assert vals == sorted(vals, reverse=True)


class TestProjectGenerator:

    def test_all_types_generate(self):
        for t in ["residential","commercial","infrastructure"]:
            p = generate_project(t)
            assert len(p.tasks) > 0

    def test_pert_means_positive(self):
        for t_type in ["residential","commercial"]:
            p = generate_project(t_type)
            for t in p.tasks:
                assert t.duration_pert > 0
                assert t.cost_pert > 0

    def test_task_order_valid(self):
        """Predecessors must be defined before dependents."""
        p = generate_project("residential")
        defined = set()
        for t in p.tasks:
            for pred in t.predecessors:
                assert pred in defined, f"{t.id} references undefined predecessor {pred}"
            defined.add(t.id)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
