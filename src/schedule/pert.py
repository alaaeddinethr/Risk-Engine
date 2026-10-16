"""
pert.py
-------
Beta-PERT distribution for task duration modelling.

The Beta-PERT distribution is the standard in project risk analysis (PMI, PMBOK).
It fits a scaled Beta distribution to the 3-point estimate:
  Optimistic (O) — best case, ~5th percentile
  Most Likely (M) — mode of the distribution
  Pessimistic (P) — worst case, ~95th percentile

Parameters:
  μ = (O + λM + P) / (λ + 2)       where λ=4 is the standard weight
  σ = (P - O) / (λ + 2)

The underlying Beta(α, β) parameters are fitted to match μ and σ.

Why Beta-PERT over Normal?
  - Bounded: durations can't be negative or infinite
  - Skewed: construction tasks are asymmetrically risky (delays more likely than early finish)
  - Mode is directly specified (intuitive for engineers)

Reference:
    Malcolm, D.G. et al. (1959). Application of a Technique for R&D Program Evaluation.
    Operations Research, 7(5), 646–669.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PERTEstimate:
    """Three-point estimate with computed PERT parameters."""
    optimistic: float
    most_likely: float
    pessimistic: float
    lambda_: float = 4.0   # weight of most-likely estimate

    def __post_init__(self):
        if not (self.optimistic <= self.most_likely <= self.pessimistic):
            raise ValueError(
                f"Must have O ≤ M ≤ P, got O={self.optimistic}, "
                f"M={self.most_likely}, P={self.pessimistic}"
            )
        if self.pessimistic == self.optimistic:
            raise ValueError("O and P must be different (non-degenerate distribution)")

    @property
    def mean(self) -> float:
        """PERT mean (μ_PERT)."""
        return (self.optimistic + self.lambda_ * self.most_likely + self.pessimistic) / (self.lambda_ + 2)

    @property
    def std(self) -> float:
        """PERT standard deviation (σ_PERT)."""
        return (self.pessimistic - self.optimistic) / (self.lambda_ + 2)

    @property
    def variance(self) -> float:
        return self.std ** 2

    @property
    def alpha(self) -> float:
        """Beta shape parameter α."""
        mu = self.mean
        sig2 = self.variance
        rng = self.pessimistic - self.optimistic
        return (mu - self.optimistic) / rng * (
            (mu - self.optimistic) * (self.pessimistic - mu) / sig2 - 1
        )

    @property
    def beta(self) -> float:
        """Beta shape parameter β."""
        mu = self.mean
        return self.alpha * (self.pessimistic - mu) / (mu - self.optimistic)

    def sample(
        self,
        n: int = 1,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """
        Draw n samples from the Beta-PERT distribution.

        Uses the parameterized Beta distribution scaled to [O, P].

        Args:
            n: Number of samples
            rng: NumPy random generator (for reproducibility)

        Returns:
            Array of shape (n,)
        """
        if rng is None:
            rng = np.random.default_rng()

        # Ensure valid Beta parameters
        a = max(self.alpha, 0.01)
        b = max(self.beta, 0.01)

        # Sample from Beta(α, β) then scale to [O, P]
        u = rng.beta(a, b, size=n)
        return self.optimistic + u * (self.pessimistic - self.optimistic)

    def cdf(self, x: float) -> float:
        """CDF at x — P(duration ≤ x)."""
        if x <= self.optimistic:
            return 0.0
        if x >= self.pessimistic:
            return 1.0
        a, b = max(self.alpha, 0.01), max(self.beta, 0.01)
        u = (x - self.optimistic) / (self.pessimistic - self.optimistic)
        return float(stats.beta.cdf(u, a, b))

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0 < p < 1)."""
        a, b = max(self.alpha, 0.01), max(self.beta, 0.01)
        u = stats.beta.ppf(p, a, b)
        return self.optimistic + u * (self.pessimistic - self.optimistic)

    def summary(self) -> dict:
        return {
            "optimistic":  self.optimistic,
            "most_likely": self.most_likely,
            "pessimistic": self.pessimistic,
            "mean":        round(self.mean, 3),
            "std":         round(self.std, 3),
            "p10":         round(self.percentile(0.10), 3),
            "p50":         round(self.percentile(0.50), 3),
            "p90":         round(self.percentile(0.90), 3),
        }

    def __str__(self) -> str:
        return (
            f"PERT({self.optimistic}, {self.most_likely}, {self.pessimistic}) "
            f"→ μ={self.mean:.2f}, σ={self.std:.2f}"
        )


def pert_from_task(task_dict: dict, field: str = "duration") -> PERTEstimate:
    """
    Create a PERTEstimate from a task dictionary.

    Args:
        task_dict: dict with keys {field}_optimistic, _most_likely, _pessimistic
        field: 'duration' or 'cost'
    """
    return PERTEstimate(
        optimistic=task_dict[f"{field}_optimistic"],
        most_likely=task_dict[f"{field}_most_likely"],
        pessimistic=task_dict[f"{field}_pessimistic"],
    )


if __name__ == "__main__":
    # Example: concrete works task
    pert = PERTEstimate(optimistic=10, most_likely=15, pessimistic=25)
    print(pert)
    print("Summary:", pert.summary())

    rng = np.random.default_rng(42)
    samples = pert.sample(10000, rng=rng)
    print(f"Simulated mean: {samples.mean():.2f} (expected: {pert.mean:.2f})")
    print(f"Simulated std:  {samples.std():.2f}  (expected: {pert.std:.2f})")
    print(f"P(≤ 15 days):  {pert.cdf(15)*100:.1f}%")
    print(f"P90:            {pert.percentile(0.90):.1f} days")
