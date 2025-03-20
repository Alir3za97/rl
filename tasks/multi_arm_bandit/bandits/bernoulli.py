import numpy as np
from scipy.stats import bernoulli

from multi_arm_bandit.bandits.base import Bandit


class BernoulliBandit(Bandit):
    def __init__(self, n_arms: int, ps: list[float] | None = None) -> None:
        """Initialize the Bernoulli Bandit.

        Args:
            n_arms: Number of arms.
            ps: List of reward probabilities for each arm.

        """
        if ps is not None and len(ps) != n_arms:
            raise ValueError("ps must be a list of length n_arms")

        self.n_arms = n_arms
        self.ps = ps

    def pull(self, arm: int) -> float:
        return bernoulli.rvs(self.ps[arm])

    def reset(self) -> None:
        pass


class NonStationaryBernoulliBandit(BernoulliBandit):
    """Non-stationary Bernoulli bandit.

    The reward probabilities are perturbed by a Gaussian random walk.

    Args:
        n_arms: Number of arms.
        ps: List of reward probabilities for each arm.
        step_std: Standard deviation of the Gaussian random walk.
        walk_every: Number of steps between walks.

    """

    def __init__(
        self,
        n_arms: int,
        ps: list[float] | None = None,
        step_std: float = 0.1,
        walk_every: int = 100,
    ) -> None:
        """Initialize the Non-stationary Bernoulli Bandit.

        Args:
            n_arms: Number of arms.
            ps: List of reward probabilities for each arm.
            step_std: Standard deviation of the Gaussian random walk.
            walk_every: Number of steps between walks.

        """
        super().__init__(n_arms, ps)
        self.original_ps = self.ps.copy()
        self.step_std = step_std
        self.walk_every = walk_every
        self.t = 0

    def pull(self, arm: int) -> float:
        if self.walk_every and self.t % self.walk_every == 0:
            self.ps = [np.clip(p + np.random.normal(0, self.step_std), 0, 1) for p in self.ps]

        self.t += 1
        return super().pull(arm)

    def reset(self) -> None:
        self.ps = self.original_ps.copy()
        self.t = 0
