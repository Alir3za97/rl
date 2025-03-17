from scipy.stats import bernoulli

from n_armed_bandit.bandits.base import Bandit


class BernoulliBandit(Bandit):
    def __init__(self, n_arms: int, ps: list[float] = None):
        assert ps is None or len(ps) == n_arms

        self.n_arms = n_arms
        self.ps = ps

    def pull(self, arm: int) -> float:
        return bernoulli.rvs(self.ps[arm])
