from collections import defaultdict

from rl.core.alg import PolicyEvaluationAlgorithm
from rl.core.env import Environment
from rl.core.policy import Policy
from rl.core.runner import ParallelRunner
from rl.core.types import A, S


class FirstVisitMonteCarloPrediction(PolicyEvaluationAlgorithm[S, A]):
    """First-Visit Monte Carlo Prediction.

    This algorithm is used to estimate the value function of a policy.
    """

    def __init__(
        self,
        env: Environment[S, A],
        policy: Policy[S, A],
        discount_factor: float = 0.99,
        num_episodes: int = 1000,
        num_steps: int = 1000,
        n_workers: int = 1,
        exploring_starts: bool = False,
    ) -> None:
        """Initialize the first visit Monte Carlo algorithm.

        Args:
            env: The environment to run the algorithm on.
            policy: The policy to evaluate.
            discount_factor: The discount factor.
            num_episodes: The number of episodes to run the algorithm on.
            num_steps: The number of steps to run the algorithm on.
            n_workers: The number of workers to run the algorithm on.
            exploring_starts: Whether to use exploring starts.

        """
        self.env = env

        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.n_workers = n_workers
        self.exploring_starts = exploring_starts
        if self.exploring_starts:
            raise NotImplementedError("Exploring starts is not yet implemented.")

        self.policy: Policy[S, A] = policy
        self.value_function: dict[S, float] = {}
        self.q_function: dict[(S, A), float] = {}

        self.returns: dict[S, float] = defaultdict(float)
        self.returns_q_a: dict[(S, A), float] = defaultdict(float)
        self.num_first_visits: dict[S, int] = defaultdict(int)
        self.num_q_a_first_visits: dict[(S, A), int] = defaultdict(int)

    def run(self) -> Policy[S, A]:
        runner = ParallelRunner(
            env=self.env,
            policy=self.policy,
            n_workers=self.n_workers,
            num_episodes=self.num_episodes,
            max_steps=self.num_steps,
        )
        episodes = runner.run()

        for episode, _ in episodes:
            g = 0
            s_visit_counts = defaultdict(int)
            q_a_visit_counts = defaultdict(int)

            for transition in episode:
                s, a, _, _ = transition.to_q_tuple()
                s_visit_counts[s] += 1
                q_a_visit_counts[(s, a)] += 1

            for transition in reversed(episode):
                s, a, r, _ = transition.to_q_tuple()
                g = self.discount_factor * g + r
                if s_visit_counts[s] == 1:
                    self.returns[s] += g
                    self.num_first_visits[s] += 1
                if q_a_visit_counts[(s, a)] == 1:
                    self.returns_q_a[(s, a)] += g
                    self.num_q_a_first_visits[(s, a)] += 1

                s_visit_counts[s] -= 1
                q_a_visit_counts[(s, a)] -= 1

        for s in self.returns:
            self.value_function[s] = self.returns[s] / self.num_first_visits[s]
        for s, a in self.returns_q_a:
            self.q_function[(s, a)] = self.returns_q_a[(s, a)] / self.num_q_a_first_visits[(s, a)]

    def get_value_function(self) -> dict[S, float]:
        return self.value_function

    def get_q_function(self) -> dict[(S, A), float]:
        return self.q_function
