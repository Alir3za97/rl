from collections import defaultdict
from enum import Enum

from rl.core.alg import PolicyEvaluationAlgorithm
from rl.core.env import Environment
from rl.core.policy import Policy
from rl.core.runner import ParallelRunner
from rl.core.types import A, S, Transition


class VisitType(Enum):
    FIRST_VISIT = "first_visit"
    EVERY_VISIT = "every_visit"


class MonteCarloPredictor(PolicyEvaluationAlgorithm[S, A]):
    """Monte Carlo Prediction Algorithm.

    This algorithm is used to estimate the value function and Q-function of a policy.
    Supports both first-visit and every-visit Monte Carlo methods.

    The algorithm works by generating episodes using the given policy, then calculating
    returns for each state (and state-action pair) encountered in those episodes.

    For each episode:
    1. Generate a complete trajectory by following the policy
    2. Calculate the return (discounted sum of rewards) for each state visited
    3. Update the value estimates by averaging returns

    With first-visit MC, we only consider the first occurrence of each state in an episode.
    With every-visit MC, we consider every occurrence of each state.

    The algorithm uses parallel processing to generate multiple episodes simultaneously,
    which can significantly speed up the learning process for computationally intensive
    environments.
    """

    def __init__(
        self,
        env: Environment[S, A],
        policy: Policy[S, A],
        discount_factor: float = 0.99,
        num_episodes: int = 1000,
        max_steps: int = 1000,
        n_workers: int = 1,
        visit_type: VisitType = VisitType.FIRST_VISIT,
        compute_value_function: bool = True,
        compute_q_function: bool = True,
    ) -> None:
        """Initialize the Monte Carlo prediction algorithm.

        Args:
            env: The environment to run the algorithm on.
            policy: The policy to evaluate.
            discount_factor: The discount factor.
            num_episodes: The number of episodes to run the algorithm on.
            max_steps: The maximum number of steps in each generated episode.
            n_workers: The number of workers to run the algorithm on.
            visit_type: The type of visit to use (first_visit or every_visit).
            compute_value_function: Whether to compute the value function.
            compute_q_function: Whether to compute the Q-function.

        """
        self.env = env
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.n_workers = n_workers
        self.visit_type = visit_type
        self.compute_value_function = compute_value_function
        self.compute_q_function = compute_q_function

        self.policy: Policy[S, A] = policy

        # State value function
        self.value_function: dict[S, float] = {}
        self.returns_sum: dict[S, float] = defaultdict(float)
        self.returns_count: dict[S, int] = defaultdict(int)

        # State-action value function (Q)
        self.q_function: dict[tuple[S, A], float] = {}
        self.q_returns_sum: dict[tuple[S, A], float] = defaultdict(float)
        self.q_returns_count: dict[tuple[S, A], int] = defaultdict(int)

    def run(self) -> Policy[S, A]:
        runner = ParallelRunner(
            env=self.env,
            policy=self.policy,
            n_workers=self.n_workers,
            num_episodes=self.num_episodes,
            max_steps=self.max_steps,
        )
        episodes = runner.run()

        for episode, _ in episodes:
            self._process_episode(episode)

        return self.policy

    def _process_episode(self, episode: list[Transition[S, A]]) -> None:
        returns = self._compute_episode_returns(episode)

        visited_states, visited_state_actions = set(), set()

        for transition, g in zip(episode, returns, strict=True):
            s, a, _, _ = transition.to_q_tuple()

            if self.compute_value_function:
                visited_states = self._process_value_step(s, g, visited_states)

            if self.compute_q_function:
                visited_state_actions = self._process_q_step(s, a, g, visited_state_actions)

    def _compute_episode_returns(self, episode: list[Transition[S, A]]) -> list[float]:
        episode_length = len(episode)
        returns = [0.0] * episode_length

        g = 0.0
        for i in range(episode_length - 1, -1, -1):
            transition = episode[i]
            _, _, r, _ = transition.to_q_tuple()
            g = self.discount_factor * g + r
            returns[i] = g

        return returns

    def _process_value_step(self, s: S, g: float, visited_states: set[S] | None = None) -> set[S]:
        if visited_states is None:
            visited_states = set()

        is_first_state_visit = s not in visited_states
        if is_first_state_visit:
            visited_states.add(s)

        if self.visit_type == VisitType.FIRST_VISIT and not is_first_state_visit:
            return visited_states

        self.returns_sum[s] += g
        self.returns_count[s] += 1
        self.value_function[s] = self.returns_sum[s] / self.returns_count[s]

        return visited_states

    def _process_q_step(
        self, s: S, a: A, g: float, visited_state_actions: set[tuple[S, A]] | None = None
    ) -> set[tuple[S, A]]:
        if visited_state_actions is None:
            visited_state_actions = set()

        state_action = (s, a)

        is_first_sa_visit = state_action not in visited_state_actions
        if is_first_sa_visit:
            visited_state_actions.add(state_action)

        if self.visit_type == VisitType.FIRST_VISIT and not is_first_sa_visit:
            return visited_state_actions

        self.q_returns_sum[state_action] += g
        self.q_returns_count[state_action] += 1
        self.q_function[state_action] = (
            self.q_returns_sum[state_action] / self.q_returns_count[state_action]
        )

        return visited_state_actions

    def get_value_function(self) -> dict[S, float]:
        if not self.compute_value_function:
            raise ValueError("Value function not computed, set compute_value_function=True")

        return self.value_function

    def get_q_function(self) -> dict[tuple[S, A], float]:
        if not self.compute_q_function:
            raise ValueError("Q-function not computed, set compute_q_function=True")

        return self.q_function
