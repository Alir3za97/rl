from collections import defaultdict

from rl.algorithms.mc.visit import VisitType
from rl.core.alg import PolicyEvaluationAlgorithm
from rl.core.env import Environment
from rl.core.policy import Policy
from rl.core.runner import ParallelRunner
from rl.core.types import A, S, Transition


class ImportanceSamplingMonteCarloPredictor(PolicyEvaluationAlgorithm[S, A]):
    """Monte Carlo Prediction Algorithm with Importance Sampling.

    This algorithm evaluates a target policy while following a different behavior policy,
    using importance sampling to correct for the difference in action selection probabilities.
    This is useful for off-policy learning, where we want to evaluate or improve one policy
    while following another policy.

    The algorithm supports both ordinary importance sampling (unbiased but high variance)
    and weighted importance sampling (biased but lower variance).

    For each episode:
    1. Generate a complete trajectory by following the behavior policy
    2. Calculate importance sampling ratios for each step using the target and behavior policies
    3. Use these ratios to weight the returns for updating the value estimates

    The algorithm uses parallel processing to generate multiple episodes simultaneously,
    which can significantly speed up the learning process for computationally intensive
    environments.
    """

    def __init__(
        self,
        env: Environment[S, A],
        target_policy: Policy[S, A],
        behavior_policy: Policy[S, A],
        discount_factor: float = 0.99,
        num_episodes: int = 1000,
        max_steps: int = 1000,
        n_workers: int = 1,
        visit_type: VisitType = VisitType.FIRST_VISIT,
        weighted_is: bool = True,
        compute_value_function: bool = True,
        compute_q_function: bool = True,
    ) -> None:
        """Initialize the Importance Sampling Monte Carlo prediction algorithm.

        Args:
            env: The environment to run the algorithm on.
            target_policy: The policy to evaluate.
            behavior_policy: The policy used to generate episodes.
            discount_factor: The discount factor.
            num_episodes: The number of episodes to run the algorithm on.
            max_steps: The maximum number of steps in each generated episode.
            n_workers: The number of workers to run the algorithm on.
            visit_type: The type of visit to use (first_visit or every_visit).
            weighted_is: Whether to use weighted importance sampling (True) or
                         ordinary importance sampling (False).
            compute_value_function: Whether to compute the value function.
            compute_q_function: Whether to compute the Q-function.

        """
        self.env = env
        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.n_workers = n_workers
        self.visit_type = visit_type
        self.weighted_is = weighted_is
        self.compute_value_function = compute_value_function
        self.compute_q_function = compute_q_function

        self.target_policy: Policy[S, A] = target_policy
        self.behavior_policy: Policy[S, A] = behavior_policy

        # State value function
        self.value_function: dict[S, float] = {}
        self.v_numerator: dict[S, float] = defaultdict(float)
        self.v_denominator: dict[S, float] = defaultdict(float)

        # State-action value function (Q)
        self.q_function: dict[tuple[S, A], float] = {}
        self.q_numerator: dict[tuple[S, A], float] = defaultdict(float)
        self.q_denominator: dict[tuple[S, A], float] = defaultdict(float)

    def run(self) -> Policy[S, A]:
        runner = ParallelRunner(
            env=self.env,
            policy=self.behavior_policy,  # Use behavior policy to generate episodes
            n_workers=self.n_workers,
            num_episodes=self.num_episodes,
            max_steps=self.max_steps,
        )
        episodes = runner.run()

        for episode, _ in episodes:
            self._process_episode(episode)

        return self.target_policy

    def _process_episode(self, episode: list[Transition[S, A]]) -> None:
        returns = self._compute_episode_returns(episode)
        importance_ratios = self._compute_importance_ratios(episode)

        visited_states, visited_state_actions = set(), set()

        for i, (transition, g) in enumerate(zip(episode, returns, strict=True)):
            s, a, _, _ = transition.to_q_tuple()

            # Calculate the cumulative importance ratio up to this point
            cumulative_ratio = 1.0
            for j in range(i, len(episode)):
                cumulative_ratio *= importance_ratios[j]

            if self.compute_value_function:
                visited_states = self._process_value_step(s, g, cumulative_ratio, visited_states)

            if self.compute_q_function:
                visited_state_actions = self._process_q_step(
                    s, a, g, cumulative_ratio, visited_state_actions
                )

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

    def _compute_importance_ratios(self, episode: list[Transition[S, A]]) -> list[float]:
        """Compute the importance sampling ratios for each step in the episode."""
        ratios = []

        for transition in episode:
            s, a, _, _ = transition.to_q_tuple()

            # Get probabilities from both policies
            target_prob = self.target_policy[s].get_action_probability(a)
            behavior_prob = self.behavior_policy[s].get_action_probability(a)

            ratio = target_prob / behavior_prob if behavior_prob > 0 else 0.0
            ratios.append(ratio)

        return ratios

    def _process_value_step(
        self, s: S, g: float, weight: float, visited_states: set[S] | None = None
    ) -> set[S]:
        if visited_states is None:
            visited_states = set()

        is_first_state_visit = s not in visited_states
        if is_first_state_visit:
            visited_states.add(s)

        if self.visit_type == VisitType.FIRST_VISIT and not is_first_state_visit:
            return visited_states

        if self.weighted_is:
            # Weighted importance sampling
            self.v_numerator[s] += weight * g
            self.v_denominator[s] += weight

            if self.v_denominator[s] > 0:
                self.value_function[s] = self.v_numerator[s] / self.v_denominator[s]
        else:
            # Ordinary importance sampling
            if s not in self.value_function:
                self.value_function[s] = 0
                self.v_denominator[s] = 0

            self.v_denominator[s] += 1
            self.value_function[s] += (weight * g - self.value_function[s]) / self.v_denominator[s]

        return visited_states

    def _process_q_step(
        self,
        s: S,
        a: A,
        g: float,
        weight: float,
        visited_state_actions: set[tuple[S, A]] | None = None,
    ) -> set[tuple[S, A]]:
        if visited_state_actions is None:
            visited_state_actions = set()

        state_action = (s, a)

        is_first_sa_visit = state_action not in visited_state_actions
        if is_first_sa_visit:
            visited_state_actions.add(state_action)

        if self.visit_type == VisitType.FIRST_VISIT and not is_first_sa_visit:
            return visited_state_actions

        if self.weighted_is:
            # Weighted importance sampling
            self.q_numerator[state_action] += weight * g
            self.q_denominator[state_action] += weight

            if self.q_denominator[state_action] > 0:
                self.q_function[state_action] = (
                    self.q_numerator[state_action] / self.q_denominator[state_action]
                )
        else:
            # Ordinary importance sampling
            if state_action not in self.q_function:
                self.q_function[state_action] = 0
                self.q_denominator[state_action] = 0

            self.q_denominator[state_action] += 1
            self.q_function[state_action] += (
                weight * g - self.q_function[state_action]
            ) / self.q_denominator[state_action]

        return visited_state_actions

    def get_value_function(self) -> dict[S, float]:
        if not self.compute_value_function:
            raise ValueError("Value function not computed, set compute_value_function=True")

        return self.value_function

    def get_q_function(self) -> dict[tuple[S, A], float]:
        if not self.compute_q_function:
            raise ValueError("Q-function not computed, set compute_q_function=True")

        return self.q_function
