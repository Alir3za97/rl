from rl.core.alg import OfflineRLAlgorithm
from rl.core.env import ModelEnvironment
from rl.core.policy import ActionDistribution, Policy
from rl.core.types import A, S


class ValueIteration(OfflineRLAlgorithm[S, A]):
    """Value iteration algorithm for MDPs.

    Classic model-based dynamic programming algorithm for computing the optimal policy
    in a fully known Markov Decision Process (MDP) via iterative value updates.

    This implementation assumes access to a full model of the environment, including
    the state space, action space, and transition dynamics (probabilities and rewards).
    It repeatedly updates the value function using the Bellman optimality equation
    and derives a greedy deterministic policy based on the resulting value estimates.

    Attributes:
        env (ModelEnvironment): The MDP environment providing the transition model.
        discount_factor (float): Gamma; discount factor for future rewards.
        max_iterations (int): Maximum number of value update iterations.
        tolerance (float): Convergence threshold for value updates.
        value_function (dict[S, float]): Maps states to their current value estimates.
        policy (dict[S, ActionDistribution[A]]): Greedy policy derived from the value function.

    Methods:
        run() -> Policy[S, A]: Executes value iteration and returns the resulting greedy policy.

    """

    def __init__(
        self,
        env: ModelEnvironment,
        discount_factor: float = 0.9,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        """Initialize the value iteration algorithm.

        Args:
            env: The environment to run the algorithm on.
            discount_factor: The discount factor.
            max_iterations: The maximum number of iterations.
            tolerance: The tolerance for the value function.

        """
        self.env = env
        self.discount_factor = discount_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.value_function: dict[S, float] = {}
        self.policy: dict[S, ActionDistribution[A]] = {}

    def run(self) -> Policy[S, A]:
        """Run the value iteration algorithm."""
        for state in self.env.state_space:
            self.value_function[state] = 0

        for _ in range(self.max_iterations):
            delta = self._iterate()
            if delta < self.tolerance:
                break

        self._update_policy()

        return self.get_policy()

    def _iterate(self) -> float:
        """Perform one iteration of value iteration.

        Returns:
            The maximum absolute difference between the old and new value function.

        """
        delta = 0.0
        for state in self.env.state_space - self.env.terminal_states:
            old_val = self.value_function[state]
            self.value_function[state] = self._calculate_value(state)
            delta = max(delta, abs(old_val - self.value_function[state]))

        return delta

    def _calculate_value(self, state: S) -> float:
        """Calculate the value of a state."""
        best_value = float("-inf")

        for action in self.env.get_possible_actions(state):
            q_value = self._calculate_q_value(state, action)
            best_value = max(best_value, q_value)

        return best_value

    def _calculate_q_value(self, state: S, action: A) -> float:
        """Calculate the Q-value of a state-action pair."""
        transitions = self.env.get_possible_transitions(state, action)

        expected_action_value = 0.0
        for next_state, probability, reward in transitions:
            expected_action_value += probability * (
                reward + self.discount_factor * self.value_function[next_state]
            )

        return expected_action_value

    def _get_best_action(self, state: S) -> A:
        """Get the best action for a state."""
        return max(
            self.env.get_possible_actions(state),
            key=lambda action: self._calculate_q_value(state, action),
        )

    def _update_policy(self) -> None:
        for state in self.env.state_space - self.env.terminal_states:
            self.policy[state] = ActionDistribution(
                action_probabilities={
                    self._get_best_action(state): 1.0,
                },
            )

    def get_policy(self) -> Policy[S, A]:
        return self.policy
