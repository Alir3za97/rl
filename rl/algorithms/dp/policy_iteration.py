from rl.core.alg import OfflineRLAlgorithm
from rl.core.env import ModelEnvironment
from rl.core.policy import ActionDistribution, Policy
from rl.core.types import A, S


class PolicyIteration(OfflineRLAlgorithm[S, A]):
    """Policy iteration algorithm for MDPs.

    Classic model-based dynamic programming algorithm that alternates between
    policy evaluation and policy improvement to compute the optimal policy
    in a fully known Markov Decision Process (MDP).

    This algorithm requires full knowledge of the environment's transition dynamics.
    At each iteration:
        1. The value function is updated to reflect the expected return under the current policy.
        2. The policy is updated to be greedy w.r.t. the current value function.
    The process continues until the policy becomes stable.

    Attributes:
        env (ModelEnvironment): The MDP environment providing full transition/reward info.
        discount_factor (float): Gamma; the discount factor for future rewards.
        max_policy_eval_iterations (int): Maximum iterations for policy evaluation loop.
        eval_tolerance (float): Convergence threshold for value updates during evaluation.
        value_function (dict[S, float]): Maps states to their current estimated values.
        policy (dict[S, ActionDistribution[A]]): The current policy mapping states to actions.

    Methods:
        run() -> Policy[S, A]: Executes policy iteration and returns the resulting greedy policy.

    """

    def __init__(
        self,
        env: ModelEnvironment,
        discount_factor: float = 0.9,
        max_policy_eval_iterations: int = 1000,
        eval_tolerance: float = 1e-6,
    ) -> None:
        """Initialize the policy iteration algorithm.

        Args:
            env: The environment to run the algorithm on.
            discount_factor: The discount factor.
            max_policy_eval_iterations: Maximum iterations for policy evaluation.
            eval_tolerance: Tolerance for convergence during policy evaluation.

        """
        self.env = env
        self.discount_factor = discount_factor
        self.max_policy_eval_iterations = max_policy_eval_iterations
        self.eval_tolerance = eval_tolerance

        self.value_function: dict[S, float] = dict.fromkeys(self.env.state_space, 0.0)
        self.policy: dict[S, ActionDistribution[A]] = {}

    def run(self) -> Policy[S, A]:
        """Run the policy iteration algorithm."""
        self._initialize_policy()

        policy_stable = False
        while not policy_stable:
            self._policy_evaluation()
            policy_stable = self._policy_improvement()

        return self.get_policy()

    def _initialize_policy(self) -> None:
        for state in self.env.state_space - self.env.terminal_states:
            self.policy[state] = ActionDistribution(
                action_probabilities={
                    action: 1.0 / len(self.env.get_possible_actions(state))
                    for action in self.env.get_possible_actions(state)
                },
            )

    def _policy_evaluation(self) -> None:
        """Evaluate the current policy by updating the value function."""
        for _ in range(self.max_policy_eval_iterations):
            delta = 0.0
            for state in self.env.state_space - self.env.terminal_states:
                old_value = self.value_function[state]
                new_value = self._calculate_policy_value(state)
                self.value_function[state] = new_value
                delta = max(delta, abs(old_value - new_value))
            if delta < self.eval_tolerance:
                break

    def _calculate_policy_value(self, state: S) -> float:
        """Calculate the value of a state under the current policy."""
        action_distribution = self.policy.get(state)
        if not action_distribution or not action_distribution.action_probabilities:
            return 0.0
        value = 0.0
        for action, prob in action_distribution.action_probabilities.items():
            q_val = self._calculate_q_value(state, action)
            value += prob * q_val
        return value

    def _policy_improvement(self) -> bool:
        """Improve the current policy based on the updated value function.

        Returns:
            bool: True if the policy is stable (no changes), False otherwise.

        """
        policy_stable = True
        for state in self.env.state_space - self.env.terminal_states:
            old_distribution = self.policy[state]
            best_action = self._get_best_action(state)
            self.policy[state] = ActionDistribution(action_probabilities={best_action: 1.0})

            if old_distribution.action_probabilities != self.policy[state].action_probabilities:
                policy_stable = False

        return policy_stable

    def _get_best_action(self, state: S) -> A:
        """Get the best action for a state."""
        return max(
            self.env.get_possible_actions(state),
            key=lambda action: self._calculate_q_value(state, action),
        )

    def _calculate_q_value(self, state: S, action: A) -> float:
        """Calculate the Q-value of a state-action pair using the current value function."""
        q_value = 0.0
        for next_state, probability, reward in self.env.get_possible_transitions(state, action):
            q_value += probability * (
                reward + self.discount_factor * self.value_function[next_state]
            )
        return q_value

    def get_policy(self) -> Policy[S, A]:
        """Return the current policy."""
        return self.policy
