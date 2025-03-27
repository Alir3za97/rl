from collections import deque
from typing import Generic

from tqdm import tqdm

from rl.core.env import ModelEnvironment
from rl.core.policy import ActionDistribution, Policy
from rl.core.types import A, S


class NStepSarsaControl(Generic[S, A]):
    """N-Step Sarsa Control Algorithm."""

    def __init__(
        self,
        env: ModelEnvironment[S, A],
        n: int = 1,
        max_steps: int = 1000,
        num_episodes: int = 1000,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        gamma: float = 0.99,
        expected: bool = False,
    ) -> None:
        """N-Step Sarsa Prediction Algorithm.

        Args:
            env (ModelEnvironment[S, A]): Environment.
            n (int, optional): Number of td-steps. Defaults to 1.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to 1000.
            num_episodes (int, optional): Number of episodes to run. Defaults to 1000.
            epsilon (float, optional): Epsilon for epsilon-greedy policy. Defaults to 0.1.
            alpha (float, optional): Learning rate. Defaults to 0.1.
            gamma (float, optional): Discount factor. Defaults to 0.99.
            expected (bool, optional): Whether to use expected SARSA. Defaults to False.

        """
        self.env = env
        self.n = n
        self.max_steps = max_steps
        self.num_episodes = num_episodes
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.expected = expected
        self.q_values: dict[S, dict[A, float]] = {
            state: dict.fromkeys(self.env.action_space, 0) for state in self.env.state_space
        }
        self.policy: Policy[S, A] = {
            state: ActionDistribution(
                action_probabilities={
                    action: 1 / len(self.env.action_space) for action in self.env.action_space
                }
            )
            for state in self.env.state_space
        }
        for state in self.env.state_space:
            self._update_epsilon_greedy(state)

    def run(self) -> None:
        for _ in tqdm(range(self.num_episodes), desc="Running N-Step SARSA"):
            self._process_episode()

    def _process_episode(self) -> None:
        t = 0
        n_step_trajectory: deque[tuple[S, A, float, S, A]] = deque()
        done = False
        episode_end = self.max_steps

        state = self.env.reset()
        action = self.policy[state].sample()
        while (
            (not done or len(n_step_trajectory) > 0)  # episode not over/there is states to update
            and t < self.max_steps
        ):
            t += 1
            if not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.policy[next_state].sample()

                # update sarsa tuple
                n_step_trajectory.append((state, action, reward, next_state, next_action))

                state = next_state
                action = next_action

                if done:
                    episode_end = t
                    next_state = None
                    action = None

            if (not done and len(n_step_trajectory) == self.n) or (
                done and len(n_step_trajectory) > 0
            ):
                g = sum(
                    [
                        reward * self.gamma**i
                        for i, (_, _, reward, _, _) in enumerate(n_step_trajectory)
                    ]
                )
                update_target_index = t - self.n + 1

                _, _, _, next_state, next_action = n_step_trajectory[-1]
                update_target_state, update_target_action, _, _, _ = n_step_trajectory.popleft()

                # bootstrap if episode does not end within n-step
                if (update_target_index + self.n) < episode_end:
                    if self.expected:
                        g += self.gamma**self.n * self._get_state_value(next_state)
                    else:
                        g += self.gamma**self.n * self.q_values[next_state][next_action]

                self.q_values[update_target_state][update_target_action] += self.alpha * (
                    g - self.q_values[update_target_state][update_target_action]
                )

                self._update_epsilon_greedy(update_target_state)

    def _update_epsilon_greedy(self, state: S) -> None:
        best_action = max(self.q_values[state], key=self.q_values[state].get)
        num_actions = len(self.env.action_space)
        best_action_prob = 1 - self.epsilon + self.epsilon / num_actions
        other_action_prob = self.epsilon / num_actions
        self.policy[state] = ActionDistribution(
            action_probabilities={
                action: best_action_prob if action == best_action else other_action_prob
                for action in self.env.action_space
            }
        )

    def _get_state_value(self, state: S) -> float:
        return sum(
            self.policy[state].get_action_probability(action) * self.q_values[state][action]
            for action in self.env.action_space
        )
