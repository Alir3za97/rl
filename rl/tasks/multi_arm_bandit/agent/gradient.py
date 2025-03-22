import numpy as np

from multi_arm_bandit.agent import Agent


class GradientAgent(Agent):
    def __init__(self, n_arms: int, alpha: float) -> None:
        """Initialize the Gradient Agent.

        Args:
            n_arms: Number of arms.
            alpha: The learning rate.

        """
        self.n_arms = n_arms
        self.alpha = alpha
        self.preferences = [0] * n_arms
        self.total_reward = 0
        self.total_steps = 0
        self.avg_reward = 0
        self.n_pulls = [0] * n_arms

    def select_arm(self) -> int:
        return np.random.choice(self.n_arms, p=self._get_policy())

    def _get_policy(self) -> np.ndarray:
        return np.exp(self.preferences) / np.sum(np.exp(self.preferences))

    def _update_preferences(self, selected_arm: int, reward: float) -> None:
        policy = self._get_policy()
        for i in range(self.n_arms):
            if i == selected_arm:
                self.preferences[i] += self.alpha * (reward - self.avg_reward) * (1 - policy[i])
            else:
                self.preferences[i] -= self.alpha * (reward - self.avg_reward) * policy[i]

    def observe(self, arm: int, reward: float) -> None:
        self.n_pulls[arm] += 1

        self.total_reward += reward
        self.total_steps += 1
        self.avg_reward = self.total_reward / self.total_steps

        self._update_preferences(arm, reward)

    def reset(self) -> None:
        self.preferences = [0] * self.n_arms
        self.total_reward = 0
        self.total_steps = 0
        self.avg_reward = 0
        self.n_pulls = [0] * self.n_arms
