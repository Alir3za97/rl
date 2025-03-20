import numpy as np

from multi_arm_bandit.agent.base import Agent


class UCBAgent(Agent):
    def __init__(self, n_arms: int, c: float) -> None:
        """Initialize the UCB Agent.

        Args:
            n_arms: Number of arms.
            c: The exploration parameter.

        """
        self.n_arms = n_arms
        self.c = c
        self.q_values = [0] * n_arms
        self.n_pulls = [0] * n_arms
        self.n_actions = 0

    def select_arm(self) -> int:
        if self.n_actions < self.n_arms:
            return self.n_actions
        return np.argmax(self.q_values + self.c * np.sqrt(np.log(self.n_actions) / self.n_pulls))

    def observe(self, arm: int, reward: float) -> None:
        self.n_actions += 1
        self.n_pulls[arm] += 1
        self.q_values[arm] = (self.q_values[arm] * (self.n_pulls[arm] - 1) + reward) / self.n_pulls[
            arm
        ]

    def reset(self) -> None:
        self.q_values = [0] * self.n_arms
        self.n_pulls = [0] * self.n_arms
        self.n_actions = 0
