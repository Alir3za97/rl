import random

import numpy as np

from n_armed_bandit.agent.base import Agent


class EpsilonGreedyAgent(Agent):
    def __init__(self, n_arms: int, epsilon: float):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.q_values = [0] * n_arms
        self.n_pulls = [0] * n_arms
        self.n_actions = 0

    def select_arm(self) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, self.n_arms - 1)
        else:
            return np.argmax(self.q_values)

    def observe(self, arm: int, reward: float) -> None:
        self.n_actions += 1
        self.n_pulls[arm] += 1
        self.q_values[arm] = (self.q_values[arm] * (self.n_pulls[arm] - 1) + reward) / self.n_pulls[arm]

    def reset(self) -> None:
        self.q_values = [0] * self.n_arms
        self.n_pulls = [0] * self.n_arms
        self.n_actions = 0
