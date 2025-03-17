from typing import List

from n_armed_bandit.agent.base import Agent
from n_armed_bandit.bandits.base import Bandit


class Simulation:
    def __init__(self, agent: Agent, bandit: Bandit):
        self.agent = agent
        self.bandit = bandit

    def run(self, n_steps: int) -> List[float]:
        """
        Run the simulation for a given number of steps.

        Args:
            n_steps (int): The number of steps to run the simulation for.

        Returns:
            List[float]: The rewards for each step.
        """

        self.reset()

        rewards = []

        for _ in range(n_steps):
            arm = self.agent.select_arm()
            reward = self.bandit.pull(arm)
            self.agent.observe(arm, reward)
            rewards.append(reward)

        return rewards

    def reset(self) -> None:
        self.agent.reset()
