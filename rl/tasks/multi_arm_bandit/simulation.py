from rl.tasks.multi_arm_bandit.agent.base import Agent
from rl.tasks.multi_arm_bandit.bandits.base import Bandit


class Simulation:
    def __init__(self, agent: Agent, bandit: Bandit) -> None:
        """Initialize the Simulation.

        Args:
            agent: The agent to use in the simulation.
            bandit: The bandit to use in the simulation.

        """
        self.agent = agent
        self.bandit = bandit

    def run(self, n_steps: int) -> list[float]:
        """
        Run the simulation for a given number of steps.

        Args:
            n_steps (int): The number of steps to run the simulation for.

        Returns:
            list[float]: The rewards for each step.

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
        self.bandit.reset()
