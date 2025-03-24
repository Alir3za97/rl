import random

from rl.tasks.multi_arm_bandit.agent.base import Agent


class RandomAgent(Agent):
    def __init__(self, n_arms: int) -> None:
        """Initialize the Random Agent.

        Args:
            n_arms: Number of arms.

        """
        self.n_arms = n_arms

    def select_arm(self) -> int:
        return random.randint(0, self.n_arms - 1)

    def observe(self, arm: int, reward: float) -> None:
        pass

    def reset(self) -> None:
        pass

    def copy(self) -> "RandomAgent":
        return RandomAgent(self.n_arms)
