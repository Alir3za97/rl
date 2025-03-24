import numpy as np
from scipy.stats import beta

from rl.tasks.multi_arm_bandit.agent.base import Agent


class ThompsonSamplingAgent(Agent):
    def __init__(self, n_arms: int, alpha: float = 1.0, beta_: float = 1.0) -> None:
        """Initialize the Thompson Sampling Agent.

        Thompson Sampling is a Bayesian algorithm that models the reward probability
        of each arm as a Beta distribution. For Bernoulli bandits, this is
        the conjugate prior, making updates straightforward.

        Args:
            n_arms: Number of arms.
            alpha: Initial alpha parameter for Beta prior (default=1 for uniform prior)
            beta_: Initial beta parameter for Beta prior (default=1 for uniform prior)

        """
        self.n_arms = n_arms
        self.alpha_init = alpha
        self.beta_init = beta_

        # Beta distribution parameters (success, failure) for each arm
        self.alphas = np.ones(n_arms) * alpha
        self.betas = np.ones(n_arms) * beta_
        self.n_pulls = np.zeros(n_arms)

    def select_arm(self) -> int:
        """Select an arm according to Thompson Sampling strategy.

        Samples from each arm's Beta distribution and selects the arm
        with the highest sampled value.

        Returns:
            The selected arm index

        """
        samples = np.array([beta.rvs(a, b) for a, b in zip(self.alphas, self.betas, strict=True)])
        return np.argmax(samples)

    def observe(self, arm: int, reward: float) -> None:
        """Update the agent's belief based on the observed reward.

        For Bernoulli rewards (0 or 1), the Beta parameters are updated:
        - alpha += reward (successes)
        - beta += (1-reward) (failures)

        Args:
            arm: The arm that was pulled
            reward: The reward received (should be 0 or 1 for Bernoulli bandit)

        """
        self.n_pulls[arm] += 1

        # For Bernoulli bandits, reward is 0 or 1
        if reward == 1:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1

    def reset(self) -> None:
        """Reset the agent to its initial state."""
        self.alphas = np.ones(self.n_arms) * self.alpha_init
        self.betas = np.ones(self.n_arms) * self.beta_init
        self.n_pulls = np.zeros(self.n_arms)

    def copy(self) -> "ThompsonSamplingAgent":
        return ThompsonSamplingAgent(self.n_arms, self.alpha_init, self.beta_init)
