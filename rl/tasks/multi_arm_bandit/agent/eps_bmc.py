import numpy as np
from scipy import stats

from rl.tasks.multi_arm_bandit.agent.base import Agent


class EpsilonBMCAgent(Agent):
    def __init__(
        self,
        num_arms: int,
        eps_alpha: float = 1.0,
        eps_beta: float = 1.0,
        lambda_: float = 1.0,
        epsilon_min: float = 0.01,
    ) -> None:
        """Initialize the Epsilon-BMC Agent.

        Epsilon-BMC: Adaptive Bayesian epsilon-greedy agent.
        https://arxiv.org/abs/2007.00869

        Key equations:
        - ε ~ Beta(α, β) with α,β updated via Bayes factor
        - Reward dist: Normal(mu, 1/(kappa * tau)) where tau ~ Gamma(alpha, beta)
        - Bayesian updates:
            * kappa_new = kappa_old + 1
            * mu_new = (kappa_old*mu_old + reward)/kappa_new
            * alpha_new = alpha_old + 0.5
            * beta_new = beta_old + 0.5*(kappa_old/kappa_new)*(reward - mu_old)^2
        - Variance estimate: beta/(alpha * (kappa/(kappa+1)))

        Args:
            num_arms (int): Number of arms in the bandit.
            eps_alpha (float): Initial alpha for Beta(α,β) prior over ε
            eps_beta (float): Initial beta for Beta(α,β) prior over ε
            lambda_ (float): Initial prior precision (inverse variance)
                for Normal-Gamma reward estimation
            epsilon_min (float): Minimum exploration probability

        """
        self.init_args = {
            "num_arms": num_arms,
            "eps_alpha": eps_alpha,
            "eps_beta": eps_beta,
            "lambda_": lambda_,
            "epsilon_min": epsilon_min,
        }
        self.num_arms = num_arms
        self.eps_alpha = eps_alpha
        self.eps_beta = eps_beta
        self.epsilon_min = epsilon_min

        # Normal-Gamma parameters (μ, κ, α, β)
        self.mu = np.zeros(num_arms)
        self.kappa = np.ones(num_arms) * lambda_
        self.alpha = np.ones(num_arms) * 1.0
        self.beta = np.ones(num_arms) * 1.0

        self.mean_reward = np.zeros(num_arms)
        self.var_reward = np.ones(num_arms)
        self.n_pulls = np.zeros(num_arms)

    def select_arm(self) -> int:
        if np.random.rand() < self._get_epsilon():
            return np.random.randint(self.num_arms)
        return np.argmax(self.mean_reward)

    def _get_epsilon(self) -> float:
        sampled_eps = np.random.beta(self.eps_alpha, self.eps_beta)
        return max(sampled_eps, self.epsilon_min)

    def observe(self, arm: int, reward: float) -> None:
        self.n_pulls[arm] += 1
        self._update_arm_parameters(arm, reward)
        self._update_posterior_estimates(arm)
        self._update_beta(reward, arm)

    def _update_arm_parameters(self, arm: int, reward: float) -> None:
        prev_kappa = self.kappa[arm]
        prev_mu = self.mu[arm]
        prev_alpha = self.alpha[arm]
        prev_beta = self.beta[arm]

        new_kappa = prev_kappa + 1
        new_mu = (prev_kappa * prev_mu + reward) / new_kappa
        new_alpha = prev_alpha + 0.5
        new_beta = prev_beta + 0.5 * prev_kappa / new_kappa * (reward - prev_mu) ** 2

        self.kappa[arm] = new_kappa
        self.mu[arm] = new_mu
        self.alpha[arm] = new_alpha
        self.beta[arm] = new_beta

    def _update_posterior_estimates(self, arm: int) -> None:
        self.mean_reward[arm] = self.mu[arm]

        var_numerator = self.beta[arm]
        var_denominator = self.alpha[arm] * (self.kappa[arm] / (self.kappa[arm] + 1))
        self.var_reward[arm] = max(var_numerator / var_denominator, 1e-6)

    def _update_beta(self, reward: float, arm: int) -> None:
        current_log_prob = self._reward_log_prob(reward, arm)
        perturbed_log_prob = self._reward_log_prob(reward, arm, perturb=True)

        log_bayes_factor = perturbed_log_prob - current_log_prob
        self.eps_alpha += np.exp(log_bayes_factor)
        self.eps_beta += 1

    def _reward_log_prob(self, reward: float, arm: int, perturb: bool = False) -> float:
        nu = 2 * self.alpha[arm]
        loc = self.mu[arm]
        scale = np.sqrt(
            self.beta[arm] * (self.kappa[arm] + 1) / (self.alpha[arm] * self.kappa[arm])
        )

        if perturb:
            # Perturbation with scale-proportional noise
            loc += np.random.normal(0, scale)

        return stats.t(df=nu, loc=loc, scale=scale).logpdf(reward)

    def reset(self) -> None:
        self.eps_alpha = self.init_args["eps_alpha"]
        self.eps_beta = self.init_args["eps_beta"]
        self.lambda_ = self.init_args["lambda_"]
        self.epsilon_min = self.init_args["epsilon_min"]

        self.kappa = np.ones(self.num_arms) * self.lambda_
        self.mu = np.zeros(self.num_arms)
        self.alpha = np.ones(self.num_arms) * 1.0
        self.beta = np.ones(self.num_arms) * 1.0

        self.mean_reward = np.zeros(self.num_arms)
        self.var_reward = np.ones(self.num_arms)
        self.n_pulls = np.zeros(self.num_arms)

    def copy(self) -> "EpsilonBMCAgent":
        return EpsilonBMCAgent(**self.init_args)
