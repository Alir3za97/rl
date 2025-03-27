from collections import defaultdict, deque

from tqdm import tqdm

from rl.core.alg import PolicyEvaluationAlgorithm
from rl.core.env import Environment
from rl.core.policy import Policy
from rl.core.types import A, S


class NStepTDPredictor(PolicyEvaluationAlgorithm[S, A]):
    """N-Step Temporal Difference Prediction Algorithm.

    This algorithm is used to estimate the value function of a policy.
    """

    def __init__(
        self,
        env: Environment[S, A],
        policy: Policy[S, A],
        n: int = 1,
        max_steps: int = 1000,
        num_episodes: int = 1000,
        alpha: float = 0.1,
        gamma: float = 0.99,
    ) -> None:
        """N-Step Temporal Difference Prediction Algorithm.

        Args:
            env: The environment to run the algorithm on.
            policy: The policy to use for the algorithm.
            n: The number of steps to look ahead.
            max_steps: The maximum number of steps to run the algorithm.
            num_episodes: The number of episodes to run the algorithm.
            alpha: The learning rate.
            gamma: The discount factor.

        """
        if n < 1 or n > max_steps:
            raise ValueError(f"n must be between 1 and {max_steps}")

        if alpha <= 0 or alpha > 1:
            raise ValueError("alpha must be between 0 and 1")

        if gamma <= 0 or gamma > 1:
            raise ValueError("gamma must be between 0 and 1")

        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.policy = policy
        self.env = env
        self.max_steps = max_steps
        self.num_episodes = num_episodes

        self.value_function: dict[S, float] = defaultdict(float)

    def run(self) -> None:
        for _ in tqdm(range(self.num_episodes), desc="Running N-Step TD Prediction"):
            self._process_episode()

    def _process_episode(self) -> None:
        """Simulate one episode and update the value function.

        G_t:n = R_t + γR_t+1 + ... + γ^n-1R_t+n + γ^nV(S_t+n+1)
        V(S_t) = V(S_t) + α(G_t:n - V(S_t))
        """
        state = self.env.reset()
        done = False
        n_step_trajectory: deque[tuple[S, float]] = deque()
        t = 0
        episode_end = self.max_steps
        while (
            (not done or len(n_step_trajectory) > 0)  # episode not over/there is states to update
            and t < self.max_steps
        ):
            t += 1
            if not done:
                action = self.policy[state].sample()
                next_state, reward, done = self.env.step(action)
                n_step_trajectory.append((state, reward))
                state = next_state

                if done:
                    episode_end = t
                    next_state = None

            if (not done and len(n_step_trajectory) == self.n) or (
                done and len(n_step_trajectory) > 0
            ):
                update_target_index = t - self.n + 1
                g = sum([reward * self.gamma**i for i, (_, reward) in enumerate(n_step_trajectory)])
                update_target_state, _ = n_step_trajectory.popleft()

                # bootstrap if episode does not end within n-step
                if (update_target_index + self.n) < episode_end:
                    g += self.gamma**self.n * self.value_function[next_state]

                self.value_function[update_target_state] += self.alpha * (
                    g - self.value_function.get(update_target_state, 0.0)
                )

    def get_value_function(self) -> dict[S, float]:
        return self.value_function
