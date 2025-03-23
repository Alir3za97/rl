from rl.core.alg import ModelFreeRLAlgorithm
from rl.core.env import Environment
from rl.core.policy import Policy
from rl.core.runner import Runner
from rl.core.types import A, S


class FirstVisitMonteCarlo(ModelFreeRLAlgorithm[S, A]):
    def __init__(
        self,
        env: Environment[S, A],
        policy: Policy[S, A],
        discount_factor: float = 0.99,
        num_episodes: int = 1000,
        num_steps: int = 1000,
        n_workers: int = 1,
    ) -> None:
        """Initialize the first visit Monte Carlo algorithm.

        Args:
            env: The environment to run the algorithm on.
            policy: The policy to evaluate.
            discount_factor: The discount factor.
            num_episodes: The number of episodes to run the algorithm on.
            num_steps: The number of steps to run the algorithm on.
            n_workers: The number of workers to run the algorithm on.

        """
        super().__init__(env=env)

        self.discount_factor = discount_factor
        self.num_episodes = num_episodes
        self.num_steps = num_steps
        self.n_workers = n_workers

        self.policy: Policy[S, A] = policy
        self.value_function: dict[S, float] = {}

        self.returns: dict[S, float] = {}
        self.num_first_visits: dict[S, int] = {}

    def run(self) -> Policy[S, A]:
        runner = Runner(
            env=self.env,
            policy=self.policy,
            n_workers=self.n_workers,
            num_episodes=self.num_episodes,
            num_steps=self.num_steps,
        )

        runner.run()

    def get_policy(self) -> Policy[S, A]:
        return self.policy
