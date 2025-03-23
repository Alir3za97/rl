import multiprocessing as mp
from typing import Generic

from rl.core.env import Environment
from rl.core.policy import Policy
from rl.core.types import A, S, Transition


class Runner(Generic[S, A]):
    def __init__(
        self,
        env: Environment[S, A],
        policy: Policy[S, A],
        max_steps: int = 1000,
    ) -> None:
        """Initialize the runner with an environment and a policy.

        Args:
            env: The environment to run the algorithm on.
            policy: The policy to run.
            max_steps: The maximum number of steps to run the algorithm on.

        """
        self.env = env
        self.policy = policy
        self.max_steps = max_steps

    def collect_episode(self) -> tuple[list[Transition[S, A]], bool]:
        """Collect a trajectory from the environment.

        Returns:
            tuple[list[Transition[S, A]], truncated]

        """
        state = self.env.reset()
        transitions: list[Transition[S, A]] = []
        done = False

        for _ in range(self.max_steps):
            if done:
                break

            action_distribution = self.policy[state]
            action = action_distribution.sample()
            next_state, reward, done = self.env.step(action)

            if not done:
                next_action = self.policy[next_state].sample()
                transition = Transition(state, action, reward, next_state, next_action, done)
            else:
                transition = Transition(state, action, reward, next_state, done=done)

            transitions.append(transition)

            state = next_state

        return transitions, not done


class ParallelRunner(Generic[S, A]):
    def __init__(
        self,
        env: Environment[S, A],
        policy: Policy[S, A],
        max_steps: int = 1000,
        n_workers: int = 1,
        num_episodes: int = 10,
    ) -> None:
        """Initialize the parallel runner with an environment and a policy.

        Args:
            env: The environment to run the algorithm on.
            policy: The policy to run.
            max_steps: The maximum number of steps to run the algorithm on.
            n_workers: The number of workers to run the algorithm on.
            num_episodes: The number of episodes to run the algorithm on.

        """
        self.env = env
        self.policy = policy
        self.max_steps = max_steps
        self.n_workers = n_workers
        self.num_episodes = num_episodes

    def run(self) -> list[tuple[list[Transition[S, A]], bool]]:
        """Run the parallel runner to collect episodes in parallel.

        Collects episodes from multiple parallel environments using multiprocessing,
        then runs the algorithm to learn from the collected experience.

        Returns:
            List of episodes (transitions, truncated) collected by each worker.

        """
        with mp.Pool(processes=self.n_workers) as pool:
            results = pool.map(self._worker_job, range(self.n_workers))

            all_episodes = []
            for worker_episodes in results:
                all_episodes.extend(worker_episodes)

        return all_episodes

    def _worker_job(self, _: int) -> list[tuple[list[Transition[S, A]], bool]]:
        """Worker job to be run in a separate process.

        Returns:
            List of episodes (transitions, truncated) collected by this worker.

        """
        env_copy = self.env.copy()

        runner = Runner(
            env=env_copy,
            policy=self.policy,
            max_steps=self.max_steps,
        )

        episodes = []
        episodes_per_worker = max(1, self.num_episodes // self.n_workers)

        for _ in range(episodes_per_worker):
            episode = runner.collect_episode()
            episodes.append(episode)

        return episodes
