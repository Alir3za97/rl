from typing import Generic

from rl.core.alg import RLAlgorithm
from rl.core.env import Environment
from rl.core.types import A, S, Transition


class Runner(Generic[S, A]):
    def __init__(self, env: Environment[S, A], algorithm: RLAlgorithm[S, A]) -> None:
        """Initialize the runner with an environment and an algorithm."""
        self.env = env
        self.algorithm = algorithm

    def collect_episode(self, max_steps: int = 1000) -> list[Transition[S, A]]:
        """Collect a trajectory from the environment."""
        state = self.env.reset()
        transitions: list[Transition[S, A]] = []
        done = False

        for _ in range(max_steps):
            if done:
                break

            policy = self.algorithm.get_policy()
            action_distribution = policy[state]
            action = action_distribution.sample()
            next_state, reward, done = self.env.step(action)

            if not done:
                next_action = policy[next_state].sample()
                transition = Transition(state, action, reward, next_state, next_action, done)
            else:
                transition = Transition(state, action, reward, next_state, done=done)

            transitions.append(transition)

            state = next_state

        return transitions
