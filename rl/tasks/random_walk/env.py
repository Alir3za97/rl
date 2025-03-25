import random
from typing import TypeAlias

from rl.core.env import ModelEnvironment
from rl.core.types import Reward

# Define specific types for SimpleRandomWalk
State: TypeAlias = int
Action: TypeAlias = str


class SimpleRandomWalk(ModelEnvironment[State, Action]):
    """Simple RandomWalk environment.

    A simple random walk environment where an agent moves left or right randomly.

    This environment implements a 1D random walk where the agent starts at a specified position
    and randomly moves left or right at each step. The episode terminates when the agent
    reaches either the leftmost state (start) or the rightmost state (end).

    States are represented as integers from 0 to n-1, where n is the number of states.
    The environment uses a "no-op" action since the transitions are random regardless
    of the action taken.

    Rewards:
        - terminate_left_reward: Received when reaching the leftmost state
        - terminate_right_reward: Received when reaching the rightmost state
        - step_reward: Received for each non-terminal step
    """

    def __init__(
        self,
        n: int = 10,
        start_state: State = 0,
        terminate_left_reward: float = 0.0,
        terminate_right_reward: float = 1.0,
        step_reward: float = 0.0,
    ) -> None:
        """Initialize the RandomWalk environment.

        Args:
            n: The number of steps.
            start_state: The starting state.
            terminate_left_reward: The reward for terminating on the left.
            terminate_right_reward: The reward for terminating on the right.
            step_reward: The reward for taking a step.

        """
        self.n = n
        self.start_state = start_state
        self.state = start_state
        self.terminate_left_reward = terminate_left_reward
        self.terminate_right_reward = terminate_right_reward
        self.step_reward = step_reward
        self.allowed_actions = {"no-op"}

    def step(self, _: Action) -> tuple[State, Reward, bool]:
        """Execute an action and return the (next state, reward, done)."""
        next_state = random.choice([self.state - 1, self.state + 1])

        next_state = max(0, min(next_state, self.n - 1))
        self.state = next_state

        if next_state == 0:
            return next_state, self.terminate_left_reward, True
        if next_state == self.n - 1:
            return next_state, self.terminate_right_reward, True
        return next_state, self.step_reward, False

    def get_possible_transitions(
        self, state: State, _: Action
    ) -> list[tuple[State, float, Reward]]:
        """Return the possible transitions for a given state and action."""
        if state in [0, self.n - 1]:
            return []

        if state == 1:
            return [
                (state - 1, 0.5, self.terminate_left_reward),
                (state + 1, 0.5, self.step_reward),
            ]
        if state == self.n - 2:
            return [
                (state - 1, 0.5, self.step_reward),
                (state + 1, 0.5, self.terminate_right_reward),
            ]

        return [
            (state - 1, 0.5, self.step_reward),
            (state + 1, 0.5, self.step_reward),
        ]

    @property
    def state_space(self) -> set[State]:
        """Return the state space of the environment."""
        return set(range(self.n))

    @property
    def terminal_states(self) -> set[State]:
        """Return the terminal states of the environment."""
        return {0, self.n - 1}

    def reset(self) -> State:
        """Reset the environment and returns the initial state."""
        self.state = self.start_state
        return self.state

    def close(self) -> None:
        """Close the environment."""

    def get_possible_actions(self, _: State) -> set[Action]:
        """Return the possible actions for a given state."""
        return self.action_space

    @property
    def action_space(self) -> set[Action]:
        """Return the action space of the environment."""
        return self.allowed_actions

    @property
    def current_state(self) -> State:
        """Return the current state of the environment."""
        return self.state

    def copy(self) -> "SimpleRandomWalk":
        """Return a copy of the environment."""
        return SimpleRandomWalk(
            n=self.n,
            start_state=self.start_state,
            terminate_left_reward=self.terminate_left_reward,
            terminate_right_reward=self.terminate_right_reward,
            step_reward=self.step_reward,
        )
