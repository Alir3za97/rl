from abc import ABC, abstractmethod
from typing import Generic

from rl.core.types import A, Reward, S


class Environment(ABC, Generic[S, A]):
    @abstractmethod
    def reset(self) -> S:
        """Reset the environment and returns the initial state."""

    @abstractmethod
    def step(self, action: A) -> tuple[S, Reward, bool]:
        """Execute an action and return the (next state, reward, done)."""

    @property
    @abstractmethod
    def action_space(self) -> set[A]:
        """Return the action space of the environment."""

    @property
    @abstractmethod
    def current_state(self) -> S:
        """Return the current state of the environment."""

    @abstractmethod
    def close(self) -> None:
        """Close the environment."""


class ModelEnvironment(Environment[S, A], ABC):
    @property
    @abstractmethod
    def state_space(self) -> set[S]:
        """Return the state space of the environment."""

    @property
    @abstractmethod
    def terminal_states(self) -> set[S]:
        """Return the terminal states of the environment."""

    @abstractmethod
    def get_possible_actions(self, state: S) -> set[A]:
        """Return a list of possible actions for a given state."""

    @abstractmethod
    def get_possible_transitions(self, state: S, action: A) -> list[tuple[S, float, Reward]]:
        """Return a list of (next_state, probability, reward) tuples given a state and action."""
