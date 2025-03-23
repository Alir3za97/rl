from abc import ABC, abstractmethod
from typing import Generic

from rl.core.env import Environment, ModelEnvironment
from rl.core.policy import Policy
from rl.core.types import A, S


class ControlAlgorithm(ABC, Generic[S, A]):
    @abstractmethod
    def run(self) -> Policy[S, A]:
        """Run the algorithm."""

    @abstractmethod
    def get_policy(self) -> Policy[S, A]:
        """Return the current policy."""


class PolicyEvaluationAlgorithm(ABC, Generic[S, A]):
    @abstractmethod
    def run(self) -> None:
        """Run the algorithm."""

    @abstractmethod
    def get_value_function(self) -> dict[S, float]:
        """Return the current value function."""


class ModelBasedControlAlgorithm(ControlAlgorithm[S, A], Generic[S, A]):
    @abstractmethod
    def __init__(self, env: ModelEnvironment[S, A]) -> None:
        """Initialize the model-based algorithm.

        Args:
            env: The model environment to run the algorithm on.

        """
        self.env = env


class ModelFreeControlAlgorithm(ControlAlgorithm[S, A], Generic[S, A]):
    @abstractmethod
    def __init__(self, env: Environment[S, A]) -> None:
        """Initialize the model-free algorithm.

        Args:
            env: The model free environment to run the algorithm on.

        """
        self.env = env
