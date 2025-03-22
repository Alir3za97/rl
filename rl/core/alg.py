from abc import ABC, abstractmethod
from typing import Generic

from rl.core.policy import Policy
from rl.core.types import A, S


class RLAlgorithm(ABC, Generic[S, A]):
    @abstractmethod
    def get_policy(self) -> Policy[S, A]:
        """Return the current policy."""


class OfflineRLAlgorithm(RLAlgorithm[S, A], Generic[S, A]):
    @abstractmethod
    def run(self) -> Policy[S, A]:
        """Run the algorithm."""
