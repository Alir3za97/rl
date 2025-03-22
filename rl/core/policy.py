import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic

from rl.core.types import A, S


@dataclass(frozen=True)
class ActionDistribution(Generic[A]):
    action_probabilities: dict[A, float]

    def sample(self) -> A:
        actions, probs = zip(*self.action_probabilities.items(), strict=True)
        return random.choices(actions, weights=probs, k=1)[0]

    def get_action_probability(self, action: A) -> float:
        return self.action_probabilities.get(action, 0.0)


class Policy(ABC, Generic[S, A]):
    @abstractmethod
    def __getitem__(self, state: S) -> ActionDistribution[A]:
        """Given a state, returns the action distribution."""
