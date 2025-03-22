from dataclasses import dataclass
from typing import Generic, TypeAlias, TypeVar

Reward: TypeAlias = float

S = TypeVar("S")  # Generic type for state
A = TypeVar("A")  # Generic type for action


@dataclass(frozen=True)
class Transition(Generic[S, A]):
    state: S
    action: A
    reward: Reward
    next_state: S
    next_action: A | None = None  # Useful for algorithms like SARSA
    done: bool = False

    def to_sarsa_tuple(self) -> tuple[S, A, Reward, S, A]:
        if self.next_action is None:
            raise ValueError("Next action required for SARSA format")
        return (self.state, self.action, self.reward, self.next_state, self.next_action)

    def to_q_tuple(self) -> tuple[S, A, Reward, S]:
        return (self.state, self.action, self.reward, self.next_state)
