from abc import ABC, abstractmethod


class Agent(ABC):
    @abstractmethod
    def select_arm(self) -> int: ...

    @abstractmethod
    def observe(self, arm: int, reward: float) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...
