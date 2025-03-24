from abc import ABC, abstractmethod


class Bandit(ABC):
    @abstractmethod
    def pull(self, arm: int) -> float: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def copy(self) -> "Bandit": ...
