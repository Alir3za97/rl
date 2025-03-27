from enum import Enum


class VisitType(Enum):
    FIRST_VISIT = "first_visit"
    EVERY_VISIT = "every_visit"

    def __str__(self) -> str:
        """Return the string representation of the visit type."""
        return self.value
