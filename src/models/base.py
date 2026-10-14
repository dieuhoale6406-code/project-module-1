from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEntity(ABC):
    """Abstract base entity for all domain models in the system."""

    def __init__(self, entity_id: str) -> None:
        self._id = str(entity_id).strip()

    @property
    def id(self) -> str:
        """Unique identifier of the entity."""
        return self._id

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize entity into a dictionary."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEntity":
        """Instantiate entity from a dictionary."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id='{self._id}'>"

    def __eq__(self, other: Any) -> bool:
        if type(self) is not type(other):
            return False
        return self._id == other._id
