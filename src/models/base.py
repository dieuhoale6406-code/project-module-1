from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEntity(ABC):
    def __init__(self, entity_id: str) -> None:
        self._id = entity_id

    @property
    def id(self) -> str:
        return self._id

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEntity":
        raise NotImplementedError
