from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseEntity(ABC):
    """
    Lớp cơ sở trừu tượng cho tất cả các thực thể trong hệ thống.
    Áp dụng nguyên lý đóng gói (Encapsulation) và trừu tượng (Abstraction).
    """

    def __init__(self, entity_id: str) -> None:
        self._id = str(entity_id).strip()

    @property
    def id(self) -> str:
        """Mã định danh duy nhất của thực thể."""
        return self._id

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi đối tượng sang dictionary nguyên bản."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseEntity":
        """Khởi tạo đối tượng từ dictionary."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id='{self._id}'>"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseEntity):
            return False
        return self._id == other._id
