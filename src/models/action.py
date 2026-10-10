import time
from typing import Any, Dict, Optional
from .base import BaseEntity


class ActionRecord(BaseEntity):
    """
    Thực thể ghi nhận một hành động thay đổi trạng thái trong hệ thống (Command Pattern).
    Được lưu trữ trong LinkedStack để phục vụ thao tác Hoàn tác (Undo) và Làm lại (Redo).
    """

    def __init__(
        self,
        action_id: str,
        action_type: str,
        target_label: str,
        payload: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> None:
        super().__init__(action_id)
        # Loại hành động: 'ADD' (thêm vector mới), 'DELETE' (xóa vector), 'UPDATE' (sửa tọa độ)
        self._action_type = str(action_type).strip().upper()
        self._target_label = str(target_label).strip().lower()
        # Dữ liệu phục hồi (ví dụ: bản ghi cũ trước khi xóa/sửa)
        self._payload = payload or {}
        self._timestamp = timestamp if timestamp is not None else time.time()

    @property
    def action_type(self) -> str:
        """Loại thao tác ('ADD', 'DELETE', 'UPDATE')."""
        return self._action_type

    @property
    def target_label(self) -> str:
        """Nhãn của vector bị tác động."""
        return self._target_label

    @property
    def payload(self) -> Dict[str, Any]:
        """Dữ liệu sao lưu phục vụ khôi phục trạng thái."""
        return self._payload

    @property
    def timestamp(self) -> float:
        """Dấu thời gian khi thao tác diễn ra."""
        return self._timestamp

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self._id,
            "action_type": self._action_type,
            "target_label": self._target_label,
            "payload": self._payload,
            "timestamp": self._timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        return cls(
            action_id=data["id"],
            action_type=data["action_type"],
            target_label=data["target_label"],
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp"),
        )

    def __repr__(self) -> str:
        return f"<ActionRecord id='{self._id}' type='{self._action_type}' target='{self._target_label}'>"
