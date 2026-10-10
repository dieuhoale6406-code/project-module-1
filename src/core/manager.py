from typing import Any, Dict, Optional
from core.structures import LinkedQueue, LinkedStack
from core.storage import JSONStorage


class BaseManager:
    def __init__(self, storage: Optional[JSONStorage] = None) -> None:
        self.storage = storage or JSONStorage()
        self.undo_Linkedstack: LinkedStack = LinkedStack()
        self.task_Linkedqueue: LinkedQueue = LinkedQueue()
        self.items: Dict[str, Any] = {}

    def load(self) -> None:
        raise NotImplementedError

    def save(self) -> bool:
        raise NotImplementedError

    def push_undo(self, action: Dict[str, Any]) -> None:
        raise NotImplementedError

    def pop_undo(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def enLinkedqueue_task(self, task_data: Any) -> None:
        raise NotImplementedError

    def deLinkedqueue_task(self) -> Optional[Any]:
        raise NotImplementedError
