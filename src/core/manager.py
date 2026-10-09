from typing import Any, Dict, Optional
from core.data_structures import Queue, Stack
from core.storage import JSONStorage


class BaseManager:
    def __init__(self, storage: Optional[JSONStorage] = None) -> None:
        self.storage = storage or JSONStorage()
        self.undo_stack: Stack = Stack()
        self.task_queue: Queue = Queue()
        self.items: Dict[str, Any] = {}

    def load(self) -> None:
        raise NotImplementedError

    def save(self) -> bool:
        raise NotImplementedError

    def push_undo(self, action: Dict[str, Any]) -> None:
        raise NotImplementedError

    def pop_undo(self) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def enqueue_task(self, task_data: Any) -> None:
        raise NotImplementedError

    def dequeue_task(self) -> Optional[Any]:
        raise NotImplementedError
