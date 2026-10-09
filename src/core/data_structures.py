from typing import Any, Optional


class Stack:
    def __init__(self) -> None:
        self._items: list = []

    def push(self, item: Any) -> None:
        raise NotImplementedError

    def pop(self) -> Any:
        raise NotImplementedError

    def peek(self) -> Optional[Any]:
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError


class Queue:
    def __init__(self) -> None:
        self._items: list = []

    def enqueue(self, item: Any) -> None:
        raise NotImplementedError

    def dequeue(self) -> Any:
        raise NotImplementedError

    def peek(self) -> Optional[Any]:
        raise NotImplementedError

    def is_empty(self) -> bool:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError
