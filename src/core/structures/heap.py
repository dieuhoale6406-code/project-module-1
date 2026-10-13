import math
from typing import Any, List, Optional, Tuple


class BoundedMaxHeap:
    """Bounded max-heap for maintaining Top-K nearest candidates."""

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError(f"capacity must be a positive integer (> 0), got: {capacity}")
        self.capacity = capacity
        self._heap: List[Tuple[float, str, Any]] = []

    def size(self) -> int:
        return len(self._heap)

    def __len__(self) -> int:
        return len(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def is_full(self) -> bool:
        return len(self._heap) >= self.capacity

    @property
    def max_distance(self) -> Optional[float]:
        """Maximum distance in the heap (root)."""
        if self.is_empty():
            return None
        return self._heap[0][0]

    def peek_max(self) -> Optional[Tuple[float, Any]]:
        """Return the worst candidate currently retained without popping."""
        if self.is_empty():
            return None
        return (self._heap[0][0], self._heap[0][2])

    @staticmethod
    def _is_worse(a: Tuple[float, str, Any], b: Tuple[float, str, Any]) -> bool:
        """Return True if candidate a ranks worse than b."""
        if a[0] != b[0]:
            return a[0] > b[0]
        return a[1] > b[1]

    def _sift_up(self, index: int) -> None:
        while index > 0:
            parent = (index - 1) // 2
            if self._is_worse(self._heap[index], self._heap[parent]):
                self._heap[index], self._heap[parent] = self._heap[parent], self._heap[index]
                index = parent
            else:
                break

    def _sift_down(self, index: int) -> None:
        n = len(self._heap)
        while True:
            worst_idx = index
            left = 2 * index + 1
            right = 2 * index + 2

            if left < n and self._is_worse(self._heap[left], self._heap[worst_idx]):
                worst_idx = left
            if right < n and self._is_worse(self._heap[right], self._heap[worst_idx]):
                worst_idx = right

            if worst_idx != index:
                self._heap[index], self._heap[worst_idx] = self._heap[worst_idx], self._heap[index]
                index = worst_idx
            else:
                break

    def push(self, distance: float, item: Any, item_id: Optional[str] = None) -> bool:
        """Insert item or replace the root if it ranks better than the root."""
        if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
            raise ValueError(f"distance must be a finite number, got: {distance}")

        dist = float(distance)
        rec_id = str(item_id) if item_id is not None else str(item.id)
        entry = (dist, rec_id, item)

        if len(self._heap) < self.capacity:
            self._heap.append(entry)
            self._sift_up(len(self._heap) - 1)
            return True

        root = self._heap[0]
        if self._is_worse(root, entry):
            self._heap[0] = entry
            self._sift_down(0)
            return True

        return False

    def get_sorted_results(self) -> List[Tuple[Any, float]]:
        """Return retained candidates sorted ascending by (distance, ID)."""
        if not self._heap:
            return []
        sorted_entries = sorted(self._heap, key=lambda entry: (entry[0], entry[1]))
        return [(entry[2], entry[0]) for entry in sorted_entries]

    def clear(self) -> None:
        self._heap.clear()
