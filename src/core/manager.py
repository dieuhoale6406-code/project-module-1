import importlib
import inspect
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

try:
    from core.structures import LinkedStack
    from models.action import ActionRecord
    from models.record import VectorRecord
except ImportError:
    from .structures import LinkedStack
    from ..models.action import ActionRecord
    from ..models.record import VectorRecord

from .algorithms.base import BaseVectorIndex
from .algorithms.linear_scan import LinearScanIndex

SUPPORTED_METRICS = ("cosine", "euclidean")
_METRIC_ALIASES = {"cosine_distance": "cosine", "euclidean_distance": "euclidean"}

# Provisional metrics inferred from teammate stubs; subject to final confirmation.
_TEAM_ALGORITHMS = (
    ("kdtree", "kdtree", "KDTreeIndex", ("euclidean",)),
    ("lsh", "lsh", "LSHIndex", ("cosine",)),
    ("ivf", "ivf", "IVFIndex", ("euclidean",)),
    ("hnsw", "hnsw", "HNSWIndex", ("euclidean",)),
)


class AlgorithmNotReadyError(NotImplementedError):
    """Raised when an algorithm is registered but not implemented or unimportable."""


@dataclass
class _CachedIndex:
    """Built index instance along with configuration parameters and build time."""

    index: BaseVectorIndex
    params: Dict[str, Any]
    build_time_ms: float


class VectorEngineManager:
    """
    Central facade for the Vector Search Engine.
    Manages vector records, coordinates search algorithms, and maintains index caching.
    """

    def __init__(
        self,
        initial_records: Optional[List[VectorRecord]] = None,
        undo_stack: Optional[Any] = None,
    ) -> None:
        self._records: Dict[str, VectorRecord] = {}
        self._label_to_id: Dict[str, str] = {}
        self._dimension: Optional[int] = None

        self._index_cache: Dict[Tuple[str, str], _CachedIndex] = {}
        self._rebuild_count: int = 0
        self._last_build_time_ms: float = 0.0

        self._algorithm_registry: Dict[str, Type[BaseVectorIndex]] = {}
        self._algorithm_metrics: Dict[str, Optional[Tuple[str, ...]]] = {}
        self._algorithm_status: Dict[str, str] = {}
        self._unavailable_reason: Dict[str, str] = {}
        self.register_algorithm("linear_scan", LinearScanIndex, ("cosine", "euclidean"))
        self._register_team_algorithms()

        # Support dependency injection for Undo stack.
        if undo_stack is not None:
            self.undo_stack = undo_stack
        else:
            try:
                self.undo_stack: Optional[LinkedStack] = LinkedStack()
            except Exception:
                self.undo_stack = None

        # Initial records are seeded without tracking undo history.
        if initial_records:
            for rec in initial_records:
                self.add_record(
                    label=rec.label,
                    values=rec.values,
                    record_id=rec.id,
                    track_undo=False,
                )

    @property
    def record_count(self) -> int:
        """Total number of vector records in memory."""
        return len(self._records)

    @property
    def dimension(self) -> int:
        """Dimensionality of the vector space (0 if empty)."""
        return self._dimension if self._dimension is not None else 0

    @property
    def is_stale(self) -> bool:
        """True if no cached index is currently valid for the dataset."""
        return not self._index_cache

    @property
    def rebuild_count(self) -> int:
        """Total number of times indexes have been built or rebuilt."""
        return self._rebuild_count

    @property
    def last_build_time_ms(self) -> float:
        """Duration of the most recent index build in milliseconds."""
        return self._last_build_time_ms

    def get_all_records(self) -> List[VectorRecord]:
        """Return all vector records currently in memory."""
        return list(self._records.values())

    def get_record(self, identifier: str) -> Optional[VectorRecord]:
        """Retrieve a vector record by label or unique ID."""
        if not identifier:
            return None

        clean_str = str(identifier).strip()
        if clean_str in self._records:
            return self._records[clean_str]

        norm_label = clean_str.lower()
        if norm_label in self._label_to_id:
            target_id = self._label_to_id[norm_label]
            return self._records.get(target_id)

        return None

    def add_record(
        self,
        label: str,
        values: List[float],
        record_id: Optional[str] = None,
        track_undo: bool = True,
    ) -> VectorRecord:
        """Add a new vector record, validating label, dimension, and coordinate finiteness."""
        if not label or not str(label).strip():
            raise ValueError("Nhãn từ vựng (label) không được để trống.")

        norm_label = str(label).strip().lower()
        if norm_label in self._label_to_id:
            raise ValueError(f"Từ vựng '{label}' đã tồn tại trong hệ thống.")

        if not values:
            raise ValueError("Tọa độ vector không được để trống.")

        for v in values:
            if not math.isfinite(v):
                raise ValueError("Tọa độ vector chứa giá trị không hợp lệ (NaN hoặc Infinity).")

        val_dim = len(values)
        if self._dimension is not None and val_dim != self._dimension:
            raise ValueError(
                f"Số chiều vector ({val_dim}) không khớp với hệ thống ({self._dimension})."
            )

        if record_id:
            clean_id = str(record_id).strip()
            if clean_id in self._records:
                raise ValueError(f"Mã ID '{record_id}' đã tồn tại trong hệ thống.")
        else:
            next_idx = len(self._records) + 1
            clean_id = f"VEC_{next_idx:05d}"
            while clean_id in self._records:
                next_idx += 1
                clean_id = f"VEC_{next_idx:05d}"

        record = VectorRecord(record_id=clean_id, label=norm_label, values=values)
        self._records[clean_id] = record
        self._label_to_id[norm_label] = clean_id

        if self._dimension is None:
            self._dimension = val_dim

        self._invalidate_indexes()

        if track_undo:
            self._push_undo_action("ADD", norm_label, {"record": record})

        return record

    def delete_record(self, label: str, track_undo: bool = True) -> bool:
        """Delete a vector record by label or ID. Return True if deleted."""
        target_rec = self.get_record(label)
        if target_rec is None:
            return False

        rec_id = target_rec.id
        norm_label = target_rec.label

        del self._records[rec_id]
        self._label_to_id.pop(norm_label, None)

        if len(self._records) == 0:
            self._dimension = None

        self._invalidate_indexes()

        if track_undo:
            self._push_undo_action("DELETE", norm_label, {"record": target_rec})

        return True

    def update_record(self, label: str, values: List[float], track_undo: bool = True) -> bool:
        """Update coordinates of an existing record by label."""
        target_rec = self.get_record(label)
        if target_rec is None:
            return False

        if not values:
            raise ValueError("Tọa độ vector không được để trống.")

        for v in values:
            if not math.isfinite(v):
                raise ValueError("Tọa độ vector chứa giá trị không hợp lệ (NaN hoặc Infinity).")

        if self._dimension is not None and len(values) != self._dimension:
            raise ValueError(
                f"Số chiều vector ({len(values)}) không khớp với hệ thống ({self._dimension})."
            )

        old_rec = target_rec
        updated_rec = VectorRecord(
            record_id=old_rec.id, label=old_rec.label, values=values
        )

        self._records[old_rec.id] = updated_rec
        self._invalidate_indexes()

        if track_undo:
            self._push_undo_action(
                "UPDATE", old_rec.label, {"old_record": old_rec, "new_record": updated_rec}
            )
        return True

    def build_index(
        self, algorithm: str, metric: Optional[str] = None, **params: Any
    ) -> float:
        """
        Explicitly build an index and return build time in milliseconds.
        Manager supplies 'dimension' and 'metric'; extra parameters are passed to constructor.
        """
        name, algorithm_cls = self._resolve_algorithm(algorithm)
        canonical = self._resolve_metric(name, metric)
        return self._build_and_cache(name, algorithm_cls, canonical, params).build_time_ms

    def search(
        self,
        algorithm: str,
        query_values: List[float],
        top_k: int,
        metric: Optional[str] = None,
        **params: Any,
    ) -> Tuple[List[Tuple[VectorRecord, float]], float]:
        """
        Search Top-K nearest neighbors. Return ([(VectorRecord, distance)], latency_ms).
        Reuses cached index when available; rebuilds if parameters or dataset changed.
        """
        name, algorithm_cls = self._resolve_algorithm(algorithm)
        canonical = self._resolve_metric(name, metric)
        self._validate_search_args(query_values, top_k)

        if not self._records:
            return [], 0.0

        entry = self._index_cache.get((name, canonical))
        if entry is None or (params and entry.params != params):
            entry = self._build_and_cache(name, algorithm_cls, canonical, params)

        t0 = time.perf_counter()
        try:
            results = entry.index.search(query_values, top_k=top_k)
        except NotImplementedError as exc:
            self._index_cache.pop((name, canonical), None)
            raise self._not_ready(name, exc) from exc
        t1 = time.perf_counter()
        return results, (t1 - t0) * 1000.0

    def get_build_time_ms(self, algorithm: str, metric: Optional[str] = None) -> Optional[float]:
        """Return cached build time (ms) for the given algorithm and metric, or None."""
        name, _ = self._resolve_algorithm(algorithm)
        entry = self._index_cache.get((name, self._resolve_metric(name, metric)))
        return entry.build_time_ms if entry is not None else None

    def _invalidate_indexes(self) -> None:
        """Invalidate all cached indexes when dataset changes."""
        self._index_cache.clear()

    def _build_and_cache(
        self,
        name: str,
        algorithm_cls: Type[BaseVectorIndex],
        metric: str,
        params: Dict[str, Any],
    ) -> _CachedIndex:
        """Instantiate, build, and cache an index for the given algorithm and metric."""
        t0 = time.perf_counter()
        try:
            index = self._create_index(name, algorithm_cls, metric, params)
            index.build(list(self._records.values()))
        except NotImplementedError as exc:
            self._index_cache.pop((name, metric), None)
            raise self._not_ready(name, exc) from exc
        build_time_ms = (time.perf_counter() - t0) * 1000.0

        entry = _CachedIndex(index=index, params=dict(params), build_time_ms=build_time_ms)
        self._index_cache[(name, metric)] = entry
        self._algorithm_status[name] = "ready"
        self._rebuild_count += 1
        self._last_build_time_ms = build_time_ms
        return entry

    def _create_index(
        self,
        name: str,
        algorithm_cls: Type[BaseVectorIndex],
        metric: str,
        params: Dict[str, Any],
    ) -> BaseVectorIndex:
        """Instantiate index adapter inspecting constructor signature for dimension and metric."""
        reserved = sorted({"dimension", "metric"} & set(params))
        if reserved:
            raise ValueError(f"Tham số {reserved} do Manager tự cấp, không được truyền thủ công.")

        signature = inspect.signature(algorithm_cls)
        kwargs = dict(params)
        if "dimension" in signature.parameters:
            kwargs["dimension"] = self.dimension
        if "metric" in signature.parameters:
            kwargs["metric"] = metric
        try:
            signature.bind(**kwargs)
        except TypeError as exc:
            raise ValueError(f"Cấu hình không hợp lệ cho thuật toán '{name}': {exc}") from exc
        return algorithm_cls(**kwargs)

    def _not_ready(self, name: str, cause: NotImplementedError) -> AlgorithmNotReadyError:
        """Record unavailable status and wrap NotImplementedError in a descriptive exception."""
        self._algorithm_status[name] = "not_implemented"
        return AlgorithmNotReadyError(f"Thuật toán '{name}' chưa được triển khai: {cause}")

    def _resolve_algorithm(self, algorithm: str) -> Tuple[str, Type[BaseVectorIndex]]:
        name = str(algorithm).lower().strip()
        algorithm_cls = self._algorithm_registry.get(name)
        if algorithm_cls is not None:
            return name, algorithm_cls
        if name in self._unavailable_reason:
            raise AlgorithmNotReadyError(
                f"Thuật toán '{name}' không khả dụng: {self._unavailable_reason[name]}"
            )
        known = ", ".join(self._algorithm_registry) or "(chưa có)"
        raise ValueError(f"Thuật toán '{algorithm}' chưa được đăng ký. Đã đăng ký: {known}.")

    @staticmethod
    def _canonical_metric(metric: str) -> str:
        key = str(metric).lower().strip()
        key = _METRIC_ALIASES.get(key, key)
        if key not in SUPPORTED_METRICS:
            raise ValueError(
                f"Metric '{metric}' không được hỗ trợ. Chỉ hỗ trợ: {', '.join(SUPPORTED_METRICS)}."
            )
        return key

    def _resolve_metric(self, name: str, metric: Optional[str]) -> str:
        """Normalize metric and verify support by the chosen algorithm."""
        supported = self._algorithm_metrics.get(name)
        if metric is None:
            return supported[0] if supported else "cosine"
        canonical = self._canonical_metric(metric)
        if supported is not None and canonical not in supported:
            raise ValueError(
                f"Thuật toán '{name}' không hỗ trợ metric '{canonical}'. "
                f"Hỗ trợ: {', '.join(supported)}."
            )
        return canonical

    def _validate_search_args(self, query_values: List[float], top_k: int) -> None:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k phải là số nguyên dương (> 0), nhận được: {top_k}.")
        if not query_values:
            raise ValueError("query_values không được rỗng.")
        for v in query_values:
            if not math.isfinite(v):
                raise ValueError("query_values chứa giá trị không hợp lệ (NaN hoặc Infinity).")
        if self._dimension is not None and len(query_values) != self._dimension:
            raise ValueError(
                f"Số chiều của query ({len(query_values)}) không khớp với hệ thống ({self._dimension})."
            )

    def build_linear_index(self, metric: str = "cosine") -> float:
        """Build LinearScanIndex explicitly (wrapper for build_index)."""
        return self.build_index("linear_scan", metric=metric)

    def search_linear(
        self, query_values: List[float], top_k: int, metric: str = "cosine"
    ) -> Tuple[List[Tuple[VectorRecord, float]], float]:
        """Perform Linear Scan exact KNN search (wrapper for search)."""
        return self.search("linear_scan", query_values, top_k, metric=metric)

    def search_kdtree(
        self, query_values: List[float], top_k: int, metric: str = "cosine"
    ) -> Tuple[List[Tuple[VectorRecord, float]], float]:
        """
        Perform KD-Tree search (backward-compatible wrapper).
        'metric' parameter is ignored for backward compatibility.
        """
        return self.search("kdtree", query_values, top_k)

    def register_algorithm(
        self,
        name: str,
        algorithm_cls: Type[BaseVectorIndex],
        supported_metrics: Optional[Tuple[str, ...]] = None,
    ) -> None:
        """Register a search algorithm class with optional supported metrics."""
        key = str(name).lower().strip()
        if not key:
            raise ValueError("Tên thuật toán không được để trống.")
        if not (
            isinstance(algorithm_cls, type)
            and callable(getattr(algorithm_cls, "build", None))
            and callable(getattr(algorithm_cls, "search", None))
        ):
            raise ValueError(f"'{name}' phải là một class có build() và search() (BaseVectorIndex).")
        if supported_metrics is not None and not supported_metrics:
            raise ValueError("supported_metrics không được rỗng (dùng None nếu không giới hạn).")

        self._algorithm_registry[key] = algorithm_cls
        self._algorithm_metrics[key] = (
            None
            if supported_metrics is None
            else tuple(self._canonical_metric(m) for m in supported_metrics)
        )
        self._algorithm_status[key] = "registered"
        self._unavailable_reason.pop(key, None)
        for cache_key in [k for k in self._index_cache if k[0] == key]:
            del self._index_cache[cache_key]

    def _register_team_algorithms(self) -> None:
        """Register teammate algorithm modules; mark unimportable ones as unavailable."""
        for name, module, class_name, metrics in _TEAM_ALGORITHMS:
            try:
                module_obj = importlib.import_module(f".algorithms.{module}", package=__package__)
                algorithm_cls = getattr(module_obj, class_name)
            except (ImportError, AttributeError) as exc:
                self._unavailable_reason[name] = f"{type(exc).__name__}: {exc}"
                continue
            self.register_algorithm(name, algorithm_cls, metrics)

    def get_algorithm(self, name: str) -> Optional[Type[BaseVectorIndex]]:
        """Retrieve registered algorithm class by name."""
        return self._algorithm_registry.get(name.lower().strip())

    def get_algorithm_status(self, name: str) -> str:
        """Return status: 'unregistered' | 'unavailable' | 'registered' | 'ready' | 'not_implemented'."""
        key = str(name).lower().strip()
        if key in self._algorithm_registry:
            return self._algorithm_status.get(key, "registered")
        return "unavailable" if key in self._unavailable_reason else "unregistered"

    def list_algorithms(self) -> Dict[str, str]:
        """Return dictionary mapping algorithm names to their current statuses."""
        names = list(self._algorithm_registry)
        names += [n for n in self._unavailable_reason if n not in self._algorithm_registry]
        return {n: self.get_algorithm_status(n) for n in names}

    def get_supported_metrics(self, name: str) -> Tuple[str, ...]:
        """Return tuple of supported metrics for the algorithm, default metric first."""
        key = str(name).lower().strip()
        if key not in self._algorithm_registry:
            raise ValueError(f"Thuật toán '{name}' chưa được đăng ký.")
        return self._algorithm_metrics.get(key) or SUPPORTED_METRICS

    def _push_undo_action(
        self, action_type: str, target_label: str, payload: Dict[str, Any]
    ) -> None:
        """Push an action onto the undo stack if available."""
        if self.undo_stack is not None:
            try:
                action = ActionRecord(
                    action_id=f"ACT_{int(time.time()*1000)}",
                    action_type=action_type,
                    target_label=target_label,
                    payload=payload,
                )
                self.undo_stack.push(action)
            except NotImplementedError:
                # LinkedStack.push not yet implemented by teammate.
                pass

    def undo(self) -> Optional[str]:
        """
        Revert the most recent mutating action in LIFO order.
        Validates ID, label, and dimension conflicts to prevent data loss.
        """
        if self.undo_stack is None:
            return None

        try:
            if hasattr(self.undo_stack, "is_empty") and self.undo_stack.is_empty():
                return None
        except NotImplementedError:
            return None

        # Prefer peek() to avoid dropping the action if validation fails.
        action: Optional[ActionRecord] = None
        used_peek = False

        if hasattr(self.undo_stack, "peek"):
            try:
                action = self.undo_stack.peek()
                used_peek = True
            except NotImplementedError:
                used_peek = False

        if not used_peek:
            try:
                action = self.undo_stack.pop()
            except NotImplementedError:
                return None

        if action is None:
            return None

        def _rollback_and_abort() -> None:
            if not used_peek:
                try:
                    self.undo_stack.push(action)
                except NotImplementedError:
                    pass

        act_type = action.action_type

        if act_type == "ADD":
            target = action.target_label
            payload_rec = action.payload.get("record") if action.payload else None

            # Fall back to payload record ID if label mapping was altered.
            expected_id = payload_rec.id if payload_rec else self._label_to_id.get(target)
            if not expected_id or expected_id not in self._records:
                _rollback_and_abort()
                return None

            if self._label_to_id.get(target) != expected_id:
                _rollback_and_abort()
                return None

            target_rec = self._records[expected_id]
            if payload_rec is not None and (
                target_rec.values != payload_rec.values or target_rec.label != payload_rec.label
            ):
                _rollback_and_abort()
                return None

            self._records.pop(expected_id, None)
            self._label_to_id.pop(target, None)

            if len(self._records) == 0:
                self._dimension = None
            self._invalidate_indexes()
            msg = f"Đã hoàn tác: Xóa lại vector '{target}'."

        elif act_type == "DELETE":
            rec: Optional[VectorRecord] = action.payload.get("record") if action.payload else None
            if rec is None:
                _rollback_and_abort()
                return None

            if rec.label in self._label_to_id:
                _rollback_and_abort()
                return None

            if rec.id in self._records:
                _rollback_and_abort()
                return None

            if self._dimension is not None and rec.dimension != self._dimension:
                _rollback_and_abort()
                return None

            self._records[rec.id] = rec
            self._label_to_id[rec.label] = rec.id

            if self._dimension is None:
                self._dimension = rec.dimension
            self._invalidate_indexes()
            msg = f"Đã hoàn tác: Khôi phục vector '{rec.label}'."

        elif act_type == "UPDATE":
            old_rec: Optional[VectorRecord] = action.payload.get("old_record") if action.payload else None
            new_rec: Optional[VectorRecord] = action.payload.get("new_record") if action.payload else None
            if old_rec is None:
                _rollback_and_abort()
                return None

            if old_rec.id not in self._records:
                _rollback_and_abort()
                return None

            current_rec = self._records[old_rec.id]
            if current_rec.label != old_rec.label:
                _rollback_and_abort()
                return None

            if self._label_to_id.get(old_rec.label) != old_rec.id:
                _rollback_and_abort()
                return None

            if self._dimension is not None and old_rec.dimension != self._dimension:
                _rollback_and_abort()
                return None

            if new_rec is not None and (
                current_rec.values != new_rec.values or current_rec.label != new_rec.label
            ):
                _rollback_and_abort()
                return None

            self._records[old_rec.id] = old_rec
            self._label_to_id[old_rec.label] = old_rec.id
            self._invalidate_indexes()
            msg = f"Đã hoàn tác: Phục hồi tọa độ cũ cho '{old_rec.label}'."

        else:
            _rollback_and_abort()
            return None

        # Pop action from stack only after successful execution when peek() was used.
        if used_peek:
            try:
                self.undo_stack.pop()
            except NotImplementedError:
                pass

        return msg
