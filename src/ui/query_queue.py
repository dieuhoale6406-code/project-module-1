"""Hàng đợi truy vấn FIFO cho phiên Streamlit hiện tại."""

from copy import deepcopy
from typing import Any, Dict, List

import streamlit as st

try:
    from core.manager import AlgorithmNotReadyError, VectorEngineManager
    from core.structures.queue import LinkedQueue
except ImportError:
    from src.core.manager import AlgorithmNotReadyError, VectorEngineManager
    from src.core.structures.queue import LinkedQueue


def sync_query_queue(manager: VectorEngineManager) -> None:
    """Discard pending queries and old results when their dataset changes."""
    state = st.session_state
    if "query_queue" not in state or state.get("query_queue_version") != manager.mutation_version:
        had_work = (
            "query_queue" in state
            and (not state["query_queue"].is_empty() or bool(state["query_queue_results"]))
        )
        state["query_queue"] = LinkedQueue()
        state["query_queue_results"] = []
        state["query_queue_next_id"] = 1
        state["query_queue_version"] = manager.mutation_version
        state.pop("query_queue_result_id", None)
        if had_work:
            st.info("Dữ liệu đã thay đổi. Đã xóa truy vấn đang chờ và kết quả hàng đợi cũ.")


def enqueue_query(
    query_name: str,
    query_values: List[float],
    algorithm: str,
    top_k: int,
    metric: str,
    params: Dict[str, Any],
) -> None:
    state = st.session_state
    state["query_queue"].enqueue({
        "id": state["query_queue_next_id"],
        "query_name": query_name,
        "query_values": list(query_values),
        "algorithm": algorithm,
        "top_k": top_k,
        "metric": metric,
        "params": deepcopy(params),
    })
    state["query_queue_next_id"] += 1


def render_query_queue(manager: VectorEngineManager, algorithm_names: Dict[str, str]) -> None:
    st.markdown("---")
    st.subheader("Hàng đợi truy vấn")
    st.caption(
        "Truy vấn được xử lý theo thứ tự thêm (FIFO) khi bạn bấm xử lý. "
        "Mỗi truy vấn giữ vector, thuật toán, Top-K và cài đặt tại lúc thêm. "
        "Hàng đợi chỉ được lưu trong phiên hiện tại; thay đổi dữ liệu sẽ xóa hàng đợi và kết quả cũ."
    )
    queue: LinkedQueue = st.session_state["query_queue"]
    history = st.session_state["query_queue_results"]

    col_next, col_all, col_clear = st.columns(3)
    run_next = col_next.button("Xử lý truy vấn đầu", disabled=queue.is_empty())
    run_all = col_all.button("Xử lý tất cả", disabled=queue.is_empty())
    clear = col_clear.button("Xóa cuối hàng đợi", disabled=queue.is_empty())

    if clear:
        if not queue.is_empty():
            queue.remove_rear()
        st.rerun()

    if run_next or run_all:
        count = queue.size() if run_all else 1
        with st.spinner(f"Đang xử lý {count} truy vấn theo FIFO..."):
            for _ in range(count):
                job = queue.dequeue()
                result = dict(job)
                try:
                    matches, latency_ms = manager.search(
                        job["algorithm"], job["query_values"], top_k=job["top_k"],
                        metric=job["metric"], **job["params"],
                    )
                    result.update(status="Hoàn thành", matches=matches, latency_ms=latency_ms)
                except (AlgorithmNotReadyError, NotImplementedError):
                    result.update(status="Lỗi", error="Thuật toán chưa sẵn sàng.")
                except Exception as exc:
                    result.update(status="Lỗi", error=f"Không thể xử lý truy vấn: {exc}")
                history.append(result)
        st.rerun()

    def job_row(job: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "Mã": job["id"],
            "Truy vấn": job["query_name"],
            "Thuật toán": algorithm_names.get(job["algorithm"], job["algorithm"]),
            "Top-K": job["top_k"],
            "Độ đo": job["metric"],
            "Cài đặt": str(job["params"]),
        }

    col_size, col_front, col_rear = st.columns(3)
    col_size.metric("Đang chờ", queue.size())
    first = queue.peek()
    last = queue.rear.data if queue.rear else None
    col_front.metric("Đầu hàng đợi", f"#{first['id']} · {first['query_name']}" if first else "Trống")
    col_rear.metric("Cuối hàng đợi", f"#{last['id']} · {last['query_name']}" if last else "Trống")

    pending = []
    node = queue.front
    while node is not None:
        pending.append(job_row(node.data))
        node = node.next
    if pending:
        st.dataframe(pending, hide_index=True)
    else:
        st.info("Chưa có truy vấn đang chờ. Nhập truy vấn phía trên rồi bấm Thêm vào hàng đợi.")

    if history:
        st.markdown("**Kết quả theo thứ tự xử lý**")
        st.dataframe([
            {**job_row(result), "Trạng thái": result["status"],
             "Thời gian (ms)": result.get("latency_ms")}
            for result in history
        ], hide_index=True)
        by_id = {result["id"]: result for result in history}
        selected = st.selectbox(
            "Xem kết quả truy vấn", options=list(by_id), key="query_queue_result_id",
            format_func=lambda job_id: f"#{job_id} · {by_id[job_id]['query_name']}",
        )
        result = by_id[selected]
        if result["status"] == "Lỗi":
            st.error(result["error"])
        else:
            st.dataframe([
                {"Thứ hạng": rank, "Từ vựng": record.label, "Mã ID": record.id,
                 "Khoảng cách": distance}
                for rank, (record, distance) in enumerate(result["matches"], start=1)
            ], hide_index=True)
