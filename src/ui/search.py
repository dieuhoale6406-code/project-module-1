import time
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from core.manager import AlgorithmNotReadyError, VectorEngineManager
except ImportError:
    from src.core.manager import AlgorithmNotReadyError, VectorEngineManager

try:
    from ui.dataset import parse_vector_input
except ImportError:
    from src.ui.dataset import parse_vector_input


ALGORITHM_DISPLAY_NAMES: Dict[str, str] = {
    "linear_scan": "Linear Scan",
    "kdtree": "KD-Tree",
    "lsh": "LSH",
    "ivf": "IVF-Flat",
    "hnsw": "HNSW",
}


def format_algorithm_status_badge(status: str) -> str:
    """Return a concise, user-friendly Vietnamese status label."""
    badges = {
        "ready": "Sẵn sàng",
        "registered": "Chưa sẵn sàng",
        "not_implemented": "Chưa sẵn sàng",
        "unavailable": "Không khả dụng",
        "unregistered": "Chưa sẵn sàng",
    }
    return badges.get(status, "Chưa sẵn sàng")


def compute_top_k_bounds(
    record_count: int, default_target: int = 5, max_cap: int = 50
) -> Tuple[int, int, int]:
    """
    Compute safe (min_k, max_k, default_k) bounds for Top-K selection.
    Capped at available records and max_cap.
    """
    if record_count <= 0:
        return 0, 0, 0
    if record_count == 1:
        return 1, 1, 1
    max_k = min(max_cap, record_count)
    val_k = min(default_target, max_k)
    return 1, max_k, val_k


def _render_advanced_settings(
    manager: VectorEngineManager, selected_alg: str, metric_key: str
) -> Dict[str, Any]:
    """Render collapsed advanced configuration with algorithm parameters and compact preparation control."""
    alg_params: Dict[str, Any] = {}

    with st.expander("Cài đặt nâng cao", expanded=False):
        # Algorithm-specific parameter tuning
        if selected_alg == "linear_scan":
            st.caption(
                "Linear Scan: Duyệt tuần tự toàn bộ vector đối chiếu chuẩn (Ground Truth). "
                "Không sử dụng chỉ mục phân vùng hay xấp xỉ."
            )
        elif selected_alg == "kdtree":
            st.caption("KD-Tree: Phân vùng trực giao theo trục tọa độ để tăng tốc tìm kiếm ở số chiều thấp.")
        elif selected_alg == "lsh":
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                alg_params["num_hyperplanes"] = st.slider(
                    "Số siêu phẳng (num_hyperplanes)", min_value=2, max_value=32, value=8
                )
            with c_p2:
                alg_params["seed"] = st.number_input("Hạt giống (Seed)", value=42, step=1)
        elif selected_alg == "ivf":
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                alg_params["num_clusters"] = st.slider(
                    "Số cụm (num_clusters)", min_value=2, max_value=64, value=8
                )
            with c_p2:
                alg_params["nprobe"] = st.slider(
                    "Số cụm dò tìm (nprobe)", min_value=1, max_value=16, value=2
                )
        elif selected_alg == "hnsw":
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1:
                alg_params["m"] = st.slider("M (liên kết/node)", min_value=4, max_value=32, value=8)
            with c_p2:
                alg_params["ef_construction"] = st.slider(
                    "ef_construction", min_value=8, max_value=128, value=32
                )
            with c_p3:
                alg_params["ef_search"] = st.slider("ef_search", min_value=4, max_value=64, value=16)

        st.markdown("---")

        # Compact manual preparation control
        st.markdown("**Chuẩn bị thuật toán**")
        if selected_alg == "linear_scan":
            desc = (
                "Linear Scan chuẩn bị dữ liệu trong bộ nhớ để tìm kiếm chính xác (Ground Truth). "
                "Hệ thống sẽ tự động chuẩn bị khi tìm kiếm."
            )
        else:
            desc = (
                "Thuật toán sẽ tự động chuẩn bị khi tìm kiếm, hoặc bạn có thể bấm chuẩn bị trước tại đây."
            )

        cached_bld_ms = manager.get_build_time_ms(selected_alg, metric_key)
        raw_status = manager.get_algorithm_status(selected_alg)

        if cached_bld_ms is not None:
            status_text = f":green[● Đã chuẩn bị] ({cached_bld_ms:.2f} ms)"
        elif raw_status in ("not_implemented", "unavailable"):
            status_text = ":orange[● Chưa sẵn sàng]"
        else:
            status_text = ":gray[● Tự động chuẩn bị khi tìm kiếm]"

        c_info, c_btn = st.columns([3, 1])
        with c_info:
            st.caption(desc)
            st.markdown(f"Trạng thái: {status_text}")
        with c_btn:
            st.write("")
            btn_prepare = st.button(
                "Chuẩn bị",
                type="secondary",
                icon=":material/tune:",
                use_container_width=True,
                help=f"Chuẩn bị trước cho thuật toán {ALGORITHM_DISPLAY_NAMES.get(selected_alg, selected_alg)}.",
            )

        if btn_prepare:
            try:
                with st.spinner(f"Đang chuẩn bị {ALGORITHM_DISPLAY_NAMES.get(selected_alg, selected_alg)}..."):
                    bld_ms = manager.build_index(selected_alg, metric=metric_key, **alg_params)
                st.success(f"Chuẩn bị thành công trong **{bld_ms:.2f} ms**.")
                st.rerun()
            except (AlgorithmNotReadyError, NotImplementedError):
                st.warning(
                    f"Thuật toán **{ALGORITHM_DISPLAY_NAMES.get(selected_alg, selected_alg)}** "
                    f"hiện chưa sẵn sàng."
                )
            except Exception:
                st.error("Không thể chuẩn bị thuật toán. Vui lòng thử lại sau.")

    return alg_params


def _render_single_search_results(data: Dict[str, Any]) -> None:
    """Render performance metrics cards at the top and Top-K results table, matching Tab 1's layout."""
    metric_name = "Khoảng cách Cosine" if data["metric"] == "cosine" else "Khoảng cách Euclidean"

    # 1. Performance metrics cards placed at top (matching Tab 1 metrics cards)
    st.markdown("---")
    c_lat, c_bld, c_res = st.columns(3)
    with c_lat:
        st.metric(
            "Thời gian tìm kiếm",
            f"{data['latency_ms']:.2f} ms",
            help="Thời gian thực thi truy vấn tìm kiếm Top-K.",
        )
    with c_bld:
        cached_bld = data.get("build_time_ms")
        if data["algorithm"] == "linear_scan":
            bld_str = f"{cached_bld:.2f} ms" if cached_bld is not None else "Đã chuẩn bị"
            help_tip = "Thời gian nạp và chuẩn bị dữ liệu trong bộ nhớ của Linear Scan."
        else:
            bld_str = f"{cached_bld:.2f} ms" if cached_bld is not None else "Đã chuẩn bị"
            help_tip = "Thời gian chuẩn bị thuật toán trong bộ nhớ."
        st.metric("Thời gian chuẩn bị", bld_str, help=help_tip)
    with c_res:
        st.metric("Số kết quả", len(data["results"]))

    # 2. Results table with standard muted caption
    result_rows = []
    for rank, (record, dist) in enumerate(data["results"], start=1):
        result_rows.append({
            "Thứ hạng": rank,
            "Từ vựng": record.label,
            "Mã ID": record.id,
            metric_name: f"{dist:.4f}",
        })
    st.caption(f"Kết quả tìm kiếm cho '{data['query_name']}' ({metric_name})")
    st.dataframe(result_rows, use_container_width=True, hide_index=True)


def _render_comparison_results(data: Dict[str, Any]) -> None:
    """Render 5-algorithm comparison summary table and individual Top-K inspector."""
    st.markdown("---")
    st.caption(
        f"Đối chiếu 5 thuật toán cho '{data['query_name']}' | "
        f"Độ đo: {'Cosine' if data['metric'] == 'cosine' else 'Euclidean'} | "
        f"Top-K: {data['top_k']} | Chuẩn: Linear Scan (Ground Truth)"
    )

    st.dataframe(data["summary_rows"], use_container_width=True, hide_index=True)

    # Detailed Top-K inspection per algorithm
    detail_dict = data.get("detail_results", {})
    available_algs = [a for a, res in detail_dict.items() if res is not None]

    if available_algs:
        with st.expander("Xem chi tiết danh sách Top-K của từng thuật toán", expanded=False):
            chosen_alg = st.selectbox(
                "Chọn thuật toán",
                options=available_algs,
                format_func=lambda a: ALGORITHM_DISPLAY_NAMES.get(a, a.upper()),
                label_visibility="collapsed",
            )
            alg_res = detail_dict.get(chosen_alg, [])
            metric_col = "Khoảng cách Cosine" if data["metric"] == "cosine" else "Khoảng cách Euclidean"
            detail_rows = []
            for rank, (record, dist) in enumerate(alg_res, start=1):
                detail_rows.append({
                    "Thứ hạng": rank,
                    "Từ vựng": record.label,
                    "Mã ID": record.id,
                    metric_col: f"{dist:.4f}",
                })
            st.dataframe(detail_rows, use_container_width=True, hide_index=True)


def _render_benchmark_section() -> None:
    """Render Section B: Benchmark by dataset size N inside a collapsed expander."""
    st.markdown("---")
    with st.expander("Thử nghiệm hiệu năng (Benchmark theo quy mô N)", expanded=False):
        st.caption("Đo lường thời gian chuẩn bị, thời gian tìm kiếm và Recall@K qua các mốc N = 100, 500, 1.000, 5.000.")

        col_bn, col_bq, col_bbtn = st.columns([3, 2, 2])
        with col_bn:
            st.multiselect(
                "Quy mô tập dữ liệu (N)",
                options=[100, 500, 1000, 5000],
                default=[100, 500, 1000, 5000],
                help="Các mốc quy mô thử nghiệm theo đặc tả đồ án.",
            )
        with col_bq:
            st.number_input(
                "Số câu hỏi mẫu",
                min_value=5,
                max_value=100,
                value=20,
                step=5,
                help="Số lượng vector truy vấn lấy mẫu để đo thời gian trung bình.",
            )
        with col_bbtn:
            st.write("")
            st.write("")
            run_bm_clicked = st.button(
                "Chạy Benchmark",
                type="secondary",
                icon=":material/speed:",
                use_container_width=True,
                help="Thực thi kịch bản đo lường hiệu năng tự động.",
            )

        if run_bm_clicked:
            st.info("Tính năng thử nghiệm hiệu năng đang được hoàn thiện.")


def render_search_tab(manager: VectorEngineManager) -> None:
    """Render the complete Search & Comparison tab matching Tab 1 visual language."""
    if manager.record_count == 0:
        _, center_col, _ = st.columns([1, 2.5, 1])
        with center_col:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-icon">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="11" cy="11" r="7"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                        </svg>
                    </div>
                    <h3 class="empty-state-title">Chưa có dữ liệu tìm kiếm</h3>
                    <p class="empty-state-desc">
                        Vui lòng nạp dữ liệu tại tab <strong>Quản lý dữ liệu</strong> trước khi thực hiện tìm kiếm hoặc so sánh thuật toán.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return

    # Invalidate cached results if dataset was mutated
    if (
        st.session_state.get("_last_dataset_count") != manager.record_count
        or st.session_state.get("_last_dataset_dim") != manager.dimension
    ):
        st.session_state["single_search_result"] = None
        st.session_state["compare_search_result"] = None
        st.session_state["_last_dataset_count"] = manager.record_count
        st.session_state["_last_dataset_dim"] = manager.dimension

    # 1. Shared Query Controls (Top Panel)
    col_alg, col_metric, col_topk = st.columns([2.5, 2.5, 2])

    algorithm_keys = ["linear_scan", "kdtree", "lsh", "ivf", "hnsw"]
    registered_statuses = manager.list_algorithms()
    for k in registered_statuses:
        if k not in algorithm_keys:
            algorithm_keys.append(k)

    with col_alg:
        selected_alg = st.selectbox(
            "Thuật toán",
            options=algorithm_keys,
            format_func=lambda a: ALGORITHM_DISPLAY_NAMES.get(a, a.upper()),
        )
        status = manager.get_algorithm_status(selected_alg)
        if status not in ("ready", "registered"):
            st.caption(":orange[● Chưa sẵn sàng]")

    supported_metrics = manager.get_supported_metrics(selected_alg)
    with col_metric:
        metric_key = st.radio(
            "Độ đo khoảng cách",
            options=supported_metrics,
            format_func=lambda m: "Cosine" if m == "cosine" else "Euclidean",
            horizontal=True,
        )

    with col_topk:
        min_k, max_k, default_k = compute_top_k_bounds(manager.record_count)
        if min_k == max_k:
            st.number_input(
                "Số kết quả (Top-K)",
                min_value=1,
                max_value=1,
                value=1,
                disabled=True,
            )
            top_k = 1
        else:
            top_k = st.slider(
                "Số kết quả (Top-K)",
                min_value=min_k,
                max_value=max_k,
                value=default_k,
            )

    st.markdown("---")

    # Query source and input row
    col_qmode, col_qinput = st.columns([1.2, 2.8])

    with col_qmode:
        query_mode = st.radio(
            "Nguồn truy vấn",
            ["Từ cần tìm", "Tọa độ vector"],
            horizontal=True,
        )

    word_query = ""
    raw_vector_input = ""

    with col_qinput:
        if query_mode == "Từ cần tìm":
            word_query = st.text_input(
                "Từ cần tìm",
                placeholder="Ví dụ: computer, vietnam, apple...",
            )
            if word_query.strip():
                matched_rec = manager.get_record(word_query.strip())
                if matched_rec is not None:
                    st.caption(f":green[✓] Đã tìm thấy từ **{matched_rec.label}** (ID: `{matched_rec.id}`).")
                else:
                    st.caption(":red[✗] Không tìm thấy từ này trong dữ liệu.")
        else:
            dim_guide = manager.dimension if manager.dimension > 0 else 50
            raw_vector_input = st.text_input(
                f"Tọa độ vector ({dim_guide} chiều, cách nhau bởi khoảng trắng hoặc dấu phẩy)",
                placeholder="Ví dụ: 0.12 -0.34 0.56 ...",
            )

    # 2. Advanced Settings (Collapsed Expander - Placed immediately below Query Controls)
    alg_params = _render_advanced_settings(manager, selected_alg, metric_key)

    # 3. Differentiated Actions (Search and Compare Buttons)
    col_btn_search, col_btn_compare, _ = st.columns([1.6, 2.4, 4])
    with col_btn_search:
        search_clicked = st.button(
            "Tìm kiếm",
            type="primary",
            icon=":material/search:",
            use_container_width=True,
        )
    with col_btn_compare:
        compare_clicked = st.button(
            "So sánh 5 thuật toán",
            type="secondary",
            icon=":material/compare_arrows:",
            use_container_width=True,
        )

    # 4. Action Execution
    if search_clicked or compare_clicked:
        query_values: Optional[List[float]] = None
        query_name = ""

        if query_mode == "Từ cần tìm":
            clean_word = word_query.strip()
            if not clean_word:
                st.error("Vui lòng nhập từ cần tìm.")
                return
            rec = manager.get_record(clean_word)
            if rec is None:
                st.error("Không tìm thấy từ này trong dữ liệu.")
                return
            query_values = rec.values
            query_name = rec.label
        else:
            vals, err = parse_vector_input(raw_vector_input, expected_dim=manager.dimension)
            if err:
                st.error(err)
                return
            query_values = vals
            query_name = "Vector tùy ý"

        if search_clicked:
            try:
                with st.spinner(f"Đang tìm kiếm bằng {ALGORITHM_DISPLAY_NAMES.get(selected_alg, selected_alg)}..."):
                    results, latency_ms = manager.search(
                        selected_alg, query_values, top_k=top_k, metric=metric_key, **alg_params
                    )
                bld_ms = manager.get_build_time_ms(selected_alg, metric_key)
                st.session_state["single_search_result"] = {
                    "query_name": query_name,
                    "algorithm": selected_alg,
                    "metric": metric_key,
                    "results": results,
                    "latency_ms": latency_ms,
                    "build_time_ms": bld_ms,
                }
                st.session_state["compare_search_result"] = None
            except (AlgorithmNotReadyError, NotImplementedError):
                st.warning(
                    f"Thuật toán **{ALGORITHM_DISPLAY_NAMES.get(selected_alg, selected_alg)}** "
                    "hiện chưa sẵn sàng."
                )
                st.info("Gợi ý: Hãy chọn thuật toán **Linear Scan** để thực hiện tìm kiếm.")
            except Exception:
                st.error("Không thể hoàn thành tìm kiếm. Vui lòng thử lại sau.")

        elif compare_clicked:
            summary_rows = []
            detail_results: Dict[str, Any] = {}

            with st.spinner("Đang chạy đối sánh 5 thuật toán..."):
                # 1. Ground Truth via Linear Scan
                gt_results: List[Tuple[Any, float]] = []
                gt_latency = 0.0
                try:
                    gt_results, gt_latency = manager.search(
                        "linear_scan", query_values, top_k=top_k, metric=metric_key
                    )
                    gt_ids = set(r.id for r, _ in gt_results)
                    summary_rows.append({
                        "Thuật toán": "Linear Scan",
                        "Trạng thái": "Sẵn sàng (Ground Truth)",
                        "Thời gian tìm kiếm (ms)": f"{gt_latency:.2f}",
                        "Recall@K": "100.0%",
                    })
                    detail_results["linear_scan"] = gt_results
                except Exception:
                    st.error("Không thể thực hiện đối chiếu Ground Truth với Linear Scan.")
                    return

                # 2. Evaluate remaining 4 algorithms
                for algo in ["kdtree", "lsh", "ivf", "hnsw"]:
                    algo_label = ALGORITHM_DISPLAY_NAMES.get(algo, algo)
                    algo_metrics = manager.get_supported_metrics(algo)

                    if metric_key not in algo_metrics:
                        summary_rows.append({
                            "Thuật toán": algo_label,
                            "Trạng thái": "Không hỗ trợ độ đo đã chọn",
                            "Thời gian tìm kiếm (ms)": "—",
                            "Recall@K": "—",
                        })
                        detail_results[algo] = None
                        continue

                    try:
                        cand_results, cand_lat = manager.search(
                            algo, query_values, top_k=top_k, metric=metric_key
                        )
                        cand_ids = set(r.id for r, _ in cand_results)
                        hits = len(cand_ids & gt_ids)
                        recall_val = (hits / len(gt_ids) * 100.0) if gt_ids else 100.0
                        summary_rows.append({
                            "Thuật toán": algo_label,
                            "Trạng thái": "Sẵn sàng",
                            "Thời gian tìm kiếm (ms)": f"{cand_lat:.2f}",
                            "Recall@K": f"{recall_val:.1f}%",
                        })
                        detail_results[algo] = cand_results
                    except (AlgorithmNotReadyError, NotImplementedError):
                        summary_rows.append({
                            "Thuật toán": algo_label,
                            "Trạng thái": "Chưa sẵn sàng",
                            "Thời gian tìm kiếm (ms)": "—",
                            "Recall@K": "—",
                        })
                        detail_results[algo] = None
                    except Exception:
                        summary_rows.append({
                            "Thuật toán": algo_label,
                            "Trạng thái": "Chưa sẵn sàng",
                            "Thời gian tìm kiếm (ms)": "—",
                            "Recall@K": "—",
                        })
                        detail_results[algo] = None

            st.session_state["compare_search_result"] = {
                "query_name": query_name,
                "metric": metric_key,
                "top_k": top_k,
                "summary_rows": summary_rows,
                "detail_results": detail_results,
            }
            st.session_state["single_search_result"] = None

    # 5. Display Rendered Results
    if st.session_state.get("single_search_result"):
        _render_single_search_results(st.session_state["single_search_result"])
    elif st.session_state.get("compare_search_result"):
        _render_comparison_results(st.session_state["compare_search_result"])

    # 6. Section B: Benchmark Section (Placed at bottom)
    _render_benchmark_section()
