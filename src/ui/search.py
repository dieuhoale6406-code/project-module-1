from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    from core.manager import AlgorithmNotReadyError, VectorEngineManager
    from core.benchmark import BenchmarkRunner
    from core.algorithms.linear_scan import LinearScanIndex
    from core.algorithms.kdtree import KDTreeIndex
except ImportError:
    from src.core.manager import AlgorithmNotReadyError, VectorEngineManager
    from src.core.benchmark import BenchmarkRunner
    from src.core.algorithms.linear_scan import LinearScanIndex
    from src.core.algorithms.kdtree import KDTreeIndex

try:
    from core.algorithms.ivf import IVFIndex
except ImportError:
    try:
        from src.core.algorithms.ivf import IVFIndex
    except ImportError:
        IVFIndex = None

try:
    from ui.dataset import parse_vector_input
    from ui.query_queue import enqueue_query, render_query_queue, sync_query_queue
except ImportError:
    from src.ui.dataset import parse_vector_input
    from src.ui.query_queue import enqueue_query, render_query_queue, sync_query_queue


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


def _render_algo_prepare_button(
    manager: VectorEngineManager, algo_key: str, metric_key: str, params: Dict[str, Any]
) -> None:
    """Render preparation status badge and manual prepare button for a specific algorithm."""
    effective_metric = "euclidean" if algo_key in ("kdtree", "ivf") else metric_key
    cached_bld_ms = manager.get_build_time_ms(algo_key, effective_metric)
    raw_status = manager.get_algorithm_status(algo_key)

    if cached_bld_ms is not None:
        status_text = f":green[● Đã chuẩn bị] ({cached_bld_ms:.2f} ms)"
    elif raw_status in ("not_implemented", "unavailable"):
        status_text = ":orange[● Chưa sẵn sàng]"
    else:
        status_text = ":gray[● Sẵn sàng (Tự động chuẩn bị khi tìm kiếm)]"

    c_info, c_btn = st.columns([3, 1])
    with c_info:
        st.markdown(f"Trạng thái: {status_text}")
    with c_btn:
        btn_prep = st.button(
            "Chuẩn bị",
            key=f"btn_prep_{algo_key}",
            type="secondary",
            icon=":material/tune:",
            use_container_width=True,
            help=f"Chuẩn bị trước cho thuật toán {ALGORITHM_DISPLAY_NAMES.get(algo_key, algo_key)}.",
        )
    if btn_prep:
        try:
            with st.spinner(f"Đang chuẩn bị {ALGORITHM_DISPLAY_NAMES.get(algo_key, algo_key)}..."):
                bld_ms = manager.build_index(algo_key, metric=effective_metric, **params)
            st.success(f"Chuẩn bị thành công trong **{bld_ms:.2f} ms**.")
            st.rerun()
        except (AlgorithmNotReadyError, NotImplementedError):
            st.warning(f"Thuật toán **{ALGORITHM_DISPLAY_NAMES.get(algo_key, algo_key)}** hiện chưa sẵn sàng.")
        except Exception as exc:
            st.error(f"Không thể chuẩn bị thuật toán: {exc}")


def _render_advanced_settings(
    manager: VectorEngineManager, selected_alg: str, metric_key: str
) -> Dict[str, Dict[str, Any]]:
    """Render collapsed advanced configuration with algorithm parameters for all 4 algorithms."""
    all_alg_params: Dict[str, Dict[str, Any]] = {
        "linear_scan": {},
        "kdtree": {},
        "ivf": {},
        "lsh": {},
    }

    with st.expander("⚙️ Cài đặt nâng cao (Tham số thuật toán & Chuẩn bị chỉ mục)", expanded=False):
        st.caption(
            "Tùy chỉnh thông số chi tiết cho cả 4 thuật toán. "
            "Khi bấm **So sánh 4 thuật toán**, hệ thống sẽ áp dụng đồng thời các thông số này "
            "dưới cùng độ đo Euclidean để đảm bảo tính khách quan và nhất quán với kết quả thực nghiệm trong slide."
        )

        tab_ivf, tab_lsh, tab_kdtree, tab_linear = st.tabs([
            "IVF-Flat (ANN)",
            "LSH (Multi-Table ANN)",
            "KD-Tree (Exact)",
            "Linear Scan (Baseline)",
        ])

        with tab_ivf:
            st.markdown("**Thuật toán IVF-Flat (Inverted File Index)**")
            st.caption("Phân cụm không gian bằng K-Means và chỉ dò tìm trong các cụm tâm gần nhất.")
            c_ivf1, c_ivf2 = st.columns(2)
            with c_ivf1:
                all_alg_params["ivf"]["num_clusters"] = st.slider(
                    "Số cụm (num_clusters)",
                    min_value=2,
                    max_value=64,
                    value=8,
                    help="Số tâm cụm K-Means. Khuyến nghị chuẩn thực nghiệm: 8 cụm.",
                    key="adv_ivf_clusters",
                )
            with c_ivf2:
                all_alg_params["ivf"]["nprobe"] = st.slider(
                    "Số cụm dò tìm (nprobe)",
                    min_value=1,
                    max_value=16,
                    value=2,
                    help="Số cụm gần nhất được quét khi truy vấn. Khuyến nghị chuẩn thực nghiệm: 2 cụm (Recall ~97%, Speedup ~3.5x).",
                    key="adv_ivf_nprobe",
                )
            _render_algo_prepare_button(manager, "ivf", metric_key, all_alg_params["ivf"])

        with tab_lsh:
            st.markdown("**Thuật toán LSH (Locality-Sensitive Hashing)**")
            st.caption("Băm ngẫu nhiên qua các siêu phẳng độc lập và gom nhóm theo bucket nhị phân.")
            c_lsh1, c_lsh2, c_lsh3 = st.columns(3)
            with c_lsh1:
                all_alg_params["lsh"]["num_hyperplanes"] = st.slider(
                    "Số siêu phẳng (m)",
                    min_value=2,
                    max_value=16,
                    value=8,
                    help="Số siêu phẳng băm ngẫu nhiên trên mỗi bảng. Khuyến nghị chuẩn thực nghiệm: 8 (Recall ~76%, Speedup ~9.3x).",
                    key="adv_lsh_hyperplanes",
                )
            with c_lsh2:
                all_alg_params["lsh"]["num_tables"] = st.slider(
                    "Số bảng băm (L)",
                    min_value=1,
                    max_value=8,
                    value=2,
                    help="Số bảng băm độc lập (Multi-Table LSH). Chuẩn thực nghiệm: 2 bảng.",
                    key="adv_lsh_tables",
                )
            with c_lsh3:
                all_alg_params["lsh"]["seed"] = st.number_input(
                    "Hạt giống ngẫu nhiên (Seed)",
                    value=42,
                    step=1,
                    help="Cố định seed để kết quả băm ngẫu nhiên tái lập chuẩn xác.",
                    key="adv_lsh_seed",
                )
            _render_algo_prepare_button(manager, "lsh", metric_key, all_alg_params["lsh"])

        with tab_kdtree:
            st.markdown("**Thuật toán KD-Tree (k-d Tree Exact Search)**")
            st.caption(
                "Phân vùng không gian bằng cây nhị phân trực giao trục tọa độ. "
                "Cấu trúc cây được tự động tối ưu theo số chiều dữ liệu d (không cần tinh chỉnh tham số ngoài)."
            )
            _render_algo_prepare_button(manager, "kdtree", metric_key, all_alg_params["kdtree"])

        with tab_linear:
            st.markdown("**Thuật toán Linear Scan (Baseline Ground Truth)**")
            st.caption(
                "Duyệt tuần tự qua 100% dữ liệu để tìm kiếm Top-K chính xác tuyệt đối (Ground Truth). "
                "Dữ liệu được nạp trực tiếp trong bộ nhớ."
            )
            _render_algo_prepare_button(manager, "linear_scan", metric_key, all_alg_params["linear_scan"])

    return all_alg_params


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
        f"Đối chiếu 4 thuật toán cho '{data['query_name']}' | "
        f"Độ đo: {'Cosine' if data['metric'] == 'cosine' else 'Euclidean (Độ đo chung)'} | "
        f"Top-K: {data['top_k']} | Chuẩn: Linear Scan (Ground Truth)"
    )

    st.dataframe(data["summary_rows"], use_container_width=True, hide_index=True)

    # Dynamic Trade-off analysis notes (thu gọn trong expander để giao diện sạch sẽ, chỉ mở khi cần thuyết trình)
    analysis_notes = data.get("analysis_notes", [])
    if analysis_notes:
        with st.expander("Phân tích đánh đổi & Lý giải kết quả (Trade-off Insights)", expanded=False):
            for note in analysis_notes:
                st.markdown(f"- {note}")

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


def _render_benchmark_section(manager: Optional[VectorEngineManager] = None) -> None:
    """Render Section B: Benchmark by dataset size N inside a collapsed expander."""
    st.markdown("---")
    with st.expander("Thử nghiệm hiệu năng (Benchmark theo quy mô N)", expanded=False):
        st.caption("Đo lường thời gian chuẩn bị, thời gian tìm kiếm và Recall@K qua các mốc N = 100, 500, 1.000, 5.000.")

        col_bn, col_bq, col_bbtn = st.columns([3, 2, 2])
        with col_bn:
            selected_scales = st.multiselect(
                "Quy mô tập dữ liệu (N)",
                options=[100, 500, 1000, 5000],
                default=[100, 500, 1000, 5000],
                help="Các mốc quy mô thử nghiệm theo đặc tả đồ án.",
            )
        with col_bq:
            query_count = st.number_input(
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
            if manager is None or manager.record_count == 0:
                st.warning("Vui lòng nạp dữ liệu ở Tab 'QUẢN LÝ DỮ LIỆU' trước khi thực thi benchmark.")
            elif not selected_scales:
                st.warning("Vui lòng chọn ít nhất một mốc quy mô N để benchmark.")
            else:
                with st.spinner("Đang thực thi đo lường hiệu năng đa chỉ số trên các mốc quy mô N..."):
                    all_records = (
                        manager.get_all_records()
                        if hasattr(manager, "get_all_records")
                        else list(getattr(manager, "_records", {}).values())
                    )
                    dim = manager.dimension or 50

                    candidate_algos: Dict[str, Any] = {
                        "linear_scan": LinearScanIndex(metric="euclidean"),
                    }
                    try:
                        candidate_algos["kdtree"] = KDTreeIndex(dimension=dim)
                    except Exception:
                        pass
                    if IVFIndex is not None:
                        try:
                            candidate_algos["ivf"] = IVFIndex(
                                dimension=dim, num_clusters=8, nprobe=2, metric="euclidean"
                            )
                        except Exception:
                            pass
                    try:
                        from core.algorithms.lsh import LSHIndex
                        candidate_algos["lsh"] = LSHIndex(
                            dimension=dim, num_hyperplanes=8, num_tables=2, seed=42, metric="euclidean"
                        )
                    except Exception:
                        pass

                    active_algos: Dict[str, Any] = {}
                    sample_probe = all_records[:2] if len(all_records) >= 2 else all_records
                    for k, algo in candidate_algos.items():
                        try:
                            algo.build(sample_probe)
                            algo.search(sample_probe[0].values, top_k=1)
                            active_algos[k] = algo
                        except (NotImplementedError, Exception):
                            continue

                    runner = BenchmarkRunner(full_records=all_records, dimension=dim)
                    results = runner.run_benchmark(
                        algorithm_instances=active_algos,
                        scale_sizes=selected_scales,
                        top_k=5,
                        query_count=int(query_count),
                    )
                    st.session_state["benchmark_results"] = results

        bm_results = st.session_state.get("benchmark_results")
        if bm_results:
            st.markdown("#### Bảng tổng hợp đa chỉ số hiệu năng")
            display_rows = []
            for r in bm_results:
                speedup_val = r.get("speedup", 1.0)
                display_rows.append({
                    "Quy mô N": r["n"],
                    "Thuật toán": r["algorithm"],
                    "Bản chất": r.get("category", "—"),
                    "Tăng tốc (Speedup)": f"{speedup_val:.2f}x",
                    "Build Time (ms)": f"{r['build_time_ms']:.2f}",
                    "Search Time TB (ms)": f"{r['mean_latency_ms']:.3f}",
                    "P95 Latency (ms)": f"{r['p95_latency_ms']:.3f}",
                    "QPS": f"{r['qps']:.1f}",
                    "Recall@K (%)": f"{r['recall_percent']:.1f}%",
                })
            st.dataframe(display_rows, use_container_width=True, hide_index=True)

            st.markdown("#### Biểu đồ đối chiếu trực quan theo quy mô N")
            col_chart1, col_chart2, col_chart3 = st.columns(3)

            import pandas as pd
            df_bm = pd.DataFrame(bm_results)

            with col_chart1:
                st.markdown("**Thời gian tìm kiếm TB (ms)**")
                chart_latency = df_bm.pivot(index="n", columns="algorithm", values="mean_latency_ms")
                st.line_chart(chart_latency)

            with col_chart2:
                st.markdown("**Hệ số tăng tốc (Speedup vs GT)**")
                chart_speedup = df_bm.pivot(index="n", columns="algorithm", values="speedup")
                st.line_chart(chart_speedup)

            with col_chart3:
                st.markdown("**Tỷ lệ thu hồi (Recall %)**")
                chart_recall = df_bm.pivot(index="n", columns="algorithm", values="recall_percent")
                st.line_chart(chart_recall)


def render_search_tab(manager: VectorEngineManager) -> None:
    """Render the complete Search & Comparison tab matching Tab 1 visual language."""
    sync_query_queue(manager)
    # Invalidate cached results if dataset was mutated, including when empty.
    if st.session_state.get("_last_dataset_version") != manager.mutation_version:
        st.session_state["single_search_result"] = None
        st.session_state["compare_search_result"] = None
        st.session_state["_last_dataset_version"] = manager.mutation_version

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

    # 1. Shared Query Controls (Top Panel)
    col_alg, col_metric, col_topk = st.columns([2.5, 2.5, 2])

    algorithm_keys = ["linear_scan", "kdtree", "lsh", "ivf"]
    registered_statuses = manager.list_algorithms()
    for k in registered_statuses:
        if k not in algorithm_keys and k != "hnsw":
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
    all_alg_params = _render_advanced_settings(manager, selected_alg, metric_key)
    current_alg_params = all_alg_params.get(selected_alg, {})

    # 3. Differentiated Actions (Search and Compare Buttons)
    col_btn_search, col_btn_compare, col_btn_queue = st.columns([1.6, 2.4, 2.4])
    with col_btn_search:
        search_clicked = st.button(
            "Tìm kiếm",
            type="primary",
            icon=":material/search:",
            use_container_width=True,
        )
    with col_btn_compare:
        compare_clicked = st.button(
            "So sánh 4 thuật toán",
            type="secondary",
            icon=":material/compare_arrows:",
            use_container_width=True,
            help="Tự động chuẩn bị chỉ mục và đối sánh đồng thời cả 4 thuật toán trên độ đo Euclidean.",
        )

    with col_btn_queue:
        enqueue_clicked = st.button("Thêm vào hàng đợi")

    # 4. Action Execution
    if search_clicked or compare_clicked or enqueue_clicked:
        query_values: Optional[List[float]] = None
        query_name = ""

        if query_mode == "Từ cần tìm":
            clean_word = word_query.strip()
            if not clean_word:
                st.error("Vui lòng nhập từ cần tìm.")
                render_query_queue(manager, ALGORITHM_DISPLAY_NAMES)
                return
            rec = manager.get_record(clean_word)
            if rec is None:
                st.error("Không tìm thấy từ này trong dữ liệu.")
                render_query_queue(manager, ALGORITHM_DISPLAY_NAMES)
                return
            query_values = rec.values
            query_name = rec.label
        else:
            vals, err = parse_vector_input(raw_vector_input, expected_dim=manager.dimension)
            if err:
                st.error(err)
                render_query_queue(manager, ALGORITHM_DISPLAY_NAMES)
                return
            query_values = vals
            query_name = "Vector tùy ý"

        if enqueue_clicked:
            enqueue_query(query_name, query_values, selected_alg, top_k, metric_key, current_alg_params)
            st.success(f"Đã thêm truy vấn '{query_name}' vào hàng đợi.")

        elif search_clicked:
            try:
                with st.spinner(f"Đang tìm kiếm bằng {ALGORITHM_DISPLAY_NAMES.get(selected_alg, selected_alg)}..."):
                    results, latency_ms = manager.search(
                        selected_alg, query_values, top_k=top_k, metric=metric_key, **current_alg_params
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
            analysis_notes: List[str] = []

            algo_display_names = {
                "linear_scan": "Linear Scan (Exact)",
                "kdtree": "KD-Tree (Exact)",
                "ivf": "IVF-Flat (ANN)",
                "lsh": "LSH (ANN)",
            }

            cmp_metric = "euclidean"
            build_times: Dict[str, Optional[float]] = {}

            with st.spinner("Đang chuẩn bị chỉ mục và đối sánh đồng thời 4 thuật toán (Độ đo Euclidean)..."):
                # 0. Pre-build all 4 algorithms explicitly to strictly isolate build time from query search latency
                for algo in ["linear_scan", "kdtree", "ivf", "lsh"]:
                    params = all_alg_params.get(algo, {})
                    try:
                        bld_ms = manager.build_index(algo, metric=cmp_metric, **params)
                        build_times[algo] = bld_ms
                    except Exception:
                        build_times[algo] = None

                # 1. Ground Truth via Linear Scan
                gt_results: List[Tuple[Any, float]] = []
                gt_latency = 0.0
                try:
                    gt_results, gt_latency = manager.search(
                        "linear_scan", query_values, top_k=top_k, metric=cmp_metric
                    )
                    gt_ids = set(r.id for r, _ in gt_results)
                    bld_val = build_times.get("linear_scan")
                    bld_str = f"{bld_val:.2f} ms" if bld_val is not None else "Đã chuẩn bị"
                    summary_rows.append({
                        "Thuật toán": algo_display_names["linear_scan"],
                        "Thời gian dựng (Build)": bld_str,
                        "Thời gian tìm kiếm (ms)": f"{gt_latency:.2f}",
                        "Tốc độ (Speedup)": "1.00x (Chuẩn)",
                        "Recall@K": "100.0%",
                    })
                    detail_results["linear_scan"] = gt_results
                except Exception:
                    st.error("Không thể thực hiện đối chiếu Ground Truth với Linear Scan.")
                    return

                # 2. Evaluate candidate algorithms (KD-Tree, IVF-Flat, LSH)
                for algo in ["kdtree", "ivf", "lsh"]:
                    algo_label = algo_display_names.get(algo, algo.upper())
                    search_kwargs = dict(all_alg_params.get(algo, {}))
                    bld_val = build_times.get(algo)
                    bld_str = f"{bld_val:.2f} ms" if bld_val is not None else "—"

                    try:
                        cand_results, cand_lat = manager.search(
                            algo, query_values, top_k=top_k, metric=cmp_metric, **search_kwargs
                        )
                        cand_ids = set(r.id for r, _ in cand_results)
                        hits = len(cand_ids & gt_ids)
                        recall_val = (hits / len(gt_ids) * 100.0) if gt_ids else 100.0
                        speedup = (gt_latency / cand_lat) if cand_lat > 0 else 1.0

                        summary_rows.append({
                            "Thuật toán": algo_label,
                            "Thời gian dựng (Build)": bld_str,
                            "Thời gian tìm kiếm (ms)": f"{cand_lat:.2f}",
                            "Tốc độ (Speedup)": f"{speedup:.2f}x",
                            "Recall@K": f"{recall_val:.1f}%",
                        })
                        detail_results[algo] = cand_results

                        if algo == "kdtree":
                            dim_info = f"d={manager.dimension}" if manager.dimension else "không gian nhiều chiều"
                            analysis_notes.append(
                                f"**KD-Tree (Exact)**: Recall đạt **{recall_val:.1f}%** (chính xác tuyệt đối), "
                                f"tốc độ đạt **{speedup:.2f}x** so với Linear Scan ({cand_lat:.2f} ms vs {gt_latency:.2f} ms). "
                                f"Tại {dim_info}, do hiệu ứng Lời nguyền số chiều (Curse of Dimensionality), "
                                f"hình cầu lân cận giao cắt với phần lớn các siêu phẳng phân chia nên cây phải duyệt lại (backtrack) "
                                f"hầu hết các nhánh, khiến tốc độ tăng tốc khiêm tốn (~1.4x)."
                            )
                        elif algo == "ivf":
                            nprobe_val = search_kwargs.get("nprobe", 2)
                            num_c_val = search_kwargs.get("num_clusters", 8)
                            analysis_notes.append(
                                f"**IVF-Flat (ANN)**: Đạt tốc độ tăng tốc **{speedup:.2f}x** "
                                f"({cand_lat:.2f} ms so với {gt_latency:.2f} ms của Linear Scan) với Recall@K đạt **{recall_val:.1f}%** "
                                f"(num_clusters={num_c_val}, nprobe={nprobe_val}). "
                                f"Thuật toán chỉ quét trong {nprobe_val}/{num_c_val} cụm Voronoi gần nhất nên giảm mạnh số lượng phép tính khoảng cách."
                            )
                        elif algo == "lsh":
                            m_val = search_kwargs.get("num_hyperplanes", 8)
                            L_val = search_kwargs.get("num_tables", 2)
                            analysis_notes.append(
                                f"**LSH (Locality-Sensitive Hashing - ANN)**: Đạt tốc độ tăng tốc **{speedup:.2f}x** "
                                f"({cand_lat:.2f} ms so với {gt_latency:.2f} ms của Linear Scan) với Recall@K đạt **{recall_val:.1f}%** "
                                f"(m={m_val} siêu phẳng, L={L_val} bảng băm). "
                                f"Thuật toán gom các vector tương đồng vào cùng bucket nhị phân và mở rộng thăm dò lân cận (Hamming radius=1), "
                                f"giảm thiểu tối đa số ứng viên cần tính khoảng cách."
                            )
                    except (AlgorithmNotReadyError, NotImplementedError):
                        summary_rows.append({
                            "Thuật toán": algo_label,
                            "Thời gian dựng (Build)": bld_str,
                            "Thời gian tìm kiếm (ms)": "Chưa sẵn sàng",
                            "Tốc độ (Speedup)": "—",
                            "Recall@K": "—",
                        })
                        detail_results[algo] = None
                    except Exception:
                        summary_rows.append({
                            "Thuật toán": algo_label,
                            "Thời gian dựng (Build)": bld_str,
                            "Thời gian tìm kiếm (ms)": "Lỗi thực thi",
                            "Tốc độ (Speedup)": "—",
                            "Recall@K": "—",
                        })
                        detail_results[algo] = None

            st.session_state["compare_search_result"] = {
                "query_name": query_name,
                "metric": cmp_metric,
                "top_k": top_k,
                "summary_rows": summary_rows,
                "detail_results": detail_results,
                "analysis_notes": analysis_notes,
            }
            st.session_state["single_search_result"] = None

    # 5. Display Rendered Results
    if st.session_state.get("single_search_result"):
        _render_single_search_results(st.session_state["single_search_result"])
    elif st.session_state.get("compare_search_result"):
        _render_comparison_results(st.session_state["compare_search_result"])

    render_query_queue(manager, ALGORITHM_DISPLAY_NAMES)

    # 6. Section B: Benchmark Section (Placed at bottom)
    _render_benchmark_section(manager)
