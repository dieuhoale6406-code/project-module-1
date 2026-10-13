import math
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st

try:
    from core.manager import VectorEngineManager
except ImportError:
    from src.core.manager import VectorEngineManager


def set_flash_message(message: str, level: str = "success") -> None:
    """Store a feedback message in session state to persist across st.rerun()."""
    st.session_state["_flash_message"] = (level, message)


def consume_flash_message() -> Optional[Tuple[str, str]]:
    """Retrieve and clear the flash feedback message from session state."""
    return st.session_state.pop("_flash_message", None)


def display_flash_message() -> None:
    """Display any flash message saved before the previous st.rerun()."""
    flash = consume_flash_message()
    if flash:
        level, msg = flash
        if level == "success":
            st.success(msg)
        elif level == "info":
            st.info(msg)
        elif level == "warning":
            st.warning(msg)
        elif level == "error":
            st.error(msg)


def parse_glove_line(line: str, expected_dim: int = 50) -> Optional[Tuple[str, List[float]]]:
    """
    Parse a single line from a GloVe-format vector file.
    Supports labels containing whitespace by extracting the last expected_dim tokens as coordinates.
    """
    raw = line.strip()
    if not raw:
        return None

    tokens = raw.split()
    if len(tokens) <= expected_dim:
        return None

    # Reject lines with more numeric coordinates than expected (dimension mismatch).
    try:
        float(tokens[-expected_dim - 1])
        return None
    except ValueError:
        pass

    label = " ".join(tokens[:-expected_dim]).strip().lower()
    if not label:
        return None

    try:
        values = [float(x) for x in tokens[-expected_dim:]]
    except ValueError:
        return None

    for v in values:
        if not math.isfinite(v):
            return None

    return label, values


def load_glove_subset(
    manager: VectorEngineManager,
    file_path: str,
    limit: int = 10000,
    expected_dim: int = 50,
) -> Tuple[int, int]:
    """
    Stream a subset of GloVe vectors into manager without loading the full file into memory.
    Appends valid records and safely skips duplicates and invalid lines.
    Returns (loaded_count, skipped_count).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Tập tin không tồn tại: {file_path}")

    loaded_count = 0
    skipped_count = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if loaded_count >= limit:
                break
            parsed = parse_glove_line(line, expected_dim=expected_dim)
            if parsed is None:
                skipped_count += 1
                continue
            label, values = parsed
            try:
                manager.add_record(label=label, values=values, track_undo=False)
                loaded_count += 1
            except ValueError:
                skipped_count += 1

    return loaded_count, skipped_count


def parse_vector_input(
    raw: str, expected_dim: Optional[int] = None
) -> Tuple[Optional[List[float]], Optional[str]]:
    """
    Parse and validate a string of numeric vector coordinates.
    Accepts space- or comma-separated floating point numbers.
    Returns (coordinates, error_message).
    """
    clean = raw.strip()
    if not clean:
        return None, "Vui lòng nhập tọa độ vector."

    tokens = clean.replace(",", " ").split()
    values: List[float] = []
    for token in tokens:
        try:
            val = float(token)
            if not math.isfinite(val):
                return None, "Tọa độ vector chứa giá trị không hợp lệ (NaN hoặc vô cực)."
            values.append(val)
        except ValueError:
            return None, f"Tọa độ '{token}' không phải là số thực hợp lệ."

    if expected_dim is not None and expected_dim > 0 and len(values) != expected_dim:
        return (
            None,
            f"Số chiều vector ({len(values)}) không khớp với hệ thống ({expected_dim}).",
        )

    return values, None


def render_dataset_tab(manager: VectorEngineManager) -> None:
    """Render the Data Workspace with dataset centerpiece, contextual toolbar, and record detail actions."""
    display_flash_message()

    # Helpful empty state when N = 0
    if manager.record_count == 0:
        _, center_col, _ = st.columns([1, 2.5, 1])
        with center_col:
            st.markdown(
                """
                <div class="empty-state-card">
                    <div class="empty-state-icon">
                        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="17 8 12 3 7 8"></polyline>
                            <line x1="12" y1="3" x2="12" y2="15"></line>
                        </svg>
                    </div>
                    <h3 class="empty-state-title">Chưa có dữ liệu</h3>
                    <p class="empty-state-desc">
                        Nạp tập dữ liệu mẫu hoặc thêm từ mới để bắt đầu tìm kiếm và phân tích vector.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Primary action: Nạp nhanh 1.000 từ mẫu
            if st.button("Nạp 1.000 từ mẫu", type="primary", icon=":material/bolt:", use_container_width=True):
                try:
                    with st.spinner("Đang nạp 1.000 từ vựng mẫu..."):
                        loaded, skipped = load_glove_subset(
                            manager, "data/vectors.txt", limit=1000, expected_dim=50
                        )
                    set_flash_message(f"Đã nạp thành công {loaded:,} từ vựng mẫu.", "success")
                    st.rerun()
                except Exception:
                    st.error("Không thể đọc tệp dữ liệu mẫu. Vui lòng kiểm tra lại tệp.")

            c_sub1, c_sub2 = st.columns(2)
            with c_sub1:
                with st.popover("Tùy chọn nạp", icon=":material/upload:", use_container_width=True):
                    st.markdown("**Nạp từ tệp dữ liệu**")
                    e_path = st.text_input("Đường dẫn tệp", value="data/vectors.txt", key="e_load_path")
                    e_limit = st.selectbox(
                        "Số lượng từ",
                        options=[100, 500, 1000, 5000, 10000],
                        index=2,
                        format_func=lambda x: f"{x:,}",
                        key="e_load_limit",
                    )
                    if st.button("Bắt đầu nạp", type="primary", use_container_width=True, key="e_load_btn"):
                        try:
                            with st.spinner(f"Đang nạp {e_limit:,} từ vựng..."):
                                loaded, skipped = load_glove_subset(
                                    manager, e_path, limit=int(e_limit), expected_dim=50
                                )
                            set_flash_message(f"Đã nạp thành công {loaded:,} từ vựng.", "success")
                            st.rerun()
                        except Exception:
                            st.error("Không thể đọc tệp dữ liệu. Vui lòng kiểm tra lại đường dẫn.")

            with c_sub2:
                with st.popover("Thêm từ mới", icon=":material/add:", use_container_width=True):
                    st.markdown("**Thêm từ vựng mới**")
                    e_lbl = st.text_input("Từ vựng", placeholder="Ví dụ: technology", key="e_add_lbl")
                    e_vals_str = st.text_input(
                        "Tọa độ (50 giá trị số)", placeholder="Ví dụ: 0.12 -0.34 0.56 ...", key="e_add_vals"
                    )
                    if st.button("Lưu từ vựng", type="primary", use_container_width=True, key="e_add_btn"):
                        if not e_lbl.strip():
                            st.error("Vui lòng nhập từ vựng.")
                        else:
                            vals, err = parse_vector_input(e_vals_str, expected_dim=50)
                            if err:
                                st.error(err)
                            else:
                                try:
                                    manager.add_record(label=e_lbl.strip(), values=vals)
                                    set_flash_message(f"Đã thêm từ '{e_lbl.strip()}'.", "success")
                                    st.rerun()
                                except ValueError as val_err:
                                    st.error(str(val_err))
                                except Exception:
                                    st.error("Không thể thêm từ vựng. Vui lòng kiểm tra lại dữ liệu.")
        return

    # 1. Dataset Overview Metrics (Undo history shown only when available)
    undo_count: Optional[int] = None
    if manager.undo_stack is not None and hasattr(manager.undo_stack, "size"):
        try:
            undo_count = manager.undo_stack.size()
        except (NotImplementedError, Exception):
            undo_count = None

    if undo_count is not None:
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Tổng số từ", f"{manager.record_count:,}")
        with col_m2:
            st.metric("Số chiều", manager.dimension)
        with col_m3:
            st.metric("Có thể hoàn tác", f"{undo_count} thao tác", help="Số lượng thay đổi gần nhất có thể hoàn tác.")
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Tổng số từ", f"{manager.record_count:,}")
        with col_m2:
            st.metric("Số chiều", manager.dimension)

    st.markdown("---")

    # 2. Contextual Toolbar
    col_search, col_load, col_add, col_undo = st.columns([3.5, 1.4, 1.4, 1.2])

    with col_search:
        search_filter = st.text_input(
            "Tìm kiếm từ vựng",
            placeholder="Tìm theo từ vựng hoặc mã ID...",
            label_visibility="collapsed",
        )

    with col_load:
        with st.popover("Nạp thêm", icon=":material/upload:", use_container_width=True):
            st.markdown("**Nạp thêm dữ liệu**")
            tb_path = st.text_input("Đường dẫn tệp", value="data/vectors.txt", key="tb_load_path")
            tb_limit = st.selectbox(
                "Số lượng từ",
                options=[100, 500, 1000, 5000, 10000],
                index=2,
                format_func=lambda x: f"{x:,}",
                key="tb_load_limit",
            )
            if st.button("Nạp dữ liệu", type="primary", use_container_width=True, key="tb_load_btn"):
                try:
                    with st.spinner(f"Đang nạp {tb_limit:,} từ vựng..."):
                        loaded, skipped = load_glove_subset(
                            manager, tb_path, limit=int(tb_limit), expected_dim=manager.dimension or 50
                        )
                    set_flash_message(
                        f"Đã nạp thành công {loaded:,} từ vựng.", "success"
                    )
                    st.rerun()
                except Exception:
                    st.error("Không thể đọc tệp dữ liệu. Vui lòng kiểm tra lại đường dẫn.")

    with col_add:
        with st.popover("Thêm từ", icon=":material/add:", use_container_width=True):
            st.markdown("**Thêm từ vựng mới**")
            dim_guide = manager.dimension if manager.dimension > 0 else 50
            tb_add_lbl = st.text_input("Từ vựng", placeholder="Ví dụ: technology", key="tb_add_lbl")
            tb_add_vals = st.text_input(
                f"Tọa độ ({dim_guide} giá trị số)", placeholder="Ví dụ: 0.12 -0.34 0.56 ...", key="tb_add_vals"
            )
            if st.button("Lưu từ vựng", type="primary", use_container_width=True, key="tb_add_btn"):
                if not tb_add_lbl.strip():
                    st.error("Vui lòng nhập từ vựng.")
                else:
                    vals, err = parse_vector_input(tb_add_vals, expected_dim=manager.dimension)
                    if err:
                        st.error(err)
                    else:
                        try:
                            manager.add_record(label=tb_add_lbl.strip(), values=vals)
                            set_flash_message(f"Đã thêm từ '{tb_add_lbl.strip()}'.", "success")
                            st.rerun()
                        except ValueError as val_err:
                            st.error(str(val_err))
                        except Exception:
                            st.error("Không thể thêm từ vựng. Vui lòng kiểm tra lại dữ liệu.")

    with col_undo:
        if st.button("Hoàn tác", icon=":material/undo:", use_container_width=True, help="Hoàn tác thay đổi dữ liệu gần nhất."):
            try:
                msg = manager.undo()
                if msg:
                    set_flash_message(msg, "success")
                    st.rerun()
                else:
                    st.info("Không có thao tác nào để hoàn tác.")
            except (NotImplementedError, Exception):
                st.info("Tính năng hoàn tác hiện chưa sẵn sàng.")

    # 3. Centerpiece Dataset Table
    all_recs = manager.get_all_records()
    if search_filter.strip():
        matched = manager.get_record(search_filter.strip())
        display_recs = [matched] if matched is not None else []
        if not display_recs:
            st.info(f"Không tìm thấy từ vựng nào khớp với '{search_filter.strip()}'.")
    else:
        display_recs = all_recs[:50]

    if display_recs:
        preview_rows = []
        for r in display_recs:
            coord_preview = f"[{', '.join(f'{v:.2f}' for v in r.values[:4])}, ...]"
            preview_rows.append({
                "Mã ID": r.id,
                "Từ vựng": r.label,
                "Tọa độ rút gọn": coord_preview,
                "Độ dài L2 (Norm)": f"{r.norm:.4f}",
            })
        caption_text = (
            f"Hiển thị {len(display_recs)} / {manager.record_count:,} từ vựng"
            if not search_filter.strip()
            else f"Kết quả tìm kiếm cho '{search_filter.strip()}'"
        )
        st.caption(caption_text)
        st.dataframe(preview_rows, use_container_width=True, hide_index=True)

        # 4. Record Inspector & Contextual Detail Actions
        with st.expander("Chi tiết từ vựng & Thao tác (Cập nhật / Xóa)", expanded=False):
            inspect_opts = [f"{r.label} ({r.id})" for r in display_recs]
            selected_inspect = st.selectbox(
                "Chọn từ vựng cần xem hoặc thao tác",
                options=inspect_opts,
                label_visibility="collapsed",
            )
            if selected_inspect:
                sel_label = selected_inspect.split(" (")[0]
                inspect_rec = next((r for r in display_recs if r.label == sel_label), None)
                if inspect_rec:
                    st.caption(
                        f"Mã ID: `{inspect_rec.id}` | Từ vựng: **{inspect_rec.label}** | "
                        f"Độ dài L2: `{inspect_rec.norm:.4f}` | Số chiều: {inspect_rec.dimension}"
                    )
                    st.code(" ".join(f"{v:.4f}" for v in inspect_rec.values), language="text")

                    # Contextual actions directly for this record
                    c_upd, c_del = st.columns([1, 1])
                    with c_upd:
                        with st.popover("Cập nhật tọa độ", icon=":material/edit:", use_container_width=True):
                            st.markdown(f"**Cập nhật cho '{inspect_rec.label}'**")
                            cur_vals_str = " ".join(f"{v:.4f}" for v in inspect_rec.values)
                            new_vals_str = st.text_input(
                                "Tọa độ mới", value=cur_vals_str, key=f"upd_{inspect_rec.id}"
                            )
                            if st.button("Lưu cập nhật", type="primary", use_container_width=True, key=f"btn_upd_{inspect_rec.id}"):
                                vals, err = parse_vector_input(new_vals_str, expected_dim=manager.dimension)
                                if err:
                                    st.error(err)
                                else:
                                    try:
                                        if manager.update_record(label=inspect_rec.label, values=vals):
                                            set_flash_message(f"Đã cập nhật từ '{inspect_rec.label}'.", "success")
                                            st.rerun()
                                        else:
                                            st.error("Không tìm thấy từ này trong dữ liệu.")
                                    except ValueError as val_err:
                                        st.error(str(val_err))
                                    except Exception:
                                        st.error("Không thể cập nhật từ vựng. Vui lòng thử lại.")

                    with c_del:
                        with st.popover("Xóa từ này", icon=":material/delete:", use_container_width=True):
                            st.markdown(f"Xác nhận xóa từ **{inspect_rec.label}** (`{inspect_rec.id}`)?")
                            st.caption(":red[Thao tác xóa sẽ loại bỏ từ này khỏi tập dữ liệu hiện tại.]")
                            if st.button("Xác nhận xóa", type="primary", use_container_width=True, key=f"btn_del_{inspect_rec.id}"):
                                try:
                                    if manager.delete_record(label=inspect_rec.label):
                                        set_flash_message(f"Đã xóa từ '{inspect_rec.label}'.", "success")
                                        st.rerun()
                                    else:
                                        st.error("Không tìm thấy từ này trong dữ liệu.")
                                except Exception:
                                    st.error("Không thể xóa từ vựng. Vui lòng thử lại.")
