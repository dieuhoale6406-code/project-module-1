import sys
from pathlib import Path

import streamlit as st

# Ensure both src and project root are in sys.path for teammate modules using dual import styles (core.* and src.core.*)
src_dir = Path(__file__).resolve().parent
root_dir = src_dir.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from core.manager import VectorEngineManager

try:
    from ui.dataset import load_glove_subset, parse_glove_line, render_dataset_tab
    from ui.search import (
        ALGORITHM_DISPLAY_NAMES,
        format_algorithm_status_badge,
        render_search_tab,
    )
except ImportError:
    from src.ui.dataset import load_glove_subset, parse_glove_line, render_dataset_tab
    from src.ui.search import (
        ALGORITHM_DISPLAY_NAMES,
        format_algorithm_status_badge,
        render_search_tab,
    )

# Re-exports for backward compatibility
__all__ = [
    "render_app",
    "load_glove_subset",
    "parse_glove_line",
    "format_algorithm_status_badge",
    "ALGORITHM_DISPLAY_NAMES",
]


def load_css() -> None:
    """Load shared design system stylesheet from src/ui/styles.css."""
    css_path = Path(__file__).resolve().parent / "ui" / "styles.css"
    if css_path.is_file():
        css_content = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)


def _render_header(is_dark: bool) -> None:
    """Render the persistent application header with brand identity and theme switch."""
    with st.container(key="app_header"):
        col_brand, col_theme = st.columns([11, 1], vertical_alignment="center")
        with col_brand:
            st.markdown(
                """
                <div class="app-header-brand">
                    <div class="app-header-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                            <circle cx="11" cy="11" r="7"></circle>
                            <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
                            <line x1="11" y1="8" x2="11" y2="14"></line>
                            <line x1="8" y1="11" x2="14" y2="11"></line>
                        </svg>
                    </div>
                    <span class="app-title">Mini Vector Search Engine</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_theme:
            theme_icon = ":material/light_mode:" if is_dark else ":material/dark_mode:"
            theme_help = "Chuyển sang giao diện Sáng" if is_dark else "Chuyển sang giao diện Tối"
            if st.button(
                "",
                key="theme_toggle_btn",
                icon=theme_icon,
                help=theme_help,
            ):
                st.session_state["theme"] = "light" if is_dark else "dark"
                st.rerun()


def render_app() -> None:
    """Render the polished, responsive Streamlit UI supporting Light and Dark themes."""
    st.set_page_config(
        page_title="Mini Vector Search Engine",
        layout="wide",
    )

    # 1. Initialize theme in session state (default to Light Mode)
    if "theme" not in st.session_state:
        st.session_state["theme"] = "light"

    # 2. Inject shared stylesheet from src/ui/styles.css
    load_css()

    # 3. Active theme marker for CSS variable switching
    is_dark = st.session_state["theme"] == "dark"
    if is_dark:
        st.markdown('<span class="theme-marker-dark" style="display:none;"></span>', unsafe_allow_html=True)

    # 4. Engine Manager singleton in session state
    if "manager" not in st.session_state or not hasattr(st.session_state.manager, "list_algorithms"):
        st.session_state.manager = VectorEngineManager()

    manager: VectorEngineManager = st.session_state.manager

    # 5. Header branding & theme toggle
    _render_header(is_dark)

    # 6. Primary navigation tabs
    tab_data, tab_search = st.tabs([
        "QUẢN LÝ DỮ LIỆU",
        "TÌM KIẾM & SO SÁNH",
    ])

    with tab_data:
        render_dataset_tab(manager)

    with tab_search:
        render_search_tab(manager)


if __name__ == "__main__":
    render_app()