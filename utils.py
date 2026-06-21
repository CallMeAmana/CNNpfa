import streamlit.components.v1 as components


def inject_sidebar_toggle():
    """No-op: the sidebar is fixed by page config and shared CSS.

    The previous hamburger button was removed to keep the sidebar visible
    and avoid inconsistent toggle behavior across pages.
    """
    return
