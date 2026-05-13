import pytest

pytest.importorskip("gradio")


def test_build_app_returns_blocks():
    import gradio as gr

    from src.ui.app import build_app

    app = build_app()
    assert isinstance(app, gr.Blocks)


def test_app_title():
    from src.ui.app import build_app

    app = build_app()
    assert app.title == "helix-rag"
