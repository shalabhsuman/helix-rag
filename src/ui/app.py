"""Gradio chat interface for helix-rag.

Wraps the OpenAI agent in a browser-based chat UI.
Conversation history is maintained per session via gr.State.
Run with: python src/ui/app.py
"""

import os

import gradio as gr
from agents import Runner
from dotenv import load_dotenv
from loguru import logger

from src.agent.agent import build_agent

load_dotenv()

TITLE = "helix-rag"
DESCRIPTION = "Agentic search across any document collection. Every answer grounded and cited."
PLACEHOLDER = "Ask a question about your documents, or type 'what papers do you have?'"


def respond(message: str, chat_history: list, agent_history: list) -> tuple:
    if not message.strip():
        return "", chat_history, agent_history

    input_data = agent_history + [{"role": "user", "content": message}] if agent_history else message

    try:
        result = Runner.run_sync(build_agent(), input_data)
        reply = result.final_output
        new_agent_history = result.to_input_list()
    except Exception as e:
        logger.error(f"Agent error: {e}")
        reply = f"Something went wrong: {e}"
        new_agent_history = agent_history

    chat_history = chat_history + [[message, reply]]
    return "", chat_history, new_agent_history


def build_app() -> gr.Blocks:
    with gr.Blocks(title=TITLE) as app:
        agent_history = gr.State([])

        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        chatbot = gr.Chatbot(
            value=[],
            height=460,
            show_label=False,
        )

        with gr.Row():
            msg = gr.Textbox(
                placeholder=PLACEHOLDER,
                show_label=False,
                scale=9,
                container=False,
            )
            send_btn = gr.Button("Send", scale=1, variant="primary")

        clear_btn = gr.Button("Clear conversation", size="sm", variant="secondary")

        send_btn.click(
            respond,
            inputs=[msg, chatbot, agent_history],
            outputs=[msg, chatbot, agent_history],
        )
        msg.submit(
            respond,
            inputs=[msg, chatbot, agent_history],
            outputs=[msg, chatbot, agent_history],
        )
        clear_btn.click(
            lambda: ([], []),
            outputs=[chatbot, agent_history],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("UI_PORT", 7860)),
        theme=gr.themes.Soft(),
    )
