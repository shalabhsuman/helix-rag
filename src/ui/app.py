"""Gradio chat interface for helix-rag.

Wraps the OpenAI agent in a browser-based chat UI with streaming responses.
Conversation history is maintained per session via gr.State.
Run with: python src/ui/app.py
"""

import os
from collections.abc import AsyncGenerator

from dotenv import load_dotenv

load_dotenv()  # must run before agent import so LANGFUSE_SECRET_KEY is set

import gradio as gr  # noqa: E402
from agents import RunConfig, Runner  # noqa: E402
from loguru import logger  # noqa: E402

from src.agent.agent import build_agent  # noqa: E402

TITLE = "helix-rag"
DESCRIPTION = "Agentic search across any document collection. Every answer grounded and cited."
PLACEHOLDER = "Ask a question about your documents, or type 'what papers do you have?'"


async def respond(
    message: str, chat_history: list, agent_history: list
) -> AsyncGenerator[tuple, None]:
    if not message.strip():
        yield "", chat_history, agent_history
        return

    input_data = (
        agent_history + [{"role": "user", "content": message}] if agent_history else message
    )
    trace_name = f"helix-rag | {message[:60]}"

    # Show user message immediately with a searching placeholder
    chat_history = chat_history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "_Searching papers..._"},
    ]
    yield "", chat_history, agent_history

    try:
        partial_reply = ""
        stream_result = Runner.run_streamed(
            build_agent(),
            input_data,
            run_config=RunConfig(workflow_name=trace_name),
        )
        async for event in stream_result.stream_events():
            if event.type == "raw_response_event":
                data = event.data
                if hasattr(data, "type") and data.type == "response.output_text.delta":
                    partial_reply += data.delta
                    chat_history[-1]["content"] = partial_reply
                    yield "", chat_history, agent_history

        new_agent_history = stream_result.to_input_list()
        if not partial_reply:
            chat_history[-1]["content"] = stream_result.final_output
        yield "", chat_history, new_agent_history

    except Exception as e:
        logger.error(f"Agent error: {e}")
        chat_history[-1]["content"] = f"Something went wrong: {e}"
        yield "", chat_history, agent_history


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
        server_name="127.0.0.1",
        server_port=int(os.getenv("UI_PORT", 7860)),
        theme=gr.themes.Soft(),
    )
