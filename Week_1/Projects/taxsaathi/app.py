"""
Gradio UI for TaxSaathi.

Run:
    python app.py
"""

from __future__ import annotations

from typing import Any

import gradio as gr

from main import ChatConfig, Message, TaxSaathiBot


LANGUAGE_CHOICES = [
    ("Auto detect", "auto"),
    ("English", "english"),
    ("Hindi", "hindi"),
]


def _history_to_messages(history: list[Any] | None) -> list[Message]:
    """Convert Gradio chat history into TaxSaathi message objects."""
    messages: list[Message] = []

    for item in history or []:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append(Message(role=role, content=content))
            continue

        if isinstance(item, (list, tuple)) and len(item) >= 2:
            user_message, assistant_message = item[0], item[1]
            if user_message:
                messages.append(Message(role="user", content=str(user_message)))
            if assistant_message:
                messages.append(Message(role="assistant", content=str(assistant_message)))

    return messages


def respond(message: str, history: list[Any], language: str) -> str:
    """Handle one Gradio chat turn."""
    try:
        bot = TaxSaathiBot(ChatConfig(language=language))
    except EnvironmentError as exc:
        return (
            f"Setup error: {exc}\n\n"
            "Create `Week_1/Projects/taxsaathi/.env` and add:\n"
            "`GEMINI_API_KEY=your_real_gemini_key`"
        )

    bot.history = _history_to_messages(history)
    return bot.chat(message)


def build_demo() -> gr.ChatInterface:
    language_dropdown = gr.Dropdown(
        choices=LANGUAGE_CHOICES,
        value="auto",
        label="Output language",
        filterable=False,
    )

    return gr.ChatInterface(
        fn=respond,
        title="TaxSaathi",
        description="Indian tax assistant powered by Gemini.",
        additional_inputs=[language_dropdown],
        additional_inputs_accordion=gr.Accordion("Settings", open=True),
        examples=[
            ["Explain section 80C in simple terms.", "english"],
            ["Meri salary 9 lakh hai, kaunsa tax regime better hoga?", "hindi"],
            ["What is the due date for ITR filing?", "auto"],
        ],
        flagging_mode="never",
    )


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
