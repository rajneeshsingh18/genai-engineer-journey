"""
TaxSaathi - Indian tax assistant chatbot using the Gemini API.

Run:
    python main.py
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types


SYSTEM_PROMPT = """
You are TaxSaathi, a careful Indian tax assistant.

Identity:
- Help Indian taxpayers understand income tax, GST, TDS, ITR filing, deductions,
  and legal tax-saving options.
- Use a professional, friendly tone like a trusted CA friend.
- Respond in the same language as the user. If the user writes Hinglish, use
  natural Hinglish.

Rules:
- Do not fabricate tax rates, deadlines, forms, limits, or legal sections.
- If unsure, say you are not certain and recommend checking the Income Tax India
  portal, GST portal, or a qualified CA.
- Do not help with tax evasion, fake invoices, hiding income, or illegal claims.
- For complex cases, high-value transactions, notices, penalties, audits, or
  litigation, recommend consulting a CA or tax professional.
- Prefer clear steps, examples, and short calculations when useful.
- Mention that tax rules can change and should be verified before filing.
""".strip()


@dataclass
class ChatConfig:
    """Centralized chatbot settings."""

    model: str = "gemini-2.5-flash"
    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 800
    max_history_messages: int = 16


@dataclass
class Message:
    """One conversation turn."""

    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ConversationStats:
    """Usage counters returned by Gemini when available."""

    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0

    def add_usage(self, usage_metadata: object | None) -> None:
        self.total_calls += 1

        if usage_metadata is None:
            return

        self.total_prompt_tokens += int(
            getattr(usage_metadata, "prompt_token_count", 0) or 0
        )
        self.total_output_tokens += int(
            getattr(usage_metadata, "candidates_token_count", 0) or 0
        )
        self.total_tokens += int(getattr(usage_metadata, "total_token_count", 0) or 0)


class TaxSaathiBot:
    """Gemini-powered chatbot with memory, stats, and JSON export."""

    def __init__(self, config: Optional[ChatConfig] = None) -> None:
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found. Create a .env file with "
                "GEMINI_API_KEY=your_key_here or set it in your terminal."
            )

        self.config = config or ChatConfig()
        self.client = genai.Client(api_key=api_key)
        self.history: list[Message] = []
        self.stats = ConversationStats()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sessions_dir = Path(__file__).parent / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)

    def _gemini_history(self, user_input: str) -> list[types.Content]:
        turns = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in self.history
        ]
        turns.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        )
        return turns

    def _trim_history(self) -> None:
        while len(self.history) > self.config.max_history_messages:
            self.history.pop(0)

    def chat(self, user_input: str) -> str:
        user_input = user_input.strip()
        if not user_input:
            return "Please enter a tax question."

        try:
            response = self.client.models.generate_content(
                model=self.config.model,
                contents=self._gemini_history(user_input),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    max_output_tokens=self.config.max_output_tokens,
                ),
            )
        except Exception as exc:
            return (
                "I could not reach Gemini right now. Please check your API key, "
                f"internet connection, or quota. ({type(exc).__name__})"
            )

        assistant_reply = (response.text or "").strip()
        if not assistant_reply:
            assistant_reply = "I could not generate a response. Please try again."

        self.stats.add_usage(getattr(response, "usage_metadata", None))
        self.history.append(Message(role="user", content=user_input))
        self.history.append(Message(role="assistant", content=assistant_reply))
        self._trim_history()

        return assistant_reply

    def reset(self) -> None:
        self.history.clear()

    def stats_display(self) -> str:
        return (
            "Session Stats | "
            f"Calls: {self.stats.total_calls} | "
            f"Prompt tokens: {self.stats.total_prompt_tokens:,} | "
            f"Output tokens: {self.stats.total_output_tokens:,} | "
            f"Total tokens: {self.stats.total_tokens:,}"
        )

    def export_conversation(self) -> Path:
        export_data = {
            "session_id": self.session_id,
            "model": self.config.model,
            "config": asdict(self.config),
            "stats": asdict(self.stats),
            "messages": [asdict(message) for message in self.history],
            "exported_at": datetime.now().isoformat(),
        }

        export_path = self.sessions_dir / f"taxsaathi_session_{self.session_id}.json"
        export_path.write_text(
            json.dumps(export_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return export_path


def print_help() -> None:
    print("\nCommands:")
    print("  help   - Show commands")
    print("  stats  - Show Gemini token usage")
    print("  reset  - Clear chat memory")
    print("  export - Save this conversation as JSON")
    print("  quit   - Export and exit")
    print("\nAsk anything about Indian tax, GST, TDS, deductions, or ITR filing.\n")


def run_cli() -> None:
    print("=" * 64)
    print("TaxSaathi - Indian Tax Assistant powered by Gemini")
    print("=" * 64)
    print("Type 'help' for commands or 'quit' to exit.\n")

    bot = TaxSaathiBot()
    print("TaxSaathi: Namaste! Ask me an Indian tax question.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except KeyboardInterrupt:
            print("\nUse 'quit' to export and exit cleanly.")
            continue

        if not user_input:
            continue

        command = user_input.lower()
        if command in {"quit", "exit", "bye"}:
            export_path = bot.export_conversation()
            print(f"\nConversation exported to: {export_path}")
            print(bot.stats_display())
            print("Goodbye!")
            break

        if command == "help":
            print_help()
            continue

        if command == "stats":
            print(bot.stats_display())
            continue

        if command == "reset":
            bot.reset()
            print("Conversation memory cleared.")
            continue

        if command == "export":
            export_path = bot.export_conversation()
            print(f"Conversation exported to: {export_path}")
            continue

        print("\nTaxSaathi: ", end="", flush=True)
        response = bot.chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    run_cli()

