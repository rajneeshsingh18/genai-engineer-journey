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


LANGUAGE_INSTRUCTIONS = {
    "auto": "Reply in the same language as the user. If they mix Hindi and English, use natural Hinglish.",
    "english": "Reply only in clear English, even if the user writes in another language.",
    "hindi": "Reply only in Hindi using Devanagari script, even if the user writes in English.",
}

LANGUAGE_LABELS = {
    "auto": "Auto detect",
    "english": "English",
    "hindi": "Hindi",
}


@dataclass
class ChatConfig:
    """Centralized chatbot settings."""

    model: str = "gemini-3.5-flash"
    temperature: float = 0.2
    top_p: float = 0.9
    max_output_tokens: int = 800
    max_history_messages: int = 16
    language: str = "auto"


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
        project_dir = Path(__file__).parent
        env_path = project_dir / ".env"
        load_dotenv(env_path)

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError(
                f"Gemini API key not found. Create {env_path} with "
                "GEMINI_API_KEY=your_key_here. Do not put the real key in "
                ".env.example."
            )

        self.config = config or ChatConfig()
        self.client = genai.Client(api_key=api_key)
        self.history: list[Message] = []
        self.stats = ConversationStats()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.sessions_dir = project_dir / "sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.set_language(self.config.language)

    def set_language(self, language: str) -> None:
        """Set response language to auto, English, or Hindi."""
        self.config.language = self._normalize_language(language)

    def _normalize_language(self, language: str) -> str:
        cleaned = (language or "auto").strip().lower()
        aliases = {
            "auto detect": "auto",
            "auto-detect": "auto",
            "automatic": "auto",
            "eng": "english",
            "en": "english",
            "hi": "hindi",
            "hin": "hindi",
        }
        cleaned = aliases.get(cleaned, cleaned)
        if cleaned not in LANGUAGE_INSTRUCTIONS:
            return "auto"
        return cleaned

    def _system_instruction(self) -> str:
        language_rule = LANGUAGE_INSTRUCTIONS[self.config.language]
        return f"{SYSTEM_PROMPT}\n\nOutput language rule: {language_rule}"

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
                    system_instruction=self._system_instruction(),
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
            f"Total tokens: {self.stats.total_tokens:,} | "
            f"Language: {LANGUAGE_LABELS[self.config.language]}"
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
    print("  help              - Show commands")
    print("  stats             - Show Gemini token usage")
    print("  language auto     - Match the user's language")
    print("  language english  - Always answer in English")
    print("  language hindi    - Always answer in Hindi")
    print("  reset             - Clear chat memory")
    print("  export            - Save this conversation as JSON")
    print("  quit              - Export and exit")
    print("\nAsk anything about Indian tax, GST, TDS, deductions, or ITR filing.\n")


def run_cli() -> None:
    print("=" * 64)
    print("TaxSaathi - Indian Tax Assistant powered by Gemini")
    print("=" * 64)
    print("Type 'help' for commands or 'quit' to exit.\n")

    try:
        bot = TaxSaathiBot()
    except EnvironmentError as exc:
        print(f"Setup error: {exc}")
        print("\nFix:")
        print("  1. In Week_1\\Projects\\taxsaathi, create a file named .env")
        print("  2. Add this line: GEMINI_API_KEY=your_real_gemini_key")
        print("  3. Run again: python main.py")
        return

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
        if command in {"language", "lang"}:
            print(f"Current language: {LANGUAGE_LABELS[bot.config.language]}")
            print("Use: language auto | language english | language hindi")
            continue

        if command.startswith("language ") or command.startswith("lang "):
            requested_language = command.split(maxsplit=1)[1]
            normalized_language = bot._normalize_language(requested_language)
            cleaned_language = requested_language.strip().lower()
            accepted_inputs = set(LANGUAGE_INSTRUCTIONS) | {
                "auto detect",
                "auto-detect",
                "automatic",
                "eng",
                "en",
                "hi",
                "hin",
            }
            if cleaned_language not in accepted_inputs:
                print("Unknown language. Use: auto, english, or hindi.")
                continue

            bot.set_language(normalized_language)
            print(f"Response language set to: {LANGUAGE_LABELS[bot.config.language]}")
            continue

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














