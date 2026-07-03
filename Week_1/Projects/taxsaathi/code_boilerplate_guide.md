# TaxSaathi - Step-by-Step Code Boilerplate Guide

This guide breaks down the core architecture of TaxSaathi into simplified, boilerplate-style Python snippets. It shows the essential syntax and structure of the main components.

---

## 📖 Table of Contents
1. [Environment & Gemini Client Setup](#1-environment--gemini-client-setup)
2. [Conversational Memory & Buffer Trimming](#2-conversational-memory--buffer-trimming)
3. [Executing a Chat Turn with System Instructions](#3-executing-a-chat-turn-with-system-instructions)
4. [Extracting API Token Usage Metadata](#4-extracting-api-token-usage-metadata)
5. [Exporting Chat History to JSON](#5-exporting-chat-history-to-json)
6. [Interactive Terminal CLI Command Loop](#6-interactive-terminal-cli-command-loop)
7. [Gradio Chat UI Interface and Sharing](#7-gradio-chat-ui-interface-and-sharing)

---

## 1. Environment & Gemini Client Setup
This boilerplate demonstrates how to load environment variables from a `.env` file and initialize the modern Google GenAI Client.

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai

# 1. Load the local configuration (.env)
project_dir = Path(__file__).parent
load_dotenv(project_dir / ".env")

# 2. Retrieve the API Key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise EnvironmentError("Missing GEMINI_API_KEY in your environment.")

# 3. Instantiate the unified client
client = genai.Client(api_key=api_key)
```

---

## 2. Conversational Memory & Buffer Trimming
To maintain conversation memory, we define a structured message schema and a FIFO queue buffer to trim older interactions.

```python
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class Message:
    role: str       # "user" or "assistant" (model)
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ChatMemory:
    def __init__(self, max_history_messages: int = 16):
        self.history: list[Message] = []
        self.max_history_messages = max_history_messages

    def add_message(self, role: str, content: str):
        self.history.append(Message(role=role, content=content))
        self._trim_history()

    def _trim_history(self):
        # Trims older messages in a First-In, First-Out (FIFO) queue manner
        while len(self.history) > self.max_history_messages:
            self.history.pop(0)

    def clear(self):
        self.history.clear()
```

---

## 3. Executing a Chat Turn with System Instructions
This boilerplate translates the internal message history into the formats expected by the Gemini API, attaches system-level guardrails, and calls the model.

```python
from google.genai import types

def run_chat_turn(client: genai.Client, system_prompt: str, history: list[Message], new_input: str) -> str:
    # 1. Format history to google.genai.types.Content format
    api_history = []
    for msg in history:
        api_role = "model" if msg.role == "assistant" else "user"
        api_history.append(
            types.Content(
                role=api_role,
                parts=[types.Part.from_text(text=msg.content)]
            )
        )
    
    # 2. Append new user query
    api_history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=new_input)]
        )
    )
    
    # 3. Call generate_content with model, system instruction, and parameters
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=api_history,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.2,
            max_output_tokens=800,
        )
    )
    
    return response.text
```

---

## 4. Extracting API Token Usage Metadata
Every response from the Google GenAI SDK returns token metrics. This snippet extracts and sums these tokens for analytical tracking.

```python
@dataclass
class TokenStats:
    prompt_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0

    def record_usage(self, response):
        self.total_calls += 1
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self.prompt_tokens += getattr(usage, "prompt_token_count", 0) or 0
            self.output_tokens += getattr(usage, "candidates_token_count", 0) or 0
            self.total_tokens += getattr(usage, "total_token_count", 0) or 0
```

---

## 5. Exporting Chat History to JSON
This boilerplate formats configuration settings, usage statistics, and conversational transcript history, and writes them to a JSON file.

```python
import json
from dataclasses import asdict

def export_session_to_json(session_id: str, history: list[Message], stats: TokenStats, export_dir: Path) -> Path:
    export_dir.mkdir(exist_ok=True)
    
    payload = {
        "session_id": session_id,
        "exported_at": datetime.now().isoformat(),
        "stats": asdict(stats),
        "messages": [asdict(msg) for msg in history]
    }
    
    file_path = export_dir / f"session_{session_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
    return file_path
```

---

## 6. Interactive Terminal CLI Command Loop
A command-line interface loops for inputs and handles system-level directives (such as clearing session memory, modifying output language, and quitting).

```python
def run_cli_loop(bot_instance):
    print("Welcome! Type 'help' for options or 'quit' to exit.")
    
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        
        command = user_input.lower()
        
        if command == "quit":
            bot_instance.export_conversation()
            print("Session saved. Goodbye!")
            break
            
        elif command == "reset":
            bot_instance.reset_history()
            print("Chat history cleared.")
            
        elif command == "stats":
            print(f"Total Calls: {bot_instance.stats.total_calls}")
            
        else:
            # Regular LLM chat turn
            response = bot_instance.chat(user_input)
            print(f"Assistant: {response}")
```

---

## 7. Gradio Chat UI Interface and Sharing
This snippet mounts our logic inside the standard Gradio Web interface framework, converting history elements and launching with a public sharing link enabled.

```python
import gradio as gr

def gradio_respond(message: str, history: list, language: str) -> str:
    # 1. Translate Gradio history format to internal format
    internal_history = []
    for turn in history or []:
        # Gradio history is often a list of lists: [[user_msg, assistant_msg], ...]
        if isinstance(turn, (list, tuple)) and len(turn) >= 2:
            if turn[0]:
                internal_history.append(Message(role="user", content=turn[0]))
            if turn[1]:
                internal_history.append(Message(role="assistant", content=turn[1]))
    
    # 2. Invoke bot instance chat with settings
    response = my_bot.chat_with_history(message, internal_history, language)
    return response

# 3. Create Interface UI
def build_and_run_ui():
    demo = gr.ChatInterface(
        fn=gradio_respond,
        title="TaxSaathi UI",
        additional_inputs=[
            gr.Dropdown(choices=["auto", "english", "hindi"], value="auto", label="Language")
        ]
    )
    
    # 4. Launch web application (share=True creates a public URL)
    demo.launch(share=True)
```
