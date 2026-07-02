# TaxSaathi

TaxSaathi is a Gemini-powered chatbot for Indian tax questions. It includes a
terminal CLI and a Gradio web UI, supports English/Hindi output, keeps short
conversation memory, and can export CLI chat sessions to JSON.

## Setup

```bash
cd Week_1/Projects/taxsaathi
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a real `.env` file in this same folder. Do not put your real key in
`.env.example` because example files are usually committed to Git.

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

PowerShell shortcut:

```powershell
copy .env.example .env
notepad .env
```

You can get a Gemini API key from Google AI Studio.

## Run CLI

```bash
python main.py
```

CLI commands:

- `language auto` matches the user's language.
- `language english` always answers in English.
- `language hindi` always answers in Hindi.
- `stats` shows token usage for the session.
- `reset` clears chat memory.
- `export` saves the conversation to `sessions/`.
- `quit` exports the conversation and exits.

## Run Gradio UI

```bash
python app.py
```

Open the local URL printed by Gradio, usually `http://127.0.0.1:7860`.
Use the **Output language** dropdown to choose Auto detect, English, or Hindi.

## Notes

TaxSaathi is for learning and general guidance. Indian tax rules change, and
answers should be verified before filing. For personalized tax advice, consult a
qualified CA or tax professional.
