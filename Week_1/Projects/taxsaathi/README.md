# TaxSaathi

TaxSaathi is a Gemini-powered CLI chatbot for Indian tax questions. It keeps
short conversation memory, tracks token usage returned by Gemini, and can export
chat sessions to JSON.

## Setup

```bash
cd Week_1/Projects/taxsaathi
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

You can get a Gemini API key from Google AI Studio.

## Run

```bash
python main.py
```

## Commands

- `help` shows commands.
- `stats` shows token usage for the session.
- `reset` clears chat memory.
- `export` saves the conversation to `sessions/`.
- `quit` exports the conversation and exits.

## Notes

TaxSaathi is for learning and general guidance. Indian tax rules change, and
answers should be verified before filing. For personalized tax advice, consult a
qualified CA or tax professional.
