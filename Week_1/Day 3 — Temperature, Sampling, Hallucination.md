Yesterday you learned *how* Transformers think (attention, Q/K/V). Today you learn *how they decide what to say* — and you'll ship your first real production chatbot by end of day.

This is the day everything becomes tangible. Let's go.

---

## 🎯 1. How Does an LLM Actually Generate Text?

After all those attention layers, the Transformer outputs one thing:

> **A probability distribution over the entire vocabulary for the next token.**
> 

```
Input: "The capital of India is"

Output probability distribution:
├── "New"      → 68.3%   ← most likely
├── "Delhi"    → 12.1%
├── "Mumbai"   → 4.2%
├── "a"        → 2.1%
├── "the"      → 1.8%
├── "located"  → 1.1%
└── [other 100K tokens] → remaining %
```

The model doesn't "know" the answer. It learned that **"New" statistically follows "The capital of India is"** more than anything else — from training on millions of documents.

This one insight explains **everything** — why LLMs are powerful, why they hallucinate, and why temperature matters.

### The Generation Loop

```
┌─────────────────────────────────────────────────────┐
│                  GENERATION LOOP                     │
│                                                      │
│  "The capital of India is"                           │
│            │                                         │
│            ▼                                         │
│    [Transformer Forward Pass]                        │
│            │                                         │
│            ▼                                         │
│    Logits: raw scores for all 100K+ tokens           │
│            │                                         │
│            ▼                                         │
│    Apply Temperature                                 │
│            │                                         │
│            ▼                                         │
│    Apply Sampling Strategy (top-k / top-p)           │
│            │                                         │
│            ▼                                         │
│    Sample → "New"                                    │
│            │                                         │
│            ▼                                         │
│  "The capital of India is New"                       │
│            │                                         │
│            ▼                                         │
│    [Repeat until EOS token or max_tokens]            │
└─────────────────────────────────────────────────────┘
```

Each token is generated **one at a time**, feeding back into the next forward pass. This is called **autoregressive generation**.

---

## 🌡️ 2. Temperature — The Most Misunderstood Parameter

### What It Actually Does

Temperature **reshapes the probability distribution** before sampling.

```
Raw logits (before softmax):
"New" = 4.2,  "Delhi" = 2.1,  "Mumbai" = 1.8,  "a" = 0.9
```

```
temperature = 1.0 (default):
softmax([4.2, 2.1, 1.8, 0.9]) → [0.683, 0.121, 0.092, 0.041, ...]
(proportional to training distribution)

temperature = 0.2 (low — more deterministic):
softmax([4.2/0.2, 2.1/0.2, ...]) = softmax([21, 10.5, 9, 4.5])
→ [0.998, 0.001, 0.0003, ...]  ← "New" dominates almost completely

temperature = 2.0 (high — more random):
softmax([4.2/2.0, 2.1/2.0, ...]) = softmax([2.1, 1.05, 0.9, 0.45])
→ [0.412, 0.215, 0.185, 0.118, ...]  ← more evenly spread
```

The math: `logits_scaled = logits / temperature` — then softmax as normal.

### Visual Intuition

```
Temperature = 0.1        Temperature = 1.0        Temperature = 2.0
(Very Peaked)            (Normal)                 (Very Flat)

    █                        █                    ▄ ▄ ▃ ▂ ▂ ▂
    █                        █ ▃                  ▄ ▄ ▃ ▂ ▂ ▂
    █ ▁ ▁ ▁ ▁ ▁             █ █ ▂ ▁ ▁            ▄ ▄ ▃ ▂ ▂ ▂
 [New][D][M][a][t][l]    [New][D][M][a][t]    [New][D][M][a][t][l]

Almost always picks       Balanced — follows     Almost random
the top token             training distribution  across many tokens
```

### When to Use What

| Use Case | Temperature | Why |
| --- | --- | --- |
| Tax / Legal / Medical facts | 0.0 – 0.2 | You need consistency and accuracy |
| Code generation | 0.2 – 0.4 | Mostly deterministic, some variation |
| Customer support bot | 0.3 – 0.5 | Accurate but natural sounding |
| General chatbot | 0.7 – 0.9 | Conversational and varied |
| Creative writing | 1.0 – 1.4 | Novel ideas, unexpected phrasing |
| Brainstorming | 1.2 – 1.5 | Diverse options, even weird ones |
| **Never use** | > 2.0 | Incoherent garbage |

> **Indian Interview Tip:** "We set temperature to 0.1 for our IRCTC refund status bot because users need deterministic, accurate answers — not creative interpretations of railway policy."
> 

---

## 🎲 3. Sampling Strategies — Top-K and Top-P

Temperature adjusts the distribution. Sampling strategies decide **which part of the distribution to sample from**.

### Greedy Sampling (temperature = 0)

Always pick the highest probability token.

```
"New" = 68.3% → always pick "New"
```

**Problem:** Repetitive and boring. Loops endlessly. *"The cat sat on the mat. The cat sat on the mat. The cat sat..."*

### Top-K Sampling

Only sample from the **K most likely tokens**, ignore the rest.

```python
K = 3
Vocabulary:
  "New"    → 68.3%  ✅ in top-3
  "Delhi"  → 12.1%  ✅ in top-3
  "Mumbai" → 4.2%   ✅ in top-3
  "a"      → 2.1%   ❌ excluded
  [others] → rest   ❌ excluded

Renormalize top-3 → sample from [72.5%, 12.8%, 4.7%]
```

**Problem:** K=50 might be too many when distribution is sharp; too few when it's flat. K is fixed regardless of context.

### Top-P (Nucleus) Sampling — The Modern Standard

Sample from the **smallest set of tokens whose cumulative probability ≥ P**.

```python
P = 0.9
Sorted vocabulary:
  "New"      → 68.3%   cumulative: 68.3%
  "Delhi"    → 12.1%   cumulative: 80.4%
  "Mumbai"   → 4.2%    cumulative: 84.6%
  "a"        → 2.1%    cumulative: 86.7%
  "the"      → 1.8%    cumulative: 88.5%
  "located"  → 1.1%    cumulative: 89.6%
  "in"       → 0.9%    cumulative: 90.5%  ← first to exceed 0.9 → stop here

Nucleus = ["New", "Delhi", "Mumbai", "a", "the", "located", "in"]
Sample from this nucleus (renormalized)
```

**Why it's better:** When distribution is sharp (model is confident), nucleus is small (1–3 tokens). When flat (uncertain), nucleus is large (many tokens). It adapts to context.

### Frequency and Presence Penalty

These prevent repetition — critical for production chatbots:

```
frequency_penalty: penalizes tokens proportional to how often they've appeared
presence_penalty:  penalizes tokens that have appeared at all (binary)

Both range from -2.0 to 2.0
0.0 = no penalty (default)
0.5-1.0 = reasonable anti-repetition
```

---

## 💻 4. Code — Sampling from Scratch

```python
"""
Sampling Strategies Implementation
Week 1, Day 3 — Understanding how LLMs decide what to say
"""

import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Apply temperature scaling to logits.

    Args:
        logits: Raw model output scores (before softmax)
        temperature: Scaling factor
                     < 1.0 → sharper distribution (more deterministic)
                     > 1.0 → flatter distribution (more random)
                     = 0.0 → greedy (always pick highest, handled separately)

    Returns:
        Temperature-scaled logits

    Critical: Never divide by 0. Clamp temperature to small positive value.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive. Use greedy() for temp=0.")

    return logits / temperature

def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    # Subtract max to prevent overflow — same math, more stable
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

def greedy_sample(logits: np.ndarray) -> int:
    """Always pick the highest-probability token. No randomness."""
    return int(np.argmax(logits))

def top_k_sample(logits: np.ndarray, k: int, temperature: float = 1.0) -> int:
    """
    Sample from the top-K most likely tokens.

    Args:
        logits: Raw model scores
        k: Number of top tokens to consider
        temperature: Applied before sampling

    Returns:
        Sampled token index
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Scale logits
    scaled = apply_temperature(logits, temperature)

    # Find top-k indices
    top_k_indices = np.argsort(scaled)[-k:]
    top_k_logits = scaled[top_k_indices]

    # Convert to probabilities (only over top-k)
    top_k_probs = softmax(top_k_logits)

    # Sample from this restricted distribution
    chosen_idx = np.random.choice(len(top_k_indices), p=top_k_probs)
    return int(top_k_indices[chosen_idx])

def top_p_sample(logits: np.ndarray, p: float, temperature: float = 1.0) -> int:
    """
    Nucleus sampling — sample from smallest set of tokens with cum. prob >= p.

    This is the industry standard for production LLM generation.

    Args:
        logits: Raw model scores
        p: Cumulative probability threshold (typically 0.9 or 0.95)
        temperature: Applied before sampling

    Returns:
        Sampled token index
    """
    if not 0 < p <= 1.0:
        raise ValueError(f"p must be in (0, 1], got {p}")

    scaled = apply_temperature(logits, temperature)
    probs = softmax(scaled)

    # Sort by probability — descending
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    # Find the nucleus — cumulative sum threshold
    cumulative_probs = np.cumsum(sorted_probs)
    # Keep tokens until we exceed p
    # The +1 ensures we include the token that pushed us over the threshold
    nucleus_size = np.searchsorted(cumulative_probs, p) + 1

    nucleus_indices = sorted_indices[:nucleus_size]
    nucleus_probs = probs[nucleus_indices]

    # Renormalize over nucleus
    nucleus_probs = nucleus_probs / nucleus_probs.sum()

    chosen = np.random.choice(nucleus_indices, p=nucleus_probs)
    return int(chosen)

def visualize_sampling_strategies(
    vocab: list[str],
    logits: np.ndarray
) -> None:
    """
    Side-by-side visualization of how different parameters
    affect the sampling distribution.
    """
    temperatures = [0.2, 1.0, 1.5]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Effect of Temperature on Token Distribution", fontsize=14)

    for ax, temp in zip(axes, temperatures):
        scaled = apply_temperature(logits, temp)
        probs = softmax(scaled)

        colors = ['#2196F3' if p > 0.1 else '#BBDEFB' for p in probs]
        bars = ax.bar(vocab, probs, color=colors, edgecolor='white')
        ax.set_title(f"Temperature = {temp}", fontsize=12, fontweight='bold')
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', rotation=45)

        # Annotate values
        for bar, prob in zip(bars, probs):
            if prob > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2.,
                       bar.get_height() + 0.01,
                       f'{prob:.2f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig("sampling_strategies.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: sampling_strategies.png")

# ─── Demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate logits for "The best city in India for tech jobs is ___"
    vocab = ["Bengaluru", "Hyderabad", "Pune", "Mumbai", "Chennai", "Delhi"]
    logits = np.array([4.5, 3.2, 2.8, 2.1, 1.9, 1.7])  # Raw model scores

    print("=" * 60)
    print("SAMPLING STRATEGY DEMO")
    print("'The best city in India for tech jobs is ___'")
    print("=" * 60)

    probs = softmax(logits)
    print("\nBase probability distribution:")
    for word, p in sorted(zip(vocab, probs), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        print(f"  {word:<12} {bar:<40} {p:.3f}")

    # Run 20 samples from each strategy
    n_samples = 20
    np.random.seed(42)

    print(f"\n--- Greedy (always deterministic) ---")
    idx = greedy_sample(logits)
    print(f"Always picks: '{vocab[idx]}'")

    print(f"\n--- Top-K (k=3, temp=0.8) — {n_samples} samples ---")
    counts = {}
    for _ in range(n_samples):
        idx = top_k_sample(logits, k=3, temperature=0.8)
        counts[vocab[idx]] = counts.get(vocab[idx], 0) + 1
    for word, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {word}: {count}/{n_samples} times")

    print(f"\n--- Top-P (p=0.9, temp=0.8) — {n_samples} samples ---")
    counts = {}
    for _ in range(n_samples):
        idx = top_p_sample(logits, p=0.9, temperature=0.8)
        counts[vocab[idx]] = counts.get(vocab[idx], 0) + 1
    for word, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {word}: {count}/{n_samples} times")

    print(f"\n--- High Temperature (temp=1.8, p=0.95) — chaotic ---")
    counts = {}
    for _ in range(n_samples):
        idx = top_p_sample(logits, p=0.95, temperature=1.8)
        counts[vocab[idx]] = counts.get(vocab[idx], 0) + 1
    for word, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {word}: {count}/{n_samples} times")

    visualize_sampling_strategies(vocab, logits)
```

---

## 🌀 5. Hallucination — Why LLMs Lie Confidently

### What It Is

Hallucination = the model generates **fluent, confident, wrong information**.

```
User: "Who is the CEO of Paytm?"

Hallucinated response (2021 model):
"The CEO of Paytm is Renu Satti."
← Sounds confident. Completely wrong. (It's Vijay Shekhar Sharma)

Why? The model learned to generate fluent CEO-attribution sentences.
It filled in a plausible-sounding name, not the correct one.
```

### Root Causes

```
1. TRAINING OBJECTIVE MISMATCH
   Model is trained to predict next token (fluency)
   NOT trained to verify facts (accuracy)
   These are different skills

2. KNOWLEDGE CUTOFF
   Training data ends at a date
   World keeps changing
   Model confidently answers about things that changed

3. LOW-FREQUENCY FACTS
   "Who won the 1987 Ranji Trophy?" — rare in training data
   Model pattern-matches to similar sentences → wrong answer
   High-frequency facts (Virat Kohli plays cricket) → usually correct

4. CONFLICTING TRAINING DATA
   Internet has contradictions
   Model averages over contradictions → neither answer is right

5. PROMPT PRESSURE
   User asks leading question: "Einstein failed math, right?"
   Model agrees to seem helpful → reinforces wrong belief
```

### Hallucination Taxonomy

```
Type 1: Factual Hallucination
  "The GST rate on gold is 5%"  ← wrong (it's 3%)

Type 2: Citation Hallucination
  "According to a 2022 RBI circular..."  ← circular may not exist

Type 3: Reasoning Hallucination
  Gets the logic wrong step by step but sounds confident

Type 4: Instruction Hallucination
  "You asked me to write 500 words" ← user asked for 100 words

Type 5: Identity Hallucination
  "As an AI trained by Google..." ← model confuses its own identity
```

### Mitigation Strategies

```
Strategy 1: RAG (Week 2)
  Ground the model in retrieved documents
  "Answer ONLY based on the provided context"

Strategy 2: Low Temperature
  temperature=0.1 reduces creative "filling in"

Strategy 3: System Prompt Constraints
  "If you don't know, say 'I don't know'"
  "Never make up citations or statistics"

Strategy 4: Structured Output + Validation
  Force JSON output → validate fields programmatically

Strategy 5: Self-Consistency
  Generate 3 answers → pick the most common one
  (More expensive but more reliable)

Strategy 6: Fine-tuning (Week 6)
  Train model on domain-specific correct data
```

---

## 🖥️ 6. System Prompts — Shaping Model Behavior

A system prompt is the **constitution** of your chatbot. It runs before every conversation and sets the model's persona, constraints, and behavior.

```
┌─────────────────────────────────────────────────────┐
│  System Prompt (hidden from user, always first)     │
│  "You are X. You do Y. You never do Z."             │
├─────────────────────────────────────────────────────┤
│  User: "Hello"                                      │
├─────────────────────────────────────────────────────┤
│  Assistant: [responds according to system prompt]   │
├─────────────────────────────────────────────────────┤
│  User: "..."                                        │
└─────────────────────────────────────────────────────┘
```

### Anatomy of a Production System Prompt

```
1. IDENTITY    — Who are you? What's your name? Persona?
2. DOMAIN      — What domain/knowledge do you operate in?
3. TASK        — What specific job do you do?
4. CONSTRAINTS — What must you NEVER do?
5. FORMAT      — How should you structure responses?
6. TONE        — Formal? Friendly? Technical?
7. FALLBACK    — What to do when you don't know?
```

---


## ⚠️ 7. Common Mistakes — Day 3

**Mistake 1: Sending conversation history without a sliding window**

```python
# ❌ BAD — history grows forever, eventually hits context limit and crashes
history.append({"role": "user", "content": msg})
# After 50 messages → API error: "context length exceeded"

# ✅ GOOD — implement sliding window like TaxSaathi above
self._apply_sliding_window()
```

**Mistake 2: System prompt too vague**

```python
# ❌ BAD
system = "You are a helpful assistant."
# Model will answer ANYTHING — off-topic, hallucinate, no guardrails

# ✅ GOOD
system = """You are TaxSaathi...
NEVER discuss non-tax topics.
If uncertain, say exactly: 'I'm not certain...'"""
```

**Mistake 3: Ignoring finish_reason**

```python
response = client.chat.completions.create(...)

# ❌ BAD — blindly use the response
answer = response.choices[0].message.content

# ✅ GOOD — check WHY generation stopped
finish_reason = response.choices[0].finish_reason

if finish_reason == "length":
    # Response was cut off — max_tokens too low
    answer += " [Response truncated — please ask me to continue]"
elif finish_reason == "content_filter":
    answer = "This response was filtered. Please rephrase your question."
elif finish_reason == "stop":
    pass  # Normal — model finished naturally
```

**Mistake 4: Not handling API errors gracefully**

```python
# ❌ BAD — crashes your entire app
response = client.chat.completions.create(...)

# ✅ GOOD — catch specific errors
from openai import RateLimitError, APITimeoutError, AuthenticationError

try:
    response = client.chat.completions.create(...)
except RateLimitError:
    return "Too many requests. Please wait 30 seconds."
except APITimeoutError:
    return "Request timed out. Please try again."
except AuthenticationError:
    raise SystemExit("Invalid API key. Check your OPENAI_API_KEY.")
```

---

## 🎤 8. Interview Questions — Day 3

**Q1. What is temperature and how do you set it for a production system?**

> Temperature scales the logits before softmax, controlling distribution sharpness. For a production factual system (legal, medical, finance), I set temperature between 0.1–0.3 for consistent, accurate outputs. For creative applications (copywriting, brainstorming), 0.8–1.2. I never use temperature > 2.0 as outputs become incoherent. I always pair it with top_p=0.9 as a secondary safety net.
> 

**Q2. What is the difference between top-k and top-p sampling?**

> Top-k samples from exactly k tokens regardless of distribution shape — problematic when the model is very confident (k=50 includes many garbage tokens) or uncertain (k=3 misses valid options). Top-p (nucleus sampling) adapts automatically: it selects the smallest set of tokens whose cumulative probability reaches p. When the model is confident, the nucleus is small (2–3 tokens); when uncertain, it's larger. Top-p is the modern industry standard.
> 

**Q3. Why do LLMs hallucinate, and how do you reduce it in production?**

> LLMs are trained to predict the next token (fluency), not verify facts (accuracy). They hallucinate because: their training objective doesn't penalize confident wrongness, low-frequency facts are poorly learned, and they pattern-match to plausible-sounding outputs. In production, I reduce hallucination via: RAG (grounding in retrieved documents), low temperature, explicit system prompt constraints ("say I don't know if uncertain"), structured output validation, and self-consistency sampling for high-stakes decisions.
> 

**Q4. What is finish_reason and why should you always check it?**

> finish_reason tells you why the model stopped generating: "stop" means natural completion, "length" means max_tokens was hit (response truncated), "content_filter" means safety filtering triggered. In production, ignoring finish_reason means silently serving truncated or filtered responses to users. Always check it and handle each case — for "length," prompt the user to ask for continuation; for "content_filter," return a safe fallback message.
> 

**Q5. How does a system prompt affect model behavior?**

> The system prompt is processed first in every API call and sets the model's operating constraints — its persona, domain scope, tone, output format, and what it must refuse. It's implemented as a special "system" role message that the model is trained to follow with high priority. In production, well-engineered system prompts significantly reduce hallucination, off-topic responses, and unsafe outputs. They are a primary mechanism for making a general-purpose LLM behave like a specialized product.
> 

---

## ✍️ 9. Practice Exercises

### Easy — Parameter Explorer

Modify TaxSaathi to accept temperature as a CLI argument. Run the same tax question 5 times with temperatures 0.1, 0.5, 1.0, 1.5, 2.0. Print all 5 responses. Observe and write 3 differences you notice.

### Medium — Hallucination Detector

Ask TaxSaathi these 5 questions and evaluate responses for accuracy:

- "What is the GST rate on gold jewellery?"
- "What is the deadline for ITR filing for AY 2024-25?"
- "What is the Section 80C limit?"
- "Who is the current Finance Minister of India?"
- "What happened in the Union Budget 2024?"

For each, verify against `incometaxindia.gov.in`. Document where it hallucinated and what prompt changes reduced it.

### Challenge (Portfolio-worthy) 🏆

Extend TaxSaathi with:

- **Streaming responses** — print tokens as they arrive (use `stream=True`)
- **Multi-language support** — auto-detect if user writes Hindi, respond in Hindi
- **Conversation summary** — when history hits 70% of context limit, call the API once to summarize history, replace full history with summary (this is a real production pattern called "summarization memory")
- **Session analytics dashboard** — at the end, print a pretty table showing cost per question, average response length, longest question, etc.

---

## 📚 Resources for Today

- **OpenAI Cookbook** — `github.com/openai/openai-cookbook` — real production patterns
- **"How to prevent hallucinations"** — Anthropic and OpenAI both have blog posts on this
- **tiktoken playground** — `platform.openai.com/tokenizer`
- **Lilian Weng's blog** — "Extrinsic Hallucinations in LLMs" — excellent deep dive

---

## 🔗 Connection to the Big Picture

```
Day 1             Day 2                Day 3               Week 2
──────────        ──────────           ──────────          ──────────
Tokens &    →    Attention &    →    Generation &    →    RAG — fix
Embeddings       Transformers         Hallucination        hallucination
                                      Chatbot built        with retrieval
```

You now understand the **complete forward pass** of an LLM — from input text to output token. You've built a real chatbot. You understand why it sometimes lies and how to constrain it.