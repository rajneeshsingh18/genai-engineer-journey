# LLMs, Tokens, Embeddings & Context Windows 🧠

---

## 🎯 1. What is an LLM?

### The Simple Answer

An LLM (Large Language Model) is a program trained on massive amounts of text that learned one deceptively simple skill:

> **"Given some text, predict what word comes next."**

That's it. But doing this billions of times across trillions of words forces the model to develop a deep internal understanding of language, facts, reasoning, and even code.

### The Analogy

Think of an LLM like an **extraordinarily well-read person** who has read the entire internet, every book, every research paper — and can continue any sentence you give them in a coherent, contextually aware way.

But unlike a search engine (which retrieves existing text), an LLM **generates new text** by predicting the most likely next token, one at a time.

### What Makes it "Large"?

Three things scale together:

| Dimension | Example |
|---|---|
| Parameters (weights) | GPT-4 ~1.7 trillion, Llama 3 8B = 8 billion |
| Training data | Trillions of tokens from the web, books, code |
| Compute | Thousands of GPUs running for months |

Parameters are the model's "memory" — numbers that get tuned during training to encode language patterns.

---

## 🔤 2. What are Tokens?

### The Problem First

Computers can't read text. They only understand numbers. So the first job is: **how do we convert text → numbers?**

The naive answer is "one number per character." But that's inefficient and loses meaning.

The smarter answer: **Tokens.**

### What is a Token?

A token is a **chunk of text** — not exactly a word, not exactly a character. It's somewhere in between, optimized for efficiency.

```
"Hello, how are you?" → ["Hello", ",", " how", " are", " you", "?"]
                          6 tokens
```

```
"Unbelievable" → ["Un", "believ", "able"]
                   3 tokens
```

```
"नमस्ते" (Namaste in Hindi) → ["न", "म", "स", "त", "े"]
                               5+ tokens (non-English = more tokens)
```

### How Tokenization Works (BPE — Byte Pair Encoding)

This is the most common algorithm (used by GPT, Llama, Claude):

1. Start with every character as its own token
2. Find the most frequent pair of adjacent tokens
3. Merge them into one new token
4. Repeat until vocabulary size is reached (usually 32K–128K tokens)

Result: common words become single tokens, rare words get split.

```python
# Let's see real tokenization with tiktoken (OpenAI's tokenizer)
import tiktoken

encoder = tiktoken.get_encoding("cl100k_base")  # GPT-4's tokenizer

texts = [
    "Hello, how are you?",
    "Unbelievable",
    "I love eating biryani in Hyderabad",
    "मुझे बिरयानी बहुत पसंद है",  # Hindi
    "def calculate_gst(price, rate=0.18):",
]

for text in texts:
    tokens = encoder.encode(text)
    print(f"Text: {text}")
    print(f"Token IDs: {tokens}")
    print(f"Token count: {len(tokens)}")
    print(f"Tokens: {[encoder.decode([t]) for t in tokens]}")
    print("---")
```

### Why This Matters Practically

- **Cost:** OpenAI charges per token. "Write me a poem" = ~6 tokens. A 10-page document = ~2500 tokens.
- **Hindi/regional languages** cost 3–5x more tokens than English for the same meaning — critical for Indian AI products.
- **Code** is tokenized efficiently since it appears heavily in training data.

**Rule of thumb:** 1 token ≈ 0.75 English words, or 4 characters.

---

## 🔢 3. What are Embeddings?

### The Problem

Token IDs are just arbitrary numbers. Token 4839 ("king") has no mathematical relationship to token 4721 ("queen") in raw ID form. We need a representation where **meaning is encoded in the numbers**.

### What is an Embedding?

An embedding is a **list of floating point numbers (a vector)** that represents the meaning of a token/word/sentence in a high-dimensional space.

```
"king"  → [0.2, -0.4, 0.8, 0.1, ..., 0.3]  # 1536 numbers
"queen" → [0.2, -0.3, 0.7, 0.2, ..., 0.4]  # 1536 numbers
"apple" → [-0.8, 0.9, -0.1, 0.6, ..., -0.2] # 1536 numbers
```

Words with similar meanings → vectors that are close together in space.

### The Famous Analogy

```
king - man + woman ≈ queen
```

This works because the "royalty" direction and "gender" direction are encoded as actual geometric directions in the vector space. This is not magic — it emerges from training on enough data.

### Visualizing It (Simplified to 2D)

```
         ^
  queen  |  king
         |
---------+-------->
  girl   |  boy
         |
```

In reality, it's 768, 1536, or 3072 dimensions — but the same geometric principle holds.

### Types of Embeddings You'll Use

| Type | What it embeds | Common Models | Used For |
|---|---|---|---|
| Word embeddings | Single words | Word2Vec, GloVe | Old-school NLP |
| Token embeddings | Subword tokens | Inside every LLM | LLM internals |
| Sentence embeddings | Full sentences | `text-embedding-3-small`, `bge-large` | RAG, Search |
| Document embeddings | Paragraphs/docs | Same models | RAG pipelines |

### Code: Generating Real Embeddings

```python
# pip install openai numpy
import os
import numpy as np
from openai import OpenAI
from typing import List

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    Generate embedding vector for a given text.
    
    Args:
        text: Input string to embed
        model: OpenAI embedding model to use
    
    Returns:
        List of floats representing the embedding vector
    """
    # Always clean the text — newlines hurt embedding quality
    text = text.replace("\n", " ").strip()
    
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Measure similarity between two vectors.
    Returns value between -1 (opposite) and 1 (identical).
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- Demo: Semantic similarity ---
sentences = [
    "How do I pay my electricity bill online?",   # Query
    "Steps to pay BESCOM bill using UPI",          # Very relevant
    "How to transfer money via NEFT?",             # Somewhat relevant  
    "Best restaurants in Bengaluru for biryani",   # Not relevant
]

query_embedding = get_embedding(sentences[0])

print(f"Query: {sentences[0]}\n")
for sentence in sentences[1:]:
    emb = get_embedding(sentence)
    score = cosine_similarity(query_embedding, emb)
    print(f"Score: {score:.4f} | Text: {sentence}")
```

**Expected output:**
```
Score: 0.8923 | Text: Steps to pay BESCOM bill using UPI       ← Most similar
Score: 0.6234 | Text: How to transfer money via NEFT?
Score: 0.2341 | Text: Best restaurants in Bengaluru for biryani ← Least similar
```

This is the **core of RAG** — you'll build an entire retrieval system with this in Week 2.

---

## 🪟 4. What is a Context Window?

### The Concept

Every LLM has a **context window** — the maximum number of tokens it can "see" at once when generating a response. Think of it as the model's **working memory**.

```
[System Prompt] + [Chat History] + [Your Message] + [Documents] < Context Limit
```

| Model | Context Window |
|---|---|
| GPT-3.5 | 16K tokens |
| GPT-4o | 128K tokens |
| Claude 3.5 Sonnet | 200K tokens |
| Gemini 1.5 Pro | 1M tokens |
| Llama 3 8B | 8K tokens |

### Why It Matters

```
128K tokens ≈ 300 pages of text ≈ an entire novel
```

But bigger isn't always better:

- **Cost** scales linearly with tokens (sending 100K tokens every call = expensive)
- **Latency** increases with context size
- **Lost in the middle problem** — LLMs struggle to use information from the middle of very long contexts (they remember the start and end better)

### The Context Window in Action

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Everything inside messages[] must fit within the context window
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        # 1. System prompt — sets behavior (doesn't change per call)
        {
            "role": "system",
            "content": "You are a helpful assistant for Indian tax queries. "
                      "Always mention relevant sections of the Income Tax Act."
        },
        # 2. Conversation history — what was said before
        {
            "role": "user",
            "content": "What is Section 80C?"
        },
        {
            "role": "assistant", 
            "content": "Section 80C allows deductions up to ₹1.5 lakh..."
        },
        # 3. New user message
        {
            "role": "user",
            "content": "Can I claim both 80C and 80D in the same year?"
        }
    ],
    temperature=0.3,   # Lower = more factual, less creative
    max_tokens=500     # Limit response length
)

print(response.choices[0].message.content)

# Check how many tokens you used
print(f"\nTokens used: {response.usage.total_tokens}")
print(f"Prompt tokens: {response.usage.prompt_tokens}")
print(f"Completion tokens: {response.usage.completion_tokens}")
```

### How LLMs Use the Context Window Internally

```
Input Tokens
     │
     ▼
┌─────────────────────────────────┐
│  Tokenizer (text → token IDs)   │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Embedding Layer                │
│  (token IDs → vectors)          │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Transformer Blocks (32–96x)    │
│  - Self-Attention               │  ← Every token looks at every
│  - Feed Forward Network         │    other token in the context
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│  Output Head                    │
│  (vector → probability over     │
│   vocabulary → next token)      │
└─────────────────────────────────┘
```

The **self-attention mechanism** is what makes the context window work — it lets every token attend to (draw information from) every other token in the window. This is the magic of Transformers. (We'll go deep on this Day 2.)

---

## 🌍 5. Real-World Indian Context Example

Here's a complete, runnable mini-project tying all three concepts together:

```python
"""
Indian Tax FAQ Bot — Week 1 Portfolio Starter
Demonstrates: Tokens, Embeddings, Context Window in one script
"""

import os
import tiktoken
from openai import OpenAI
from typing import Generator

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
encoder = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens before sending to API — avoid surprise bills."""
    return len(encoder.encode(text))


def estimate_cost(prompt_tokens: int, completion_tokens: int,
                  model: str = "gpt-4o-mini") -> float:
    """
    Rough cost estimate in USD.
    gpt-4o-mini: $0.15/1M input, $0.60/1M output
    """
    costs = {
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4o":      {"input": 2.50 / 1_000_000, "output": 10.0 / 1_000_000},
    }
    c = costs.get(model, costs["gpt-4o-mini"])
    return (prompt_tokens * c["input"]) + (completion_tokens * c["output"])


def chat_with_tax_bot(user_question: str, conversation_history: list) -> str:
    """
    Simple tax chatbot demonstrating context window management.
    
    Args:
        user_question: The user's current question
        conversation_history: List of previous messages (maintained externally)
    
    Returns:
        Assistant's response as string
    """
    system_prompt = """You are TaxSaathi, an expert Indian tax assistant.
    You help users understand Income Tax Act provisions, ITR filing, 
    GST, and tax-saving instruments.
    
    Guidelines:
    - Always cite relevant IT Act sections (e.g., Section 80C, 44AD)
    - Use INR (₹) for amounts
    - Keep answers concise but accurate
    - Mention if professional CA advice is needed
    """
    
    # Build the full message list
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": user_question}
    ]
    
    # Count tokens BEFORE sending (good production habit)
    full_prompt = system_prompt + str(conversation_history) + user_question
    estimated_tokens = count_tokens(full_prompt)
    print(f"📊 Estimated input tokens: {estimated_tokens}")
    
    # Safety check — warn if approaching context limit
    if estimated_tokens > 100_000:
        print("⚠️  Warning: Approaching context limit. Consider summarising history.")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,  # Low temperature for factual tax info
        max_tokens=600
    )
    
    answer = response.choices[0].message.content
    actual_cost = estimate_cost(
        response.usage.prompt_tokens,
        response.usage.completion_tokens
    )
    
    print(f"💰 Actual cost this call: ${actual_cost:.6f} USD")
    print(f"🔢 Total tokens used: {response.usage.total_tokens}")
    
    return answer


# --- Run a demo conversation ---
if __name__ == "__main__":
    history = []
    
    questions = [
        "What is the difference between old and new tax regime?",
        "Under old regime, what deductions can a salaried person claim?",
        "If my salary is ₹12 LPA, which regime saves more tax?"
    ]
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"👤 User: {question}")
        print(f"{'='*60}")
        
        answer = chat_with_tax_bot(question, history)
        print(f"\n🤖 TaxSaathi: {answer}")
        
        # Maintain conversation history (this IS the context window in practice)
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})
```

---

## ⚠️ 6. Common Mistakes — Day 1

**Mistake 1: Ignoring token costs in production**
```python
# ❌ BAD — sending entire document every call
response = client.chat.completions.create(
    messages=[{"role": "user", "content": entire_10000_word_doc + question}]
)
# This can cost ₹50+ per call at scale

# ✅ GOOD — count first, truncate if needed
if count_tokens(doc) > 3000:
    doc = doc[:12000]  # rough character limit
```

**Mistake 2: Confusing embeddings with token IDs**
- Token IDs = arbitrary integers (just lookup table indices)
- Embeddings = meaningful float vectors (geometric relationships)
- They are NOT the same thing

**Mistake 3: Thinking context window = model memory**
```
# The model has NO memory between API calls
# Each call is completely stateless
# You must manually pass history in messages[]
```

**Mistake 4: Using high temperature for factual tasks**
```python
# ❌ BAD for a tax/legal/medical bot
temperature=1.5  # Very creative → hallucinates facts

# ✅ GOOD for factual tasks
temperature=0.1  # Deterministic → consistent, accurate
```

---

## 🎤 7. Top 10 Interview Questions (With Answers)

These are asked at companies like Swiggy, Razorpay, Sarvam AI, Krutrim, and product-based MNCs hiring GenAI engineers in India.

---

**Q1. What is a token and how does tokenization work?**

> A token is a subword unit — the basic chunk of text that LLMs process. Tokenization converts raw text into a sequence of integer IDs using algorithms like BPE (Byte Pair Encoding). BPE starts with character-level tokens and iteratively merges the most frequent adjacent pairs until a target vocabulary size is reached. Common words become single tokens; rare words get split into subword pieces.

---

**Q2. Why does the same text in Hindi cost more tokens than in English?**

> LLMs are trained predominantly on English data. Their tokenizers (built with BPE on English-heavy corpora) have a large vocabulary of English subwords but limited Hindi subwords. So Hindi text gets split into more, smaller pieces — often single Unicode characters — resulting in 3–5x more tokens for the same semantic content. This directly increases API costs and context window consumption.

---

**Q3. What is an embedding and why is cosine similarity used to compare them?**

> An embedding is a dense vector representation of text that encodes semantic meaning. Semantically similar texts map to geometrically close vectors in high-dimensional space. Cosine similarity measures the angle between two vectors (not their magnitude), making it robust to texts of different lengths. It returns a value from -1 to 1, where 1 means identical direction (identical meaning).

---

**Q4. What is the "lost in the middle" problem?**

> Research shows LLMs are better at using information from the beginning and end of their context window than information in the middle. If you stuff a 100K-token document into the context, facts in the middle sections are often ignored or poorly utilized. This is why RAG (retrieving only relevant chunks) often outperforms naive "just dump everything in context" approaches.

---

**Q5. What is the difference between `temperature` and `top_p`?**

> Both control output randomness. `temperature` scales the probability distribution over the vocabulary — higher values flatten it (more random), lower values sharpen it (more deterministic). `top_p` (nucleus sampling) restricts sampling to the smallest set of tokens whose cumulative probability exceeds `p`. In practice, adjust one and leave the other at default. For production factual tasks, use temperature 0.1–0.3.

---

**Q6. How does the context window affect RAG architecture?**

> In RAG, you retrieve relevant document chunks and inject them into the context window along with the user query. The context window size limits how many chunks you can inject. With a 128K context, you can inject ~100 medium-sized chunks — but cost and latency scale accordingly. Smart RAG systems use reranking to pick only the top 3–5 most relevant chunks before injecting, keeping context lean and cost low.

---

**Q7. What is the difference between an LLM and a traditional ML model?**

> Traditional ML models are task-specific (a fraud detection model only detects fraud). LLMs are general-purpose — the same model can translate, summarize, reason, and code. LLMs are also autoregressive (generate one token at a time using previous outputs as input) and are based on the Transformer architecture. Traditional ML models use various architectures (decision trees, CNNs, etc.) and don't generate text.

---

**Q8. What are the parameters of an LLM and what do they store?**

> Parameters are the numerical weights of the neural network — floating point numbers stored in matrices. During training, they're adjusted (via backpropagation) to minimize prediction error. They store learned patterns: syntactic rules, factual associations, reasoning strategies, coding patterns — all encoded implicitly as numbers. An 8B parameter model has ~8 billion such numbers, taking ~16GB in float16.

---

**Q9. What is hallucination and why does it happen?**

> Hallucination is when an LLM confidently generates factually incorrect information. It happens because LLMs are trained to predict the next most likely token — not to retrieve verified facts. The model learns statistical patterns, not a ground-truth knowledge base. When the model encounters a query about something outside its training distribution (or something it saw rarely), it "fills in" plausible-sounding but incorrect tokens. RAG and grounding are the main mitigation strategies.

---

**Q10. What is the difference between `prompt_tokens` and `completion_tokens` in the API response?**

> `prompt_tokens` = the number of tokens in your input (system prompt + history + user message). `completion_tokens` = tokens generated by the model in the response. Total cost = (prompt_tokens × input_price) + (completion_tokens × output_price). Completion tokens are usually priced higher. For cost optimization, minimize prompt tokens (shorter system prompts, fewer history turns) and limit completion tokens with `max_tokens`.

---


---

## 📚 Resources for Day 1

- **Andrej Karpathy** — "Let's build the GPT tokenizer" (YouTube) — watch the first 30 mins
- **tiktoken** docs — `github.com/openai/tiktoken`
- **OpenAI Tokenizer** — `platform.openai.com/tokenizer` — paste any text and see live tokenization
- **Paper:** "Lost in the Middle" (2023) — Google Scholar — just read the abstract and conclusion

---

## 🔗 Connection to the Big Picture

```
Today (Day 1)          Week 2 (RAG)           Week 3 (LangChain)
─────────────         ──────────────         ──────────────────
Tokens → IDs    →     Embed chunks     →     Chain: Retrieve +
Embeddings      →     FAISS index      →     Prompt + Generate
Context Window  →     Inject chunks    →     Memory management
```

Everything you learned today is the atomic unit of everything else. When you build your RAG pipeline next week, you'll use embeddings for retrieval. When you build agents in Week 4, context window management becomes critical. When you fine-tune in Week 6, tokens are literally what you train on.

---

