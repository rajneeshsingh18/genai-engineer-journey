# Week 1, Day 2 — Transformer Architecture, Attention & Q/K/V 🧠

Yesterday you learned what LLMs *process* (tokens) and how meaning is *represented* (embeddings). Today you learn **how LLMs actually think** — the engine underneath every model you'll ever use.

This is the most important architectural concept in all of modern AI. Let's make it stick.

---

## 🎯 1. The Problem Transformers Solve

Before Transformers (pre-2017), the best sequence models were **RNNs and LSTMs**. They had a fatal flaw:

```
"The cat, which had been sitting on the mat near the window 
 in the corner of the old dusty room, was hungry."
```

An RNN reads this **left to right, one word at a time**, compressing everything into a single hidden state vector. By the time it reaches "was hungry," it has nearly forgotten "cat" — the subject.

This is called the **vanishing gradient problem** — information from early tokens gets diluted over long sequences.

### The Transformer's Answer

> **"Don't process sequentially. Let every token look at every other token simultaneously."**

This one idea changed everything.

```
RNN:   cat → ... → ... → ... → was hungry  (forgets early context)
                                
Transformer: cat ←────────────→ was hungry  (direct connection always)
             cat ←──────────→ hungry
             cat ←────────→ old
             (every token attends to every other token in parallel)
```

---

## 🔬 2. The Transformer Architecture — Full Picture

Here's the complete flow of text through a Transformer:

```
Input Text: "What is GST rate for gold?"
                │
                ▼
    ┌───────────────────────┐
    │   TOKENIZER           │  "What", "is", "GST", "rate", "for", "gold", "?"
    └───────────────────────┘         → [2061, 374, 9528, 4478, 369, 7267, 30]
                │
                ▼
    ┌───────────────────────┐
    │   TOKEN EMBEDDINGS    │  Each token ID → dense vector (e.g., 4096 dims)
    └───────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │  POSITIONAL ENCODING  │  "WHERE is this token in the sequence?"
    └───────────────────────┘  Added to embedding so order is preserved
                │
                ▼
    ┌───────────────────────────────────────────┐
    │         TRANSFORMER BLOCK × N             │  (32 blocks in Llama 3 8B)
    │                                           │
    │   ┌───────────────────────────────────┐   │
    │   │     MULTI-HEAD SELF-ATTENTION     │   │  ← THE CORE MECHANISM
    │   └───────────────────────────────────┘   │
    │                   │                       │
    │   ┌───────────────────────────────────┐   │
    │   │       FEED FORWARD NETWORK        │   │  ← Processes each token
    │   └───────────────────────────────────┘   │
    │                                           │
    └───────────────────────────────────────────┘
                │
                ▼
    ┌───────────────────────┐
    │    OUTPUT HEAD (LM)   │  Vector → probabilities over vocabulary
    └───────────────────────┘
                │
                ▼
         Next Token Prediction
         "gold" → 3% → next token could be "is", "has", "in"...
```

Now let's zoom into the most important part — **Self-Attention**.

---

## 🔑 3. Self-Attention — The Core Idea

### The Analogy First

Imagine you're at a conference and you hear: **"The RBI governor said inflation will fall."**

When you process the word **"inflation"**, your brain automatically asks:
- What does "inflation" relate to in this sentence?
- "RBI governor" — very relevant (they're talking about it)
- "said" — somewhat relevant (how we know)
- "fall" — very relevant (what's happening to it)
- "The" — not relevant at all

Self-attention does exactly this — for every token, it computes a **relevance score** against every other token, then **uses those scores to create a new, context-enriched representation** of that token.

```
Before attention:  "bank" = generic vector for the word "bank"
                   
After attention:   "I went to the river bank"
                   "bank" = weighted mix of river(0.7) + went(0.2) + I(0.1)
                   → now the vector means "river bank", not "financial bank"
```

**This is how LLMs understand context and disambiguate meaning.**

---

## 🔑 4. Q, K, V — The Heart of Attention

### The Library Analogy

Think of a **library search system**:

- **Q (Query)** — Your search query: *"books about machine learning"*
- **K (Key)** — The index card labels on each book: *"Python", "Deep Learning", "Cooking"*
- **V (Value)** — The actual book content

You compare your **Query** against all **Keys** to get relevance scores, then retrieve a weighted mix of the **Values** based on those scores.

In self-attention, every token simultaneously acts as Q, K, AND V.

### How Q, K, V are Created

Each token's embedding gets multiplied by three separate learned weight matrices:

```
Token embedding: x = [0.2, 0.5, -0.3, 0.8, ...]  (e.g., 4096 dims)

Q = x × W_Q    (learned matrix, maps to query space, e.g., 128 dims)
K = x × W_K    (learned matrix, maps to key space,   e.g., 128 dims)  
V = x × W_V    (learned matrix, maps to value space, e.g., 128 dims)
```

These matrices **W_Q, W_K, W_V** are learned during training. They learn to project tokens into a space where semantically related tokens have similar Q-K dot products.

---

## 🔢 5. The Attention Calculation — Step by Step

Let's use a concrete mini-example:

```
Sentence: "Virat scored a century"
Tokens:   [Virat] [scored] [a] [century]
```

### Step 1: Create Q, K, V for each token

```python
# Simplified — real dims are much larger
Virat.Q   = [0.9, 0.1]    Virat.K   = [0.8, 0.2]    Virat.V   = [1.0, 0.0]
scored.Q  = [0.7, 0.3]    scored.K  = [0.6, 0.4]    scored.V  = [0.5, 0.5]
a.Q       = [0.1, 0.9]    a.K       = [0.1, 0.9]    a.V       = [0.0, 1.0]
century.Q = [0.8, 0.2]    century.K = [0.7, 0.3]    century.V = [0.9, 0.1]
```

### Step 2: Compute attention scores (for token "century")

Century's Query asks: *"Who is relevant to me?"*

```
Score(century → Virat)   = century.Q · Virat.K   = (0.8×0.8) + (0.2×0.2) = 0.68
Score(century → scored)  = century.Q · scored.K  = (0.8×0.6) + (0.2×0.4) = 0.56
Score(century → a)       = century.Q · a.K       = (0.8×0.1) + (0.2×0.9) = 0.26
Score(century → century) = century.Q · century.K = (0.8×0.7) + (0.2×0.3) = 0.62
```

Higher score = more relevant.

### Step 3: Scale and Softmax

```
Scale: divide by √(key_dimension) = √2 ≈ 1.41
→ Scaled scores: [0.48, 0.40, 0.18, 0.44]

Softmax → probabilities (sum to 1.0):
→ Attention weights: [0.34, 0.29, 0.17, 0.31]
  (Virat=0.34, scored=0.29, a=0.17, century=0.31)
```

"Century" pays most attention to "Virat" and itself — makes sense! The person who scored the century is the most relevant context.

### Step 4: Weighted sum of Values

```
New century vector = 0.34 × Virat.V 
                   + 0.29 × scored.V
                   + 0.17 × a.V
                   + 0.31 × century.V

= 0.34×[1.0,0.0] + 0.29×[0.5,0.5] + 0.17×[0.0,1.0] + 0.31×[0.9,0.1]
= [0.34,0.0] + [0.145,0.145] + [0.0,0.17] + [0.279,0.031]
= [0.764, 0.346]
```

The new "century" vector now **contains information about Virat**, because Virat had the highest attention weight. This is context-enriched representation.

### The Full Formula

```
                    ┌  Q × Kᵀ  ┐
Attention(Q,K,V) = softmax│ ──────── │ × V
                    └   √d_k   ┘
```

- `Q × Kᵀ` — dot product of all queries against all keys (similarity matrix)
- `√d_k` — scaling factor (prevents dot products from getting too large)
- `softmax` — converts raw scores to probabilities
- `× V` — weighted sum of values

---

## 🎭 6. Multi-Head Attention — Why Multiple Heads?

Single attention head learns one type of relationship. But language has many:

```
"The bat flew out of the cave at night"
 ├── Syntactic: "bat" is subject of "flew" 
 ├── Semantic: "bat" = animal (not cricket bat) because of "cave" and "flew"
 └── Coreference: "The bat" refers to one specific bat
```

Multi-head attention runs **H independent attention heads in parallel**, each with its own W_Q, W_K, W_V:

```
                    Input Token Embedding
                           │
          ┌────────────────┼────────────────┐
          │                │                │
     Head 1            Head 2           Head 8
   (syntactic)        (semantic)      (positional)
    Q₁,K₁,V₁         Q₂,K₂,V₂       Q₈,K₈,V₈
          │                │                │
     Attention₁       Attention₂      Attention₈
          │                │                │
          └────────────────┼────────────────┘
                    Concatenate all heads
                           │
                     Linear projection
                           │
                  Final enriched vector
```

In GPT-4, there are ~128 attention heads per layer. Each specializes in capturing different linguistic patterns automatically — no human labeling required.

---

## 💻 7. Code — Attention from Scratch

Let's implement scaled dot-product attention in pure NumPy so you understand every line:

```python
"""
Self-Attention Implementation from Scratch
Week 1, Day 2 — Understanding the core mechanism
"""

import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray, 
    V: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention mechanism.
    
    This is the mathematical heart of every Transformer model.
    
    Args:
        Q: Query matrix  shape (seq_len, d_k)
        K: Key matrix    shape (seq_len, d_k)
        V: Value matrix  shape (seq_len, d_v)
        mask: Optional causal mask — used in decoder to prevent
              attending to future tokens. shape (seq_len, seq_len)
    
    Returns:
        output: Context-enriched representations (seq_len, d_v)
        attention_weights: Attention probability matrix (seq_len, seq_len)
    
    The formula: softmax(Q @ K.T / sqrt(d_k)) @ V
    """
    d_k = Q.shape[-1]  # Key dimension — used for scaling
    
    # Step 1: Compute raw attention scores
    # Q @ K.T gives similarity between every query-key pair
    # Shape: (seq_len, seq_len)
    scores = Q @ K.T  # Matrix multiplication
    
    # Step 2: Scale — prevents vanishing gradients in softmax
    # Without this, dot products grow large with d_k, 
    # pushing softmax into saturation (near-zero gradients)
    scores = scores / np.sqrt(d_k)
    
    # Step 3: Apply mask (for causal/decoder attention)
    # Future tokens get -infinity → softmax makes them 0
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    
    # Step 4: Softmax over the key dimension
    # Each row becomes a probability distribution
    # (how much does this token attend to each other token)
    def softmax(x: np.ndarray) -> np.ndarray:
        # Subtract max for numerical stability (prevents overflow)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    attention_weights = softmax(scores)
    
    # Step 5: Weighted sum of Values
    # Each token's new representation is a mix of all values,
    # weighted by how much it attends to each token
    output = attention_weights @ V
    
    return output, attention_weights


def visualize_attention(
    attention_weights: np.ndarray,
    tokens: list[str],
    title: str = "Self-Attention Weights"
) -> None:
    """
    Visualize attention as a heatmap.
    Rows = which token is attending (query)
    Cols = which token is being attended to (key)
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1
    )
    plt.title(title)
    plt.xlabel("Keys (tokens being attended to)")
    plt.ylabel("Queries (tokens attending)")
    plt.tight_layout()
    plt.savefig("attention_heatmap.png", dpi=150)
    plt.show()
    print("Saved: attention_heatmap.png")


class SingleHeadAttention:
    """
    A complete single-head self-attention layer with learned weights.
    
    In real transformers, these weights are trained via backpropagation.
    Here we initialize randomly to demonstrate the mechanism.
    """
    
    def __init__(self, d_model: int, d_k: int):
        """
        Args:
            d_model: Input embedding dimension
            d_k: Query/Key/Value dimension (usually d_model // num_heads)
        """
        self.d_k = d_k
        
        # Learnable weight matrices — initialized randomly
        # In training, these are optimized to minimize loss
        np.random.seed(42)  # For reproducibility
        self.W_Q = np.random.randn(d_model, d_k) * 0.1
        self.W_K = np.random.randn(d_model, d_k) * 0.1
        self.W_V = np.random.randn(d_model, d_k) * 0.1
    
    def forward(
        self, 
        x: np.ndarray,
        causal: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through self-attention.
        
        Args:
            x: Token embeddings, shape (seq_len, d_model)
            causal: If True, apply causal mask (for decoder/generation)
        
        Returns:
            output: Enriched representations (seq_len, d_k)
            weights: Attention weight matrix (seq_len, seq_len)
        """
        seq_len = x.shape[0]
        
        # Project embeddings into Q, K, V spaces
        Q = x @ self.W_Q  # (seq_len, d_k)
        K = x @ self.W_K  # (seq_len, d_k)
        V = x @ self.W_V  # (seq_len, d_k)
        
        # Causal mask: lower triangular matrix
        # Token i can only attend to tokens 0..i (not future tokens)
        # This is what makes GPT a "causal" language model
        mask = None
        if causal:
            mask = np.tril(np.ones((seq_len, seq_len)))
        
        output, weights = scaled_dot_product_attention(Q, K, V, mask)
        return output, weights


# ============================================================
# DEMO: Run attention on an Indian cricket sentence
# ============================================================

if __name__ == "__main__":
    
    # --- Setup ---
    tokens = ["Rohit", "smashed", "a", "six", "at", "Wankhede"]
    seq_len = len(tokens)
    d_model = 8   # Tiny embedding dim for demo (real = 4096)
    d_k = 4       # Key/Query dimension
    
    # Simulate token embeddings (in reality these come from the embedding layer)
    np.random.seed(0)
    token_embeddings = np.random.randn(seq_len, d_model)
    
    print("=" * 60)
    print("SELF-ATTENTION DEMO")
    print("Sentence: 'Rohit smashed a six at Wankhede'")
    print("=" * 60)
    print(f"\nToken embeddings shape: {token_embeddings.shape}")
    print(f"(seq_len={seq_len}, d_model={d_model})\n")
    
    # --- Single Head Attention ---
    attention = SingleHeadAttention(d_model=d_model, d_k=d_k)
    
    # Non-causal (encoder-style — every token sees every other)
    output, weights = attention.forward(token_embeddings, causal=False)
    
    print("Attention weights matrix (row i attends to col j):")
    print(f"Shape: {weights.shape}\n")
    for i, token in enumerate(tokens):
        top_attended = tokens[np.argmax(weights[i])]
        print(f"'{token}' attends most to: '{top_attended}' "
              f"(weight={np.max(weights[i]):.3f})")
    
    # Visualize
    visualize_attention(weights, tokens, 
                       "Self-Attention: 'Rohit smashed a six at Wankhede'")
    
    # --- Causal Attention Demo ---
    print("\n" + "=" * 60)
    print("CAUSAL ATTENTION (Decoder / GPT-style)")
    print("Future tokens are masked — model can't cheat!")
    print("=" * 60)
    
    _, causal_weights = attention.forward(token_embeddings, causal=True)
    
    print("\nCausal attention weights (lower triangle only):")
    for i, token in enumerate(tokens):
        visible = [f"{tokens[j]}({causal_weights[i][j]:.2f})" 
                  for j in range(i+1)]
        print(f"'{token}' sees: {' | '.join(visible)}")
    
    visualize_attention(causal_weights, tokens, 
                       "Causal Attention (GPT-style) — Future tokens masked")
    
    print("\n✅ Understanding:")
    print("- Encoder attention: all tokens ↔ all tokens (BERT, embedding models)")
    print("- Decoder attention: each token → only past tokens (GPT, Llama, Claude)")
```

---

## 🌍 8. Positional Encoding — Why Order Matters

Pure attention has no sense of order. "Rohit hit Kohli" and "Kohli hit Rohit" produce the same attention if word embeddings are the same.

Positional encoding injects **position information** into embeddings:

```python
import numpy as np
import matplotlib.pyplot as plt


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal positional encoding from "Attention is All You Need" (2017).
    
    Uses sine for even dimensions, cosine for odd dimensions.
    Each position gets a unique pattern that the model can learn to read.
    
    Why sinusoidal?
    - Unique encoding for each position
    - Relative positions can be expressed as linear functions
    - Generalizes to sequences longer than seen in training
    
    Args:
        seq_len: Length of the sequence
        d_model: Embedding dimension
    
    Returns:
        PE matrix shape (seq_len, d_model)
    """
    PE = np.zeros((seq_len, d_model))
    
    positions = np.arange(seq_len).reshape(-1, 1)       # (seq_len, 1)
    dimensions = np.arange(0, d_model, 2)               # Even indices
    
    # Different frequencies for different dimensions
    # Low dimensions = high frequency (captures local patterns)
    # High dimensions = low frequency (captures global position)
    div_term = np.power(10000, dimensions / d_model)
    
    PE[:, 0::2] = np.sin(positions / div_term)  # Even dims → sine
    PE[:, 1::2] = np.cos(positions / div_term)  # Odd dims  → cosine
    
    return PE


# Visualize positional encodings
pe = positional_encoding(seq_len=50, d_model=64)

plt.figure(figsize=(12, 4))
plt.pcolormesh(pe, cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.ylabel('Token Position in Sequence')
plt.title('Positional Encodings — Each row is a unique position signature')
plt.colorbar()
plt.tight_layout()
plt.savefig("positional_encoding.png", dpi=150)
plt.show()

print("The final input to the Transformer:")
print("final_input = token_embedding + positional_encoding")
print("Shape: (seq_len, d_model) — carries BOTH meaning AND position")
```

Modern models (Llama, GPT-4) use **RoPE (Rotary Position Embedding)** instead — it's more efficient and supports longer contexts. But the concept is the same: inject position into the vector.

---

## 🏗️ 9. Feed-Forward Network — What Happens After Attention

After attention, each token passes through a **Feed-Forward Network (FFN)** independently:

```
token_vector → Linear(d_model → 4×d_model) → GELU → Linear(4×d_model → d_model)
```

```python
import numpy as np

def feed_forward(x: np.ndarray, d_model: int, d_ff: int) -> np.ndarray:
    """
    Position-wise Feed-Forward Network.
    
    Applied identically and independently to each token.
    This is where the model stores 'facts' and 'knowledge'.
    
    Attention = routing (which information to mix)
    FFN       = processing (what to do with that information)
    
    Args:
        x: Token representations (seq_len, d_model)
        d_model: Model dimension
        d_ff: Feed-forward inner dimension (usually 4 × d_model)
    
    Returns:
        Processed representations (seq_len, d_model)
    """
    # Learnable weight matrices
    W1 = np.random.randn(d_model, d_ff) * 0.1  # Expand
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.1  # Contract
    b2 = np.zeros(d_model)
    
    # GELU activation — smoother than ReLU, used in modern LLMs
    def gelu(x: np.ndarray) -> np.ndarray:
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
    
    hidden = gelu(x @ W1 + b1)   # (seq_len, d_ff)
    output = hidden @ W2 + b2    # (seq_len, d_model)
    
    return output

# Research insight: FFN layers store factual knowledge
# "The capital of France is ___" → answer lives in FFN weights
# Attention routes the query; FFN supplies the answer
print("FFN: Attention routes, FFN knows.")
```

---

## ⚠️ 10. Common Mistakes & Misconceptions

**Mistake 1: Thinking attention "retrieves" from a database**
```
❌ Wrong: "Attention looks up facts from storage"
✅ Right: "Attention mixes representations of tokens in the CURRENT context"
          It can't access anything outside the context window
```

**Mistake 2: Confusing encoder vs decoder attention**
```
Encoder (BERT, embedding models):
  - Bidirectional — every token sees every other token
  - Used for: understanding, classification, embeddings

Decoder (GPT, Llama, Claude):
  - Causal / unidirectional — each token sees only past tokens
  - Used for: text generation
  
Most modern LLMs are decoder-only (GPT, Llama, Mistral, Claude)
```

**Mistake 3: Thinking more heads = always better**
```
GPT-2 small: 12 heads — great for many tasks
GPT-4:      ~128 heads — more specialized attention patterns

But beyond a point, adding heads has diminishing returns.
Architecture choices matter more than raw head count.
```

**Mistake 4: Misunderstanding what Q, K, V "are"**
```
❌ "Q is the question, K is the keyword, V is the value in a database"
✅ Q, K, V are all derived from the SAME input (in self-attention)
   They're just three different linear projections of the same embedding
   The "database" analogy breaks down — it's self-referential
```

---

## 🎤 11. Interview Questions — Day 2

**Q1. Why do we scale by √d_k in attention?**
> Without scaling, when d_k is large, dot products Q·K grow large in magnitude, pushing softmax into regions where gradients are near zero (saturation). Dividing by √d_k keeps the dot products in a reasonable range, maintaining healthy gradient flow during training. It's analogous to Xavier initialization — controlling variance of activations.

**Q2. What's the difference between encoder-only, decoder-only, and encoder-decoder Transformers?**
> Encoder-only (BERT): bidirectional attention, sees full context, used for understanding tasks (classification, embeddings). Decoder-only (GPT, Llama, Claude): causal attention, generates text autoregressively, one token at a time. Encoder-decoder (T5, BART): encoder processes input bidirectionally, decoder generates output conditioned on encoder output — used for translation and summarization. Most production LLMs today are decoder-only.

**Q3. Why is self-attention O(n²) in complexity and why does it matter?**
> Computing Q×Kᵀ requires comparing every token with every other token — that's n×n comparisons for a sequence of length n. Memory and compute both scale quadratically with sequence length. A 128K context window requires 128K² = 16 billion comparisons per layer. This is why very long contexts are expensive and why techniques like FlashAttention (memory-efficient attention) and sliding window attention exist.

**Q4. What is the role of the Feed-Forward Network in a Transformer block?**
> Attention mixes information across tokens (routing). The FFN processes each token independently after mixing. Research shows FFN layers act as "key-value memories" that store factual associations learned during training. The expand-contract structure (d_model → 4×d_model → d_model) gives the model a larger representational space to process within each layer.

**Q5. What is causal masking and why is it necessary for language model training?**
> During training, the model processes all tokens in parallel for efficiency. Without causal masking, a token could attend to future tokens — essentially "cheating" by looking at the answer. The causal mask sets future attention scores to -∞ (→ 0 after softmax), so each token can only attend to itself and previous tokens. This mirrors inference time, where future tokens genuinely don't exist yet.

---

## ✍️ 12. Practice Exercises

### Easy — Attention Visualizer
Modify the code above to run on any sentence you type. Try these and observe which tokens attend to which:
- *"The Prime Minister of India addressed the nation"*
- *"Sachin Tendulkar played cricket for India for 24 years"*

### Medium — Causal vs Bidirectional Comparison
Run both causal and bidirectional attention on the same sentence. Plot both heatmaps side by side. Write 3 observations about how the attention patterns differ and what that means for generation vs understanding tasks.

### Challenge (Portfolio-worthy) 🏆
Build a **Mini Transformer Block** that chains together:
- Positional encoding
- Single-head self-attention
- Residual connection (`output = attention(x) + x`)
- Layer normalization
- Feed-forward network
- Another residual connection

Test it on the token embeddings from Day 1's TaxSaathi bot. Print the shape at each stage and verify it matches expected dimensions. Write a README explaining each component. This demonstrates you understand Transformer internals — a question almost every GenAI interview asks.

---

## 📚 Resources for Today

- **Andrej Karpathy** — "Let's build GPT from scratch" (YouTube) — the best 2 hours you'll spend this week
- **"Attention Is All You Need"** (Vaswani et al., 2017) — just read the abstract and Figure 1
- **Jay Alammar** — "The Illustrated Transformer" — `jalammar.github.io/illustrated-transformer` — visual gold
- **3Blue1Brown** — "Attention in Transformers, visually explained" — YouTube

---

## 🔗 Connection to the Big Picture

```
Day 1                    Day 2                    Day 3 (Tomorrow)
──────────────          ──────────────────        ─────────────────────
Tokens → vectors   →   Attention enriches    →   Temperature, sampling,
Embeddings encode       those vectors with         hallucination, system
   meaning              cross-token context        prompts — how the
                                                   output head works
                                                   + your first chatbot
```
---

