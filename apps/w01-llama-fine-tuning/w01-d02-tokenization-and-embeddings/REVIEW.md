# Week 1, Day 2 — Tokenization & Embeddings

*Part of a 4-week accelerated program to build a production-grade AI engineering portfolio and earn the AWS Certified Machine Learning Engineer - Associate (MLA-C01) certification.*

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [The Problem: AWS Blocked the Model](#2-the-problem-aws-blocked-the-model)
3. [The Decision: A Dual-Track Strategy](#3-the-decision-a-dual-track-strategy)
4. [Core Concepts](#4-core-concepts)
5. [Implementation: Track A — Local Embeddings](#5-implementation-track-a--local-embeddings)
6. [Implementation: Track B — AWS Embeddings](#6-implementation-track-b--aws-embeddings)
7. [Track Comparison](#7-track-comparison)
8. [Glossary](#8-glossary)

---

## 1. What Was Built

The goal for Day 2 was to process a domain-specific dataset — tokenize the text and generate dense vector embeddings that capture semantic meaning. These artifacts feed directly into the LoRA fine-tuning session on Day 4.

The project scenario: *a company needs an LLM that understands its internal legal jargon without spending millions training a model from scratch.* The dataset was 15 legal FAQ entries across three categories — **Contracts** (5), **NDAs** (5), and **Intellectual Property** (5) — each containing a question, an answer, and a category label.

The deliverables were:
- `embeddings.npy` — a `[15, 384]` NumPy array of semantic vectors, one per FAQ entry
- `metadata.csv` — the original FAQ data aligned by row index to the embeddings
- `tsne_plot.png` — a 2D visualization confirming that same-category entries cluster together
- A cosine similarity validation confirming the embeddings encode domain structure

---

## 2. The Problem: AWS Blocked the Model

The original plan called for loading `meta-llama/Meta-Llama-3-8B-Instruct` on AWS SageMaker and generating embeddings via PyTorch on a GPU instance. AWS did not release the model to the account — a common blocker when models are gated, region-restricted, or pending approval.

Waiting was not an option. The project had a daily delivery cadence and Day 4's fine-tuning session depended on having embeddings ready.

---

## 3. The Decision: A Dual-Track Strategy

Rather than blocking on AWS, a dual-track strategy was adopted that separated concerns and kept the project moving:

```text
TRACK A: LOCAL (Running Now — No Cloud Required)

  dataset.json ──→ sentence-transformers/all-MiniLM-L6-v2
  (15 FAQs)        (22M params, 384-dim embeddings)
                         ↓
                    embeddings.npy
                    PCA/t-SNE plots
                    Cosine similarity validation

  Runs 100% locally via PyTorch on GPU/CPU
  No AWS account needed. No API key needed.
  Model: ~90 MB VRAM — fits on ANY machine


TRACK B: AWS (Ready When Model Becomes Available)

  dataset.json ──→ AutoModel.from_pretrained(...)
  (15 FAQs)        (Llama-3-8B, 4-bit quantized)
                         ↓
                    embeddings.npy (4,096-dim)
                    Same PCA/t-SNE pipeline
                    Same cosine similarity validation

  Requires: AWS account with SageMaker GPU quota
  Requires: HuggingFace token for gated model access
```

Track A is not a replacement — it is an interim strategy. When the Llama-3-8B model becomes available, the exact same pipeline runs with Track B's script and produces 4,096-dimensional embeddings instead of 384. The dataset, validation logic, and visualization code are **identical** — only the model loading changes.

`all-MiniLM-L6-v2` was chosen for Track A because it is the standard open-source embedding model used in production RAG systems worldwide, and it maps directly to the same concepts tested in the MLA-C01 exam on vector databases and semantic search.

| Property | Value | Comparison |
|---|---|---|
| **Parameters** | 22 million | vs 8 billion (Llama-3-8B) |
| **VRAM required** | ~90 MB | vs ~4 GB (4-bit quantized Llama) |
| **Embedding dimension** | 384 | vs 4,096 (Llama-3-8B) |
| **Speed** | 5x faster than BERT-base | Instant on CPU |
| **License** | Apache 2.0 | Free, no restrictions |

---

## 4. Core Concepts

### The Tokenizer

Models don't read words — they read numbers. A tokenizer is a translator that converts human text into a sequence of integer IDs using a fixed vocabulary (~128K tokens for Llama-3, ~30K for BERT). It chops text into subwords using Byte-Pair Encoding (BPE), assigning each subword a unique integer.

```
"What is an NDA?"  →  Tokenizer  →  [4512, 331, 268, 12049, 15]
```

The tokenizer performs **zero understanding**. It is purely a lookup table — no neural network, no semantics. The integer IDs are arbitrary labels with no inherent meaning.

### The Embedding Model

Token IDs are arbitrary — the number 12,049 for "NDA" could have been any other number. The embedding model solves this by placing each token into a **learned semantic coordinate system**. A neural network processes the token IDs through transformer layers, and the final hidden state produces a vector — typically 384 to 4,096 dimensions — that captures the token's position in semantic space.

```
Token ID [12049]  →  Embedding Model  →  [0.89, 0.92, -0.12, ..., 0.33]
                                               (384 or 4,096 dimensions)
```

Each dimension captures a latent semantic feature learned during training — formality level, legal-ness, sentiment, and thousands of other nuanced attributes.

### Tokenizer vs Embedding Model

| | Tokenizer | Embedding Model |
|---|---|---|
| **Input** | Raw text string | Integer token IDs |
| **Output** | Integer IDs (e.g., `[4512, 331]`) | Dense vectors (e.g., `[0.89, -0.05]`) |
| **Understands meaning?** | ❌ No — pure lookup table | ✅ Yes — trained on billions of texts |
| **Parameters** | ~0 (vocabulary file only) | Millions to billions (neural network) |

### Cosine Similarity

Cosine similarity measures the **angle** between two vectors — not the length. This matters because two documents about "confidentiality" — one short, one long — have different magnitudes but the same semantic direction. Cosine ignores magnitude and captures meaning; Euclidean distance would not.

```
Small angle (similar meaning):  cos(θ) ≈ 0.95  →  "NDA" and "confidential agreement"
Wide angle (unrelated):         cos(θ) ≈ 0.10  →  "NDA" and "pizza"
```

Amazon Bedrock Knowledge Bases uses cosine similarity by default for vector search in RAG architectures.

---

## 5. Implementation: Track A — Local Embeddings

### Repository Structure

```text
week-1-fine-tuning/
├── dataset.json                          ← 15 legal FAQs
├── notebooks/
│   ├── day2_local_embeddings.py          ← Track A: sentence-transformers
│   ├── day2_aws_embeddings.py            ← Track B: Llama-3-8B on SageMaker
│   └── day2_summary.py                   ← Compact version of Track A
├── artifacts/
│   ├── embeddings.npy                    ← [15, 384] embedding matrix
│   ├── metadata.csv                      ← FAQ metadata aligned by row index
│   └── tsne_plot.png                     ← PCA/t-SNE visualization
└── README.md
```

### Imports

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
```

`SentenceTransformer` wraps tokenization, embedding generation, and mean pooling into a single object. `PCA` and `TSNE` are used only for visualization. `cosine_similarity` computes pairwise similarity scores between embedding vectors.

### Step 1 — Dataset and Text Preparation

```python
faq_data = [
    {"category": "Contracts", "question": "What makes a contract legally binding?",
     "answer": "A contract is binding when it includes an offer, acceptance, consideration, and mutual intent to be bound."},
    # ... 15 entries total (5 Contracts, 5 NDAs, 5 IP)
]

texts = [f"Question: {item['question']} Answer: {item['answer']}" for item in faq_data]
categories = [item['category'] for item in faq_data]
```

Each FAQ entry is concatenated into a single string per entry. Prefixing with `"Question:"` and `"Answer:"` gives the model light structural context. `categories` is extracted as a parallel list so index `i` in `categories` always corresponds to index `i` in `texts` and later in `embeddings`.

### Step 2 — Load Model

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

Downloads and caches the `all-MiniLM-L6-v2` checkpoint (~90 MB) from HuggingFace Hub on first run. This model was trained with a contrastive objective to produce sentence-level embeddings — its vectors are calibrated so that semantically similar sentences have high cosine similarity. It runs on CPU with no GPU required.

### Step 3 — Generate Embeddings

```python
embeddings = model.encode(texts)
print(f"Embeddings shape: {embeddings.shape}")  # [15, 384]
```

`model.encode()` runs the full pipeline internally: tokenizes each string, passes the token IDs through the transformer layers, and applies mean pooling to produce one 384-dimensional vector per input. The return value is a NumPy array of shape `[15, 384]`.

### Step 4 — PCA → t-SNE Visualization

```python
pca = PCA(n_components=min(len(embeddings), 50)).fit_transform(embeddings)
tsne = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(pca)

plt.figure(figsize=(10, 7))
for category in set(categories):
    idx = [i for i, cat in enumerate(categories) if cat == category]
    plt.scatter(tsne[idx, 0], tsne[idx, 1], label=category, s=100, alpha=0.7)
plt.title("t-SNE Visualization of Legal FAQ Embeddings (all-MiniLM-L6-v2)")
plt.legend()
plt.savefig("artifacts/tsne_plot.png")
```

With only 15 samples, t-SNE cannot be applied directly to 384-dimensional embeddings — it requires more samples than dimensions. PCA first reduces to `min(15, 50) = 15` principal components, then t-SNE maps those to 2D for plotting. `perplexity=5` is deliberately low because t-SNE's perplexity must be less than the number of samples. If the embeddings capture domain semantics correctly, entries from the same category cluster visually in the 2D plot.

### Step 5 — Within-Category Cosine Similarity Validation

```python
for category in set(categories):
    idx = [i for i, cat in enumerate(categories) if cat == category]
    sim = cosine_similarity(embeddings[idx])
    upper = np.triu_indices(len(idx), k=1)
    avg_sim = np.mean(sim[upper])
    print(f"{category}: {avg_sim:.4f}")
```

`cosine_similarity()` computes the full `[n, n]` pairwise similarity matrix for each category's rows. `np.triu_indices(len(idx), k=1)` returns the upper triangle indices, excluding the diagonal (self-similarity, always 1.0), to avoid counting each pair twice. A high within-category score (> 0.6) confirms that entries about Contracts are semantically closer to other Contracts entries than to NDA or IP entries.

### Step 6 — Save Artifacts

```python
np.save("artifacts/embeddings.npy", embeddings)
pd.DataFrame(faq_data).to_csv("artifacts/metadata.csv", index=False)
```

`np.save()` serializes the `[15, 384]` array to a binary `.npy` file, preserving exact float32 values and shape for direct loading in downstream notebooks. `index=False` omits the pandas row index since the positional order already aligns with the `.npy` rows.

---

## 6. Implementation: Track B — AWS Embeddings

Track B uses `meta-llama/Llama-3.2-3B-Instruct` loaded with 4-bit quantization, producing 3,072-dimensional embeddings. Unlike Track A, every step of the pipeline is explicit — tokenization, model loading, pooling, and batch inference are all separate operations.

### Cell 1 — Imports and Tokenizer

```python
import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
```

The tokenizer is loaded separately from the model because tokenization requires no GPU and no neural network — it is a pure lookup table. Separating this step makes it explicit that the tokenizer performs zero semantic processing.

### Cell 2 — Tokenization

```python
encoded = tokenizer(
    faq_texts,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
# Shape: [num_samples, 512]
```

`padding="max_length"` pads every sequence to exactly 512 tokens — required because PyTorch tensors must be rectangular. `return_tensors="pt"` returns PyTorch tensors for passing to the GPU model. The resulting `encoded` object contains `input_ids` (integer token IDs) and `attention_mask` (1 for real tokens, 0 for padding).

### Cell 3 — 4-bit Quantization Config and Model Load

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",  # ~2.5 GB VRAM instead of ~16 GB
)
```

`load_in_4bit=True` stores each weight as a 4-bit integer instead of 16-bit float, reducing memory by ~75%. `bnb_4bit_quant_type="nf4"` uses NormalFloat4, a data type optimized for normally-distributed neural network weights. `device_map="auto"` automatically distributes model layers across available GPUs and CPU RAM.

### Cell 4 — Mean Pooling Function

```python
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts  # Shape: [batch_size, 3072]
```

The model outputs one vector per token (`last_hidden_state` shape: `[batch, 512, 3072]`). Mean pooling collapses those 512 token vectors into one document-level embedding by averaging — but padding tokens must be excluded or they corrupt the average. `clamp(min=1e-9)` prevents division by zero if a sequence is entirely padding.

### Cell 5 — Generate Embeddings in Batches

```python
with torch.no_grad():
    for i in range(0, num_samples, 4):
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        batch_emb = mean_pool(outputs.last_hidden_state, batch_mask)
        all_embeddings.append(batch_emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0).numpy()  # Shape: [15, 3072]
```

`torch.no_grad()` disables gradient computation — gradients are only needed during training. Processing 4 samples at a time keeps peak VRAM predictable. `.cpu()` moves each batch back to CPU RAM before appending, freeing GPU VRAM after each batch.

---

## 7. Track Comparison

| Aspect | Track A (Local) | Track B (AWS) |
|---|---|---|
| **Model** | `all-MiniLM-L6-v2` | `Meta-Llama-3.2-3B-Instruct` |
| **Parameters** | 22M | 3B |
| **Embedding dim** | 384 | 3,072 |
| **VRAM** | ~90 MB | ~2.5 GB (4-bit quantized) |
| **Tokenization** | Built into `model.encode()` | Manual via `AutoTokenizer` |
| **Mean pooling** | Built into `model.encode()` | Manual `mean_pool()` function |
| **Setup time** | `pip install sentence-transformers` | HuggingFace token + AWS GPU quota |
| **Status** | ✅ Running now | ⏳ Ready when AWS releases the model |

---

## 8. Glossary

| Term | Definition |
|---|---|
| **Token** | A subword unit — the atomic piece of text the model processes |
| **Tokenizer** | Algorithm that converts text into integer token IDs (e.g., BPE, WordPiece) |
| **Embedding** | A dense vector representation of a token in a learned semantic coordinate space |
| **Embedding Dimension** | The number of coordinates in the semantic space (e.g., 384 for MiniLM, 4,096 for Llama-3-8B) |
| **Tensor** | A multi-dimensional array that can live on GPU — PyTorch's equivalent of NumPy arrays |
| **Mean Pooling** | Averaging all token embeddings in a sequence to produce a single document-level vector |
| **4-bit Quantization** | Compressing model weights from 16-bit to 4-bit precision, reducing memory ~75% |
| **Cosine Similarity** | Metric measuring the angle between two vectors, ignoring magnitude; range -1 to +1 |
| **PCA** | Principal Component Analysis — dimensionality reduction for visualizing high-dimensional data |
| **t-SNE** | Non-linear dimensionality reduction technique for 2D visualization |
| **Attention Mask** | Binary mask indicating which tokens are real (1) vs padding (0) |
| **SentenceTransformer** | HuggingFace library that wraps tokenization + embedding + pooling into a single `encode()` call |
