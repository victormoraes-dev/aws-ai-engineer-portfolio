# Week 1, Day 2 — Review: Tokenization & Embeddings

*Study reference for the AWS Certified Machine Learning Engineer - Associate (MLA-C01)*

---

## Table of Contents

1. [Sprint Context](#1-sprint-context)
2. [Core Concepts](#2-core-concepts)
3. [Code & Implementation](#3-code--implementation)
4. [Certification Alignment (MLA-C01)](#4-certification-alignment-mla-c01)
5. [Key Takeaways](#5-key-takeaways)
6. [Glossary](#6-glossary)

---

## 1. Sprint Context

Week 1 focuses on building a domain-specialized LLM assistant using Fine-Tuning & Prompting. The project goal is to adapt a foundation model (Llama-3-8B) to understand legal/domain-specific jargon without training from scratch.

**Day 2 Mission:** Process a domain-specific dataset — tokenize the text and generate dense embeddings that capture semantic meaning. These artifacts feed directly into Thursday's LoRA fine-tuning session.

---

## 2. Core Concepts

### 2.1 The Tokenizer

Models don't read words — they read numbers. A tokenizer is a translator that converts human text into a sequence of integer IDs.

The tokenizer has a fixed vocabulary (~128K tokens for Llama-3). It chops text into subwords using Byte-Pair Encoding (BPE), assigning each subword a unique integer.

```
"What is an NDA?"  →  Tokenizer  →  [4512, 331, 268, 12049, 15]
```

**Key property:** The tokenizer performs **zero understanding**. It is purely a lookup table — no neural network, no semantics. The integer IDs are arbitrary labels with no inherent meaning.

### 2.2 The Embedding Model

Token IDs are arbitrary — the number 12,049 for "NDA" could have been any other number. The embedding model solves this by placing each token into a **learned semantic coordinate system**.

A neural network processes the token IDs through transformer layers. The final hidden state produces a vector — typically 4,096 dimensions for Llama-3-8B — that captures the token's **position in semantic space**.

```
Token ID [12049]  →  Embedding Model  →  [0.89, 0.92, -0.12, ..., 0.33]
                                               (4,096-dimensional vector)
```

Each dimension captures a latent semantic feature learned during training — formality level, legal-ness, sentiment, and thousands of other nuanced attributes the model can use.

### 2.3 Tokenizer vs Embedding Model

| | Tokenizer | Embedding Model |
|---|---|---|
| **Input** | Raw text string | Integer token IDs |
| **Output** | Integer IDs (e.g., `[4512, 331]`) | Dense vectors (e.g., `[0.89, -0.05]`) |
| **Understands meaning?** | ❌ No — pure lookup table | ✅ Yes — trained on billions of texts |
| **Parameters** | ~0 (vocabulary file only) | Billions (neural network layers) |

> **[CERTIFICATION FOCUS]** Domain 1 (Data Preparation for ML, 28%) tests your understanding of encoding techniques. The exam expects you to know the difference between tokenization (text → IDs) and embedding (IDs → semantic vectors), and when to use each AWS service for these tasks.

### 2.4 Tensors

A tensor is a NumPy array that can run on a GPU.

| Concept | NumPy | PyTorch Tensor |
|---|---|---|
| 1D array | `np.array([1,2,3])` — shape `(3,)` | `torch.tensor([1,2,3])` — shape `(3,)` |
| 2D matrix | `np.array([[1,2],[3,4]])` — shape `(2,2)` | `torch.tensor([[1,2],[3,4]])` — shape `(2,2)` |
| Location | CPU | GPU (`.to("cuda")`) or CPU |

### 2.5 Mean Pooling

The model produces one vector per token. Mean pooling averages them into one vector per document.

```
Tokens: ["I"]  ["signed"]  ["an"]  ["NDA"]  ["yesterday"]
          ↓        ↓          ↓       ↓          ↓
       [vec1]   [vec2]    [vec3]  [vec4]    [vec5]    ← 5 vectors
          
                (vec1 + vec2 + vec3 + vec4 + vec5) / 5
                               ↓
                      One document vector
```

### 2.6 4-Bit Quantization

A Llama-3-8B model needs ~16 GB of GPU memory at full precision. With 4-bit quantization (NF4 via `bitsandbytes`), it fits in ~4 GB — a ~75% reduction with minimal accuracy loss.

> **[CERTIFICATION FOCUS]** Domain 2 (ML Model Development, 26%) explicitly tests *"Reducing model size by altering data types, pruning, compression"* — quantization is a core exam topic.

### 2.7 Cosine Similarity

Cosine similarity measures the **angle** between two vectors — not the **length**.

```
Small angle (similar meaning):  cos(θ) ≈ 0.95  →  "NDA" and "confidential agreement"
Wide angle (unrelated):         cos(θ) ≈ 0.10  →  "NDA" and "pizza"
Opposite direction:             cos(θ) ≈ -0.80 →  "I love" and "I hate"
```

$$cos(\theta) = \frac{A \cdot B}{||A|| \times ||B||}$$

| Piece | Meaning |
|---|---|
| $A \cdot B$ | Dot product: multiply each dimension pair, sum everything |
| $\|\|A\|\|$ | Magnitude: length of the vector |
| $cos(\theta)$ | Score from -1 (opposite) to +1 (identical direction) |

**Why cosine and not Euclidean distance?** Vector length varies with document length. Two documents about "confidentiality" — one short, one long — have different magnitudes but the same semantic direction. Cosine ignores magnitude and captures meaning.

> **[CERTIFICATION FOCUS]** Domain 2 (ML Model Development, 26%) — *"Selecting and interpreting evaluation metrics."* Amazon Bedrock Knowledge Bases uses **cosine similarity** by default for vector search in RAG architectures.

---

## 3. Code & Implementation

The complete pipeline lives in `code/day2_embeddings.py`:

```python
# 1. LOAD TOKENIZER — converts text to integer IDs (no semantics)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 2. TOKENIZE — chop text into IDs, pad/truncate to 512 tokens
encoded = tokenizer(
    faq_texts,
    padding="max_length",
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
# Shape: [num_samples, 512] — every FAQ padded or truncated to equal length

# 3. LOAD EMBEDDING MODEL — 4-bit quantized for memory efficiency
from transformers import AutoModel, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModel.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"  # ~4 GB VRAM instead of ~16 GB
)

# 4. MEAN POOLING — average token vectors into one document vector
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts  # Shape: [batch_size, 4096]

# 5. GENERATE EMBEDDINGS — batch processing to avoid OOM
with torch.no_grad():
    for i in range(0, num_samples, 4):
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        batch_emb = mean_pool(outputs.last_hidden_state, batch_mask)
        all_embeddings.append(batch_emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0).numpy()  # Shape: [15, 4096]

# 6. VALIDATE — cosine similarity within categories
from sklearn.metrics.pairwise import cosine_similarity

for category in ["Contracts", "NDAs", "Intellectual Property"]:
    idx = df[df["category"] == category].index
    sim = cosine_similarity(embeddings[idx])
    mean_sim = (sim.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
    print(f"{category}: within-group similarity = {mean_sim:.4f}")
```

**Fallback:** The notebook falls back to `bert-base-uncased` if the Llama model is gated or unavailable — ensuring it runs on any environment (Colab, SageMaker Studio Lab, or local GPU).

---

## 4. Certification Alignment (MLA-C01)

| Concept | Domain | Weight |
|---|---|---|
| Tokenization (BPE, subword encoding) | Domain 1: Data Prep | 28% |
| Data cleaning & transformation | Domain 1: Data Prep | 28% |
| Embedding generation for semantic search | Domain 2: Model Development | 26% |
| Cosine similarity as evaluation metric | Domain 2: Model Development | 26% |
| 4-bit quantization for model compression | Domain 2: Model Development | 26% |

> **[CERTIFICATION FOCUS — Exam Trap]**
>
> **Question:** An ML engineer is building a RAG application on Amazon Bedrock. They need the most **cost-effective** embedding strategy for a vector database storing 10M document chunks (~500 tokens each). Which approach minimizes storage cost while maintaining retrieval accuracy?
>
> 1. Titan Text Embeddings V2 with full-precision (float32) vectors
> 2. Titan Text Embeddings V2 with **binary embeddings** ✅
> 3. SageMaker-hosted BERT embeddings with PCA reduction
> 4. Custom Llama-3 embeddings quantized to 4-bit
>
> **Answer:** Option 2 — Binary embeddings reduce storage by 96% (from 4 bytes/dim to 1 bit/dim) with <2% accuracy loss.

---

## 5. Key Takeaways

1. **Tokenizer ≠ Embedding Model.** The tokenizer maps text to arbitrary integer IDs (no semantics). The embedding model maps those IDs to meaningful semantic vectors.

2. **Embeddings are positions in semantic space.** An embedding of 4,096 dimensions is a coordinate in a learned meaning-space — not "4,096 similar words".

3. **Mean pooling creates document-level vectors.** The model outputs one vector per token. Mean pooling averages them into one vector per document, ignoring padding via the attention mask.

4. **Cosine similarity measures direction, not length.** Two documents about "confidentiality" — one short, one long — have the same direction. Cosine captures this correctly; Euclidean distance would not.

5. **4-bit quantization makes large models practical.** An 8B-parameter model drops from ~16 GB to ~4 GB VRAM with minimal accuracy loss.

6. **Embedding quality is measurable.** Within-category cosine similarity confirms semantically related documents cluster together — a prerequisite for RAG and fine-tuning.

7. **This pipeline feeds Day 4 LoRA fine-tuning.** The artifacts produced (`tokenized_dataset.pt`, `embeddings.npy`) are the direct input for the PEFT/LoRA training session.

---

## 6. Glossary

| Term | Definition |
|---|---|
| **Token** | A subword unit — the atomic piece of text the model processes |
| **Tokenizer** | Algorithm that converts text into integer token IDs (e.g., BPE, WordPiece) |
| **Embedding** | A dense vector representation of a token in a learned semantic coordinate space |
| **Embedding Dimension** | The number of coordinates in the semantic space (e.g., 1,536 for Titan V2, 4,096 for Llama-3-8B) |
| **Tensor** | A multi-dimensional array that can live on GPU — PyTorch's equivalent of NumPy arrays |
| **Mean Pooling** | Averaging all token embeddings in a sequence to produce a single document-level vector |
| **4-bit Quantization** | Compressing model weights from 16-bit to 4-bit precision, reducing memory ~75% |
| **Cosine Similarity** | Metric measuring the angle between two vectors, ignoring magnitude; range -1 to +1 |
| **Euclidean Distance** | Straight-line distance between two points in vector space; sensitive to magnitude |
| **PCA** | Principal Component Analysis — dimensionality reduction for visualizing high-dimensional data in 2D |
| **t-SNE** | Non-linear dimensionality reduction technique for visualization |
| **Attention Mask** | Binary mask indicating which tokens are real (1) vs padding (0) |

---

*Study reference for AWS MLA-C01 exam preparation.*
