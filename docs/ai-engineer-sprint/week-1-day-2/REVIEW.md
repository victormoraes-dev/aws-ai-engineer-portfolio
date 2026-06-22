## Week 1, Day 2 — Complete Review: Tokenization & Embeddings

**Study reference for the AWS Certified Machine Learning Engineer - Associate (MLA-C01)**

---

### Table of Contents

- [1. Course Roadmap Overview](#1-course-roadmap-overview)
- [2. Core Concepts Learned](#2-core-concepts-learned)
- [3. Code & Implementation](#3-code--implementation)
- [4. Certification Alignment (MLA-C01)](#4-certification-alignment-mla-c01)
- [5. Key Takeaways](#5-key-takeaways)
- [6. Glossary](#6-glossary)
- [7. LinkedIn Post (Build in Public)](#7-linkedin-post-build-in-public)

---

### 1. Course Roadmap Overview

**Sprint Context:** Week 1 focuses on building a domain-specialized LLM assistant using Fine-Tuning & Prompting. The project goal is to adapt a foundation model (Llama-3.2-3B) to understand legal/domain-specific jargon without training from scratch.

**Day 2 Mission:** Process a domain-specific dataset — tokenize the text and generate dense embeddings that capture semantic meaning. These artifacts feed directly into Thursday's LoRA fine-tuning session.

| Concept | MLA-C01 Domain | Weight | Relevance |
|---|---|---|---|
| Tokenization & Encoding | Domain 1: Data Preparation for ML | 28% | *Task 1.2* — Encoding techniques (tokenization, BPE) |
| Embedding Generation | Domain 2: ML Model Development | 26% | *Task 2.1* — Choosing the right representation for modeling |
| Data Transformation | Domain 1: Data Preparation for ML | 28% | *Task 1.2* — Transforming data using AWS tools |
| Cosine Similarity | Domain 2: ML Model Development | 26% | *Task 2.3* — Model evaluation metrics and techniques |

---

### 2. Core Concepts Learned

#### 2.1 The Tokenizer (Station 1 in the Assembly Line)

**Intuition:** Models don't read words — they read numbers. A tokenizer is a translator that converts human text into a sequence of integer IDs.

**Mechanism:** The tokenizer has a fixed vocabulary (e.g., ~128K tokens for Llama-3). It chops text into subwords using Byte-Pair Encoding (BPE), assigning each subword a unique integer.

```
"What is an NDA?"  →  Tokenizer  →  [4512, 331, 268, 12049, 15]
```

**Key Property:** The tokenizer performs **zero understanding**. It is purely a lookup table — no neural network, no semantics. The integer IDs are arbitrary labels with no inherent meaning.

#### 2.2 The Embedding Model (Station 2 in the Assembly Line)

**Intuition:** Token IDs are arbitrary. The number 12,049 for "NDA" could have been any other number. The embedding model solves this by placing each token into a **learned semantic coordinate system**.

**Mechanism:** A multi-billion parameter neural network processes the token IDs through multiple transformer layers. The final hidden state produces a vector — typically 1,536 to 3,072 dimensions — that captures the token's **position in semantic space**.

```
Token ID [12049]  →  Embedding Model  →  [0.89, 0.92, -0.12, ..., 0.33]
                                               (1,536-dimensional vector)
```

**What each dimension captures:** Not "similar words" but latent semantic features learned during training — formality level, legal-ness, sentiment, noun-verb-ness, and thousands of other nuanced attributes we cannot name but the model can use.

#### 2.3 The Critical Distinction: Tokenizer vs Embedding Model

| | Tokenizer | Embedding Model |
|---|---|---|
| **Input** | Raw text string | Integer token IDs |
| **Output** | Integer IDs (e.g., `[4512, 331]`) | Dense vectors (e.g., `[0.89, -0.05]`) |
| **Understands meaning?** | ❌ No. Pure lookup table. | ✅ Yes. Trained on billions of texts. |
| **Parameters** | ~0 (vocabulary file only) | Billions (actual neural network layers) |

> **[CERTIFICATION FOCUS]** Domain 1 (Data Preparation for ML, 28%) tests your understanding of encoding techniques. The exam expects you to know the difference between tokenization (text → IDs) and embedding (IDs → semantic vectors), and when to use each AWS service for these tasks.

#### 2.4 What Is a Tensor?

**Intuition:** A tensor is a NumPy array that can run on a GPU. Same concept, different hardware.

| Concept | NumPy | PyTorch Tensor |
|---|---|---|
| 1D array | `np.array([1,2,3])` — shape `(3,)` | `torch.tensor([1,2,3])` — shape `(3,)` |
| 2D matrix | `np.array([[1,2],[3,4]])` — shape `(2,2)` | `torch.tensor([[1,2],[3,4]])` — shape `(2,2)` |
| Location | CPU | GPU (`.to("cuda")`) or CPU |

#### 2.5 Mean Pooling

**Intuition:** The model produces one vector per token. But we want one vector per document. Mean pooling averages them.

```
Tokens: ["I"]  ["signed"]  ["an"]  ["NDA"]  ["yesterday"]
          ↓        ↓          ↓       ↓          ↓
       [vec1]   [vec2]    [vec3]  [vec4]    [vec5]    ← 5 vectors
          
                (vec1 + vec2 + vec3 + vec4 + vec5) / 5
                               ↓
                      One document vector
```

#### 2.6 4-Bit Quantization

**Intuition:** A JPEG compresses a 24 MB photo to 3 MB. 4-bit quantization does the same for neural networks — from 16-bit precision to 4-bit precision, reducing memory by ~75%.

A Llama-3.2-3B model needs ~16 GB of GPU memory at full precision. With 4-bit quantization, it fits in ~2.5 GB.

> **[CERTIFICATION FOCUS]** Domain 2 (ML Model Development, 26%) explicitly tests *"Reducing model size by altering data types, pruning, compression"* — quantization is a core exam topic.

#### 2.7 Cosine Similarity

**Intuition:** Imagine two arrows starting from the same point. Cosine similarity measures the **angle** between them — not the **length**.

```
Small angle (similar meaning):    cos(θ) ≈ 0.95  →  "NDA" and "confidential agreement"
Wide angle (unrelated):           cos(θ) ≈ 0.10  →  "NDA" and "pizza"
Opposite direction (opposite):    cos(θ) ≈ -0.80 →  "I love" and "I hate"
```

**Formula:**

$$cos(\theta) = \frac{A \cdot B}{||A|| \times ||B||}$$

| Piece | Meaning |
|---|---|
| $$A \cdot B$$ | Dot product: multiply each dimension pair, sum everything |
| $$||A||$$ | Magnitude: length of the vector |
| $$cos(\theta)$$ | Score from -1 (opposite) to +1 (identical direction) |

**Why cosine and not Euclidean distance?** Because vector length varies with document length. Two documents about "confidentiality" — one short, one long — have very different magnitudes but the same semantic direction. Cosine ignores magnitude and captures meaning.

> **[CERTIFICATION FOCUS]** Domain 2 (ML Model Development, 26%) — *"Selecting and interpreting evaluation metrics."* The exam expects you to know that Amazon Bedrock Knowledge Bases uses **cosine similarity** by default for vector search in RAG architectures.

---

### 3. Code & Implementation

The following complete pipeline was deployed in the notebook `day2_embeddings.py`:### 3. Code & Implementation

The complete pipeline lives in `day2_embeddings.py`. Here is the critical code flow, condensed:

```python
# 1. LOAD TOKENIZER — converts text to integer IDs (no semantics)
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

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
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"  # ~2.5 GB VRAM instead of ~16 GB
)

# 4. MEAN POOLING — average token vectors into one document vector
def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts  # Shape: [batch_size, 1536]

# 5. GENERATE EMBEDDINGS — batch processing to avoid OOM
with torch.no_grad():
    for i in range(0, num_samples, 4):
        outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        batch_emb = mean_pool(outputs.last_hidden_state, batch_mask)
        all_embeddings.append(batch_emb.cpu())

embeddings = torch.cat(all_embeddings, dim=0).numpy()  # Shape: [15, 1536]

# 6. VALIDATE — cosine similarity within categories
from sklearn.metrics.pairwise import cosine_similarity

for category in ["Contracts", "NDAs", "IP"]:
    idx = df[df["category"] == category].index
    sim = cosine_similarity(embeddings[idx])
    mean_sim = (sim.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
    print(f"{category}: within-group similarity = {mean_sim:.4f}")
```

**Fallback mechanism:** The notebook gracefully falls back to `bert-base-uncased` if the Llama model is gated or unavailable — ensuring it runs on any environment (Colab, SageMaker Studio Lab, or local GPU).

---

### 4. Certification Alignment (MLA-C01)

| Concept | Domain | Weight | Exam Relevance |
|---|---|---|---|
| **Tokenization (BPE, subword encoding)** | Domain 1: Data Prep | 28% | *Task 1.2* — "Encoding techniques (one-hot, binary, label encoding, tokenization)" |
| **Data cleaning & transformation** | Domain 1: Data Prep | 28% | *Task 1.2* — "Data cleaning techniques (missing data, deduplication)" |
| **Embedding generation for semantic search** | Domain 2: Model Development | 26% | *Task 2.1* — "Choosing appropriate ML models/algorithms" |
| **Cosine similarity as evaluation metric** | Domain 2: Model Development | 26% | *Task 2.3* — "Selecting and interpreting evaluation metrics" |
| **4-bit quantization for model compression** | Domain 2: Model Development | 26% | *Task 2.2* — "Reducing model size by altering data types" |
| **Feature engineering & storage** | Domain 1: Data Prep | 28% | *Task 1.2* — "Creating and managing features (SageMaker Feature Store)" |

> **[CERTIFICATION FOCUS — Exam Trap Q&A]**

> **Question:** An ML engineer is building a RAG application on Amazon Bedrock. They need the most **cost-effective** embedding strategy for a vector database storing 10M document chunks (~500 tokens each). Which approach minimizes storage cost while maintaining retrieval accuracy?
>
> 1. Titan Text Embeddings V2 with full-precision (float32) vectors
> 2. Titan Text Embeddings V2 with **binary embeddings** ✅
> 3. SageMaker-hosted BERT embeddings with PCA reduction
> 4. Custom Llama-3 embeddings quantized to 4-bit
>
> **Answer:** Option 2 — Binary embeddings reduce storage by 96% (from 4 bytes/dim to 1 bit/dim) with <2% accuracy loss.

---

### 5. Key Takeaways

1. **Tokenizer ≠ Embedding Model.** The tokenizer maps text to arbitrary integer IDs (no semantics). The embedding model maps those IDs to meaningful semantic vectors. Never confuse the two.

2. **Embeddings are positions in semantic space.** An embedding of 1,536 dimensions is not "1,536 similar words" — it's a coordinate in a learned meaning-space. Each dimension captures a latent semantic feature the model discovered during training.

3. **Mean pooling creates document-level vectors.** The model outputs one vector per token. Mean pooling averages them into one vector per document, ignoring padding via the attention mask.

4. **Cosine similarity measures direction, not length.** Two documents about "confidentiality" — one short, one long — have different vector lengths but the same direction. Cosine similarity captures this correctly; Euclidean distance would not.

5. **4-bit quantization makes large models practical on T4 GPUs.** A 3B-parameter model drops from ~16 GB to ~2.5 GB VRAM with minimal accuracy loss, enabling local fine-tuning on consumer hardware.

6. **Embedding quality is measurable.** Within-category cosine similarity confirms that semantically related documents cluster together in vector space — a prerequisite for downstream tasks like RAG and fine-tuning.

7. **The pipeline feeds Thursday's LoRA fine-tuning.** The artifacts prepared today (`tokenized_dataset.pt`, `embeddings.npy`) are the direct input for the PEFT/LoRA training session on Day 4.

---

### 6. Glossary

| Term | Definition |
|---|---|
| **Token** | A subword unit — the atomic piece of text the model processes (word fragment, word, or character) |
| **Tokenizer** | Algorithm that converts text into integer token IDs (e.g., BPE, WordPiece) |
| **Embedding** | A dense vector representation of a token in a learned semantic coordinate space |
| **Embedding Dimension** | The number of coordinates in the semantic space (e.g., 1,536 for Titan V2, 3,072 for Llama-3.2-3B) |
| **Tensor** | A multi-dimensional array that can live on GPU — PyTorch's equivalent of NumPy arrays |
| **Mean Pooling** | Averaging all token embeddings in a sequence to produce a single document-level vector |
| **4-bit Quantization** | Compressing model weights from 16-bit to 4-bit precision, reducing memory ~75% |
| **Cosine Similarity** | A metric measuring the angle between two vectors, ignoring magnitude; range -1 to +1 |
| **Euclidean Distance** | Straight-line distance between two points in vector space; sensitive to magnitude |
| **PCA** | Principal Component Analysis — dimensionality reduction technique to visualize high-dimensional data in 2D |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding — non-linear dimensionality reduction for visualization |
| **Attention Mask** | Binary mask indicating which tokens are real (1) vs padding (0), used to ignore padding during computation |

---

### 7. LinkedIn Post (Build in Public)

> **Context:** Training a domain-specific LLM requires high-quality data preparation. Raw text won't cut it — models need tokenized, embedded data before any fine-tuning can succeed.
>
> **Challenge:** Most teams skip rigorous data preparation and wonder why their fine-tuned models underperform. Without proper tokenization analysis and embedding validation, you're optimizing on top of a weak foundation.
>
> **Technical Solution:** Today I built a data preparation pipeline for a legal FAQ dataset (Contracts, NDAs, IP categories):
>
> 1. **Tokenized** 15 legal FAQ entries using the Llama-3.2 tokenizer (BPE, 128K vocabulary) → analyzed the token length distribution to set optimal padding/truncation at 512 tokens
> 2. **Generated dense embeddings** using a 4-bit quantized Llama-3.2-3B-Instruct model → mean pooling reduced 512 token vectors into single 3,072-dim document representations
> 3. **Validated semantic clustering** via PCA visualization and within-category cosine similarity → confirmed that NDAs cluster together, Contracts cluster together, and IP topics form a distinct semantic group
>
> The result? Embedding artifacts ready for LoRA fine-tuning later this week — with measurable quality metrics proving the data is well-structured before training begins.
>
> **Cost insight:** Running this entire pipeline consumed ~2.5 GB VRAM thanks to 4-bit quantization. A full-precision 3B model would need ~16 GB.
>
> **GitHub link:** [coming soon]
>
> #AIEngineering #AWS #MachineLearning #LLM #FineTuning

---