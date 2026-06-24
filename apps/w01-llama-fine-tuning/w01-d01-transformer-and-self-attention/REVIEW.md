# Week 1, Day 1 — Review: Foundations of AI Engineering

*Study reference for the AWS Certified Machine Learning Engineer - Associate (MLA-C01)*

---

## Table of Contents

1. [Course Roadmap Overview](#1-course-roadmap-overview)
2. [Transformer Architecture & Self-Attention](#2-transformer-architecture--self-attention)
3. [The Inference Pipeline](#3-the-inference-pipeline)
4. [GPU VRAM Physics & Quantization](#4-gpu-vram-physics--quantization)
5. [RAG Architecture](#5-rag-architecture-retrieval-augmented-generation)
6. [PyTorch Learning Ladder](#6-pytorch-learning-ladder)
7. [GPU Validation Script](#7-gpu-validation-script)
8. [inference.py — Line-by-Line Breakdown](#8-inferencepy--line-by-line-breakdown)
9. [Key Certification Takeaways (MLA-C01)](#9-key-certification-takeaways-mla-c01)
10. [Glossary](#10-glossary)

---

## 1. Course Roadmap Overview

A 4-week accelerated program designed to build a production-grade AI portfolio and earn the **AWS Certified Machine Learning Engineer - Associate (MLA-C01)** certification.

**Week 1: Fine-Tuning & Prompting** — Creation of a specialized Llama-3 8B assistant using PEFT/LoRA and quantization techniques.

### Weekly Schedule (Week 1)

| Day | Focus | Deliverable |
|---|---|---|
| **Monday** | Transformer/Attention theory, PyTorch setup, basic inference | Repository, notebook, GPU validation |
| **Tuesday** | Data preparation, tokenization, embeddings | JSONL dataset for fine-tuning |
| **Wednesday** | Prompt engineering on AWS Bedrock | Zero-shot vs Few-shot baselines |
| **Thursday** | PEFT/LoRA fine-tuning | Trained LoRA adapters |
| **Friday** | 4-bit quantization optimization | Quantized model, cost analysis |

### Exam Domains (MLA-C01)

| Domain | Weight | Week 1 Lab |
|---|---|---|
| Domain 1: Data Preparation | 28% | Tuesday (Tokenization, JSONL) |
| Domain 2: ML Model Development | 26% | Thursday (LoRA fine-tuning) |
| Domain 3: Deployment & Orchestration | 22% | SageMaker instance provisioning |
| Domain 4: Monitoring & Security | 24% | Cost analysis, budgets |

**Target:** Score **720/1000** to pass.

---

## 2. Transformer Architecture & Self-Attention

### The Problem with Older Models

RNNs and LSTMs process text sequentially (word-by-word, left-to-right). This creates a computational bottleneck because they cannot parallelize across GPUs effectively.

### The Transformer Solution

Introduced in the paper *"Attention Is All You Need"* (Vaswani et al., 2017). The Transformer processes the **entire sequence simultaneously** using the **Self-Attention mechanism**, enabling massive parallelization across GPU hardware.

### Self-Attention Mechanism

- Calculates an **attention score** (weight) between every word in a sequence at the same time
- Allows the model to capture deep contextual relationships regardless of token distance
- Every token can "see" every other token in the sequence

### The Three Transformer Architectures

| Architecture | Example | Use Case |
|---|---|---|
| **Encoder-Only** | BERT | Classification, Sentiment Analysis, NER |
| **Decoder-Only** | Llama-3, GPT-4 | Auto-regressive text generation, Chatbots |
| **Encoder-Decoder** | T5, BART | Translation, Summarization |

### Why Decoder-Only?

Llama-3 uses a **Decoder-Only** architecture because its sole objective is **auto-regressive generation** (predicting the next token). The Decoder has its own Self-Attention mechanism (Masked Self-Attention) to understand the prompt, making a separate Encoder unnecessary for generative tasks.

### Causal (Masked) Self-Attention

In a Decoder-Only model, each token can only attend to itself and previous tokens — never future tokens. This is enforced by a triangular mask matrix that sets future positions to `-infinity` before softmax.

---

## 3. The Inference Pipeline

The lifecycle of a single prompt through Llama-3:

```
"Hello world"  →  Tokenizer  →  [15043, 3186]  →  Model (GPU)  →  [2990]  →  Tokenizer  →  "!"
```

**Step 1: Tokenization** — `AutoTokenizer` converts raw text into Token IDs (integers). Llama-3 uses a BPE tokenizer with ~32,000 tokens in vocabulary.

**Step 2: Embedding Lookup** — Each Token ID is mapped to a dense vector of dimension **4096**.

**Step 3: Decoder Layers** — The input passes through **32 Transformer Decoder layers**. Each layer contains: Masked Multi-Head Self-Attention + Feed-Forward Network (SwiGLU) + Residual connections + LayerNorm.

**Step 4: Output Projection (LM Head)** — The final hidden state passes through `nn.Linear(4096, 32000)` producing **logits** — raw scores for each token in the vocabulary.

**Step 5: Softmax & Sampling** — Logits are converted to probabilities via softmax. The next token is selected (Greedy, Top-K, or Temperature sampling).

**Step 6: Auto-Regressive Loop** — The predicted token is appended to the input sequence. Steps 1–5 repeat until `<|end_of_text|>` is generated.

---

## 4. GPU VRAM Physics & Quantization

### Memory Requirements for Llama-3 8B

| Precision | Bits per Parameter | 8B Model VRAM |
|---|---|---|
| FP32 (Full) | 32 bits = 4 bytes | ~32 GB |
| FP16 / BF16 | 16 bits = 2 bytes | ~16 GB |
| INT4 (4-bit) | 4 bits = 0.5 bytes | ~4 GB |

### The KV Cache Problem

The **KV Cache** (Key-Value Cache) stores the mathematical representations of all previous tokens to avoid recalculating them for each new token. It grows **linearly with context length**:

```
KV Cache Size ≈ 2 × num_layers × hidden_dim × seq_len × precision_bytes
```

For Llama-3 8B (32 layers, 4096 hidden dim):
- 4K context → ~2 GB at FP16
- 32K context → ~16 GB at FP16

This is the **primary cause of CUDA Out of Memory (OOM) errors** during long context windows.

### Solution: 4-bit Quantization (NF4)

Uses the `bitsandbytes` library to compress model weights from ~16 GB to ~4 GB, freeing **~12 GB of VRAM** for the KV Cache and long context windows.

### Target Hardware

| Instance | GPU | VRAM | CUDA Cores | Use Case |
|---|---|---|---|---|
| `ml.g5.2xlarge` | NVIDIA A10G | 24 GB | 9,216 | Fine-tuning 7B-13B models |

---

## 5. RAG Architecture (Retrieval-Augmented Generation)

### The Problem

A 500-page PDF cannot be fed entirely into the LLM's context window — the KV Cache would explode and OOM.

### The RAG Solution

Instead of sending the entire document, RAG retrieves only the relevant chunks:

1. **Chunking** — Split the document into small segments (256-512 tokens each)
2. **Embedding** — Convert each chunk into a vector representation
3. **Storage** — Store embeddings in a Vector Database (Amazon OpenSearch Serverless, Pinecone)
4. **Retrieval** — User query is embedded, find the 3-5 most similar chunks
5. **Generation** — Query + retrieved chunks are sent to the LLM as context

### AWS Services for RAG

| Service | Role |
|---|---|
| **Amazon OpenSearch Serverless** | Vector database for similarity search |
| **Amazon Kendra** | Managed enterprise search |
| **Amazon Bedrock Knowledge Bases** | Fully managed, serverless RAG |

### RAG vs Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|---|---|---|
| Knowledge source | External documents | Model weights |
| Update cost | Re-index documents | Re-train model |
| Hallucination risk | Lower (grounded in retrieved text) | Higher |
| Best for | Facts, policies, documentation | Tone, style, behavior |

---

## 6. PyTorch Learning Ladder

### Level 1: Tensors & CUDA

**Concept:** PyTorch processes **Tensors** — multi-dimensional matrices.

| Type | Dimension | Example |
|---|---|---|
| Scalar | 0D | `tensor(5)` |
| Vector | 1D | `tensor([1, 2, 3])` |
| Matrix | 2D | `tensor([[1,2],[3,4]])` |
| Tensor | 3D+ | `tensor([[[...]]])` |

**Why GPU?**
- CPU: ~8-16 powerful cores, sequential processing
- GPU (NVIDIA A10G): **9,216 CUDA cores**, parallel processing

**CUDA Cores:** Individual processing units that execute one arithmetic operation at a time. PyTorch ships tensors to CUDA cores via `tensor.to("cuda")`.

```python
import torch

x = torch.randn(4096, 4096)
if torch.cuda.is_available():
    x = x.to("cuda")
```

**CUDA vs PyTorch:**
- **CUDA** is NVIDIA's parallel computing platform/API — the hardware abstraction layer
- **PyTorch** is the Python framework that packages neural network math specifically for CUDA core execution
- PyTorch's C++ backend (ATen) calls the CUDA driver, which schedules work across thousands of cores

---

### Level 2: Autograd (Automatic Differentiation)

**The Problem:** Training requires gradients (derivatives) of the loss with respect to every parameter. Doing this manually for 8 billion parameters is impossible.

**The Solution:** Autograd builds a **Dynamic Computation Graph** (a Directed Acyclic Graph) during the Forward Pass:

- Each mathematical operation is recorded as a **node**
- Each node stores: operation type, input tensors, and the **gradient function** (the derivative)
- When `loss.backward()` is called, Autograd traverses the graph **in reverse**, applying the **Chain Rule** at each node
- Each parameter receives its `.grad` attribute with the exact direction and magnitude for weight updates

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1   # Forward pass: builds graph
y.backward()              # Backward pass: computes gradients
print(x.grad)             # dy/dx = 2x + 3 = 7.0
```

**[CERTIFICATION FOCUS]**
- **Training:** Autograd enabled → Computation Graph built → 2-3x VRAM consumption
- **Inference:** Autograd disabled via `torch.no_grad()` → No graph built → VRAM reserved for KV Cache

---

### Level 3: nn.Module Architecture

**The Problem:** Managing 8 billion parameters manually is impossible. We need a standardized blueprint.

**The Solution:** `nn.Module` is a base class providing a blueprint for building neural networks.

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4096, 4096)
        self.dropout = nn.Dropout(0.1)
        self.layer2 = nn.Linear(4096, 32000)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
```

| PyTorch Concept | Real-World Analogy |
|---|---|
| `nn.Module` | Architectural Blueprint |
| `__init__` | Materials list (lumber, wiring, drywall) |
| `forward` | Construction sequence (frame → wire → drywall) |
| `.parameters()` | Inventory of all materials used |
| `.to("cuda")` | Moving construction to a specialized facility |

**How Llama-3 Uses nn.Module (Simplified):**

```python
class LlamaForCausalLM(nn.Module):
    def __init__(self):
        self.model = LlamaModel()          # 32 Decoder layers
        self.lm_head = nn.Linear(4096, 32000)

    def forward(self, input_ids):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states)
        return logits
```

---

## 7. GPU Validation Script

Run this inside a SageMaker PyTorch notebook to verify CUDA availability:

```python
import torch

def check_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\n--- GPU {i} ---")
            print(f"Name: {props.name}")
            print(f"Compute Capability: {props.major}.{props.minor}")
            print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"Multi-Processor Count: {props.multi_processor_count}")

check_cuda()

# Expected output:
# CUDA available: True
# GPU 0: NVIDIA A10G
# Total Memory: 23.65 GB
```

---

## 8. inference.py — Line-by-Line Breakdown

The script answers one question: **"Given a text prompt, how do I get a response from Llama-3 8B on a GPU?"**

Flow: configure memory → load model → set inference mode → tokenize input → generate output → decode output.

---

### Imports

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
```

| Import | What it is |
|---|---|
| `torch` | PyTorch — handles all tensor math and GPU communication |
| `AutoTokenizer` | Loads the correct tokenizer for any Hugging Face model automatically |
| `AutoModelForCausalLM` | Loads a Causal Language Model (text generation) |
| `BitsAndBytesConfig` | Configuration for 4-bit quantization via the `bitsandbytes` library |

---

### Step 1 — Configure 4-bit Quantization

```python
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)
```

Llama-3 8B at FP16 requires ~16 GB VRAM. This config compresses the weights to ~4 GB.

| Parameter | What it does |
|---|---|
| `load_in_4bit=True` | Stores model weights as 4-bit integers instead of 16-bit floats |
| `bnb_4bit_compute_dtype=torch.bfloat16` | Expands weights back to BF16 during math operations (better numerical stability than FP16 on Ampere GPUs) |
| `bnb_4bit_quant_type="nf4"` | NormalFloat4 — designed for normally-distributed neural network weights, less error than generic FP4 |
| `bnb_4bit_use_double_quant=True` | Applies a second quantization on the quantization constants, saving ~0.4 extra bits per parameter |

---

### Step 2 — Load Tokenizer and Model

```python
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)
```

The **tokenizer** converts text into Token IDs (integers) because the model only understands numbers, and converts them back when decoding:

```
"Hello world"  →  tokenizer  →  [15043, 3186]
```

| Parameter | What it does |
|---|---|
| `model_name` | Hugging Face Hub identifier — downloads the model on first run |
| `quantization_config` | Applies the 4-bit compression from Step 1 during loading |
| `device_map="auto"` | Automatically distributes model layers across available GPUs — no manual `.to("cuda")` needed |

---

### Step 3 — Set to Evaluation Mode

```python
model.eval()
```

| Mode | When used | Behavior |
|---|---|---|
| `model.train()` | Training | Dropout randomly deactivates neurons; BatchNorm updates its statistics |
| `model.eval()` | Inference | Dropout disabled; BatchNorm frozen |

**Dropout** is a training technique that randomly silences a percentage of neurons to prevent overfitting. During inference it must be off — otherwise outputs are random and inconsistent on every call.

---

### Step 4 — Prepare Input

```python
prompt = "Explain the concept of Self-Attention in Transformers."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
```

| Part | What it does |
|---|---|
| `tokenizer(prompt, ...)` | Converts the string into Token IDs |
| `return_tensors="pt"` | Returns PyTorch tensors (not Python lists or NumPy arrays) |
| `.to("cuda")` | Moves input tensors from CPU RAM to GPU VRAM — model and inputs must be on the same device |

Result: `{"input_ids": tensor([[128000, 849, 21435, ...]], device='cuda:0')}`

---

### Step 5 — Run Inference

```python
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
```

**`torch.no_grad()`** disables Autograd for everything inside the block. During training, Autograd builds a computation graph to calculate gradients — during inference that graph wastes 2-3x VRAM that should go to the KV Cache.

```
Without torch.no_grad()  →  Computation graph builds  →  2-3x VRAM wasted  →  OOM
With torch.no_grad()     →  No graph                  →  Minimal VRAM usage
```

| Parameter | What it does |
|---|---|
| `**inputs` | Unpacks the `input_ids` dict as keyword arguments |
| `max_new_tokens=200` | Maximum tokens to generate (prompt tokens excluded) |
| `temperature=0.7` | Randomness control: `< 1.0` = more focused, `> 1.0` = more creative, `1.0` = neutral |
| `do_sample=True` | Enables sampling; required when `temperature != 1.0` (otherwise use greedy decoding) |

**What `generate()` does internally:**

```
input tokens → 32 Decoder layers → logits (scores for 32,000 vocab tokens)
→ softmax + temperature → sample next token → append to input → repeat
```

Loop repeats until `max_new_tokens` is reached or the model emits a stop token.

---

### Step 6 — Decode and Print Response

```python
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

| Part | What it does |
|---|---|
| `outputs[0]` | Selects the first sequence from the batch |
| `tokenizer.decode(...)` | Converts Token IDs back into a human-readable string |
| `skip_special_tokens=True` | Strips control tokens like `<\|begin_of_text\|>` and `<\|end_of_text\|>` |

---

### Complete Execution Flow

```
Text prompt
    ↓  tokenizer()
Token IDs tensor on CUDA
    ↓  model.generate()
       ↳ 32 Decoder layers (Self-Attention + FFN) × N tokens
Output Token IDs tensor
    ↓  tokenizer.decode()
Text response
```

### What Breaks If You Skip Each Critical Step

| Step | Consequence |
|---|---|
| `model.eval()` | Dropout randomly degrades outputs — non-deterministic results every run |
| `torch.no_grad()` | Autograd fills VRAM with computation graph → OOM during generation |
| `.to("cuda")` | Inputs on CPU, model on GPU → device mismatch runtime error |

---

## 9. Key Certification Takeaways (MLA-C01)

1. **Transformer Architecture:** Three types (Encoder, Decoder, Encoder-Decoder) and when to use each
2. **Memory Math:** `parameters × bytes_per_param = VRAM required`
3. **Quantization:** NF4 reduces memory 4x vs FP16 — primary cost optimization strategy
4. **RAG vs Fine-Tuning:** RAG for facts, Fine-Tuning for behavior/tone
5. **Autograd:** Training requires Computation Graph (more VRAM). Inference requires `torch.no_grad()`
6. **nn.Module:** `model.eval()` disables Dropout — mandatory before inference
7. **Service Quotas:** GPU quotas are 0 by default on new accounts — request early
8. **AWS Budgets:** Configure billing alerts to prevent runaway costs
9. **Instance Selection:** Match GPU VRAM to model size + KV Cache requirements

### MLA-C01 Exam Alignment (Week 1)

| Task | Domain | Day |
|---|---|---|
| Data ingestion and formatting | Domain 1 (28%) | Tue — JSONL dataset |
| Tokenization and embeddings | Domain 1 (28%) | Tue — Dataset prep |
| Model selection (Decoder-Only) | Domain 2 (26%) | Wed — Prompt baselines |
| Training and fine-tuning | Domain 2 (26%) | Thu — LoRA PEFT |
| Model evaluation and optimization | Domain 2 (26%) | Fri — Quantization benchmarks |
| Cost optimization | Domain 4 (24%) | Fri — Quantization for cost reduction |

---

## 10. Glossary

| Term | Definition |
|---|---|
| **Attention Score** | Weight measuring relevance between tokens |
| **Autograd** | PyTorch's automatic differentiation engine |
| **CUDA** | NVIDIA's parallel computing platform for GPU programming |
| **CUDA Core** | Individual parallel processor inside an NVIDIA GPU |
| **Decoder-Only** | Architecture for auto-regressive text generation |
| **DAG** | Directed Acyclic Graph (the Computation Graph structure) |
| **KV Cache** | Stores previous token keys/values to avoid recalculation |
| **LoRA** | Low-Rank Adaptation — PEFT method for fine-tuning |
| **NF4** | 4-bit NormalFloat quantization format |
| **OOM** | Out of Memory error (CUDA out of memory) |
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **Quantization** | Reducing precision of model weights to save memory |
| **RAG** | Retrieval-Augmented Generation |
| **Self-Attention** | Mechanism allowing tokens to attend to all other tokens |
| **Tensor** | Multi-dimensional array (fundamental PyTorch data structure) |
| **Token** | Unit of text (word piece, subword) |
| **Tokenizer** | Converts text to Token IDs and vice versa |
| **VRAM** | Video RAM — GPU memory |

---

*Study reference for AWS MLA-C01 exam preparation.*
