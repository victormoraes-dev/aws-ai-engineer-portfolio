# Week 1, Day 1 — Transformer Architecture & Self-Attention

*Part of a 4-week accelerated program to build a production-grade AI engineering portfolio and earn the AWS Certified Machine Learning Engineer - Associate (MLA-C01) certification.*

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [The Problem With Older Models](#2-the-problem-with-older-models)
3. [The Transformer Solution](#3-the-transformer-solution)
4. [How Inference Actually Works](#4-how-inference-actually-works)
5. [The Hardware Reality: VRAM & Quantization](#5-the-hardware-reality-vram--quantization)
6. [PyTorch: From Tensors to Neural Networks](#6-pytorch-from-tensors-to-neural-networks)
7. [The Inference Script: Line by Line](#7-the-inference-script-line-by-line)
8. [Glossary](#8-glossary)

---

## 1. What Was Built

On the first day, the goal was to go from zero to a working Llama-3 8B inference script running on a GPU — and to understand every line of code well enough to explain it to someone else.

The deliverables were:
- A repository with a structured project layout
- A GPU validation notebook confirming CUDA availability on the target hardware
- An `inference.py` script that loads Llama-3 8B with 4-bit quantization and generates a response to a text prompt

Before writing any code, the foundational theory was studied: why Transformers replaced RNNs, how Self-Attention works, and what actually happens inside the model when a prompt is submitted.

**Week 1 at a glance:**

| Day | Focus | Deliverable |
|---|---|---|
| Monday | Transformer theory, PyTorch setup, basic inference | Repository, notebook, GPU validation |
| Tuesday | Data preparation, tokenization, embeddings | JSONL dataset for fine-tuning |
| Wednesday | Prompt engineering baseline | Zero-shot vs Few-shot evaluation |
| Thursday | PEFT/LoRA fine-tuning | Trained LoRA adapters |
| Friday | 4-bit quantization optimization | Quantized model, cost analysis |

---

## 2. The Problem With Older Models

Before Transformers, the dominant architectures for language tasks were RNNs (Recurrent Neural Networks) and LSTMs (Long Short-Term Memory networks). Both process text **sequentially** — one word at a time, left to right.

This created two fundamental problems:

1. **No parallelization.** Because each word depended on the previous one, the computation could not be distributed across GPU cores. Training was slow and expensive.
2. **Vanishing gradients over long sequences.** Information from early tokens degraded as the sequence grew longer, making it hard for the model to connect a pronoun at position 200 to the noun it referred to at position 5.

---

## 3. The Transformer Solution

The Transformer architecture, introduced in *"Attention Is All You Need"* (Vaswani et al., 2017), solved both problems with a single mechanism: **Self-Attention**.

Instead of processing tokens one by one, the Transformer processes the **entire sequence simultaneously**. Every token calculates an attention score against every other token at the same time — enabling full GPU parallelization and capturing long-range dependencies regardless of distance.

### The Three Transformer Architectures

| Architecture | Example | Use Case |
|---|---|---|
| **Encoder-Only** | BERT | Classification, Sentiment Analysis, NER |
| **Decoder-Only** | Llama-3, GPT-4 | Auto-regressive text generation, Chatbots |
| **Encoder-Decoder** | T5, BART | Translation, Summarization |

### Why Llama-3 Uses Decoder-Only

Llama-3's sole objective is **auto-regressive generation** — predicting the next token given all previous tokens. The Decoder has its own Masked Self-Attention mechanism to understand the prompt, making a separate Encoder unnecessary. Each token can only attend to itself and previous tokens, enforced by a triangular mask that sets future positions to `-infinity` before softmax.

---

## 4. How Inference Actually Works

The lifecycle of a single prompt through Llama-3 8B:

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

## 5. The Hardware Reality: VRAM & Quantization

One of the first practical challenges was understanding why a model with 8 billion parameters doesn't just "run" on any machine.

### Memory Requirements for Llama-3 8B

| Precision | Bits per Parameter | 8B Model VRAM |
|---|---|---|
| FP32 (Full) | 32 bits = 4 bytes | ~32 GB |
| FP16 / BF16 | 16 bits = 2 bytes | ~16 GB |
| INT4 (4-bit) | 4 bits = 0.5 bytes | ~4 GB |

### The KV Cache Problem

The **KV Cache** stores the mathematical representations of all previous tokens to avoid recalculating them on each generation step. It grows linearly with context length:

```
KV Cache Size ≈ 2 × num_layers × hidden_dim × seq_len × precision_bytes
```

For Llama-3 8B (32 layers, 4096 hidden dim):
- 4K context → ~2 GB at FP16
- 32K context → ~16 GB at FP16

This is the primary cause of CUDA Out of Memory (OOM) errors during long conversations.

### Solution: 4-bit Quantization (NF4)

Using the `bitsandbytes` library, model weights are compressed from ~16 GB to ~4 GB — freeing ~12 GB of VRAM for the KV Cache and enabling long context windows on accessible hardware.

### Target Hardware

| Instance | GPU | VRAM | CUDA Cores | Use Case |
|---|---|---|---|---|
| `ml.g5.2xlarge` | NVIDIA A10G | 24 GB | 9,216 | Fine-tuning 7B-13B models |

---

## 6. PyTorch: From Tensors to Neural Networks

Three PyTorch concepts were studied in depth because they appear in every training and inference script.

### Tensors & CUDA

PyTorch processes **Tensors** — multi-dimensional matrices that can live on a GPU.

| Type | Dimension | Example |
|---|---|---|
| Scalar | 0D | `tensor(5)` |
| Vector | 1D | `tensor([1, 2, 3])` |
| Matrix | 2D | `tensor([[1,2],[3,4]])` |
| Tensor | 3D+ | `tensor([[[...]]])` |

The distinction between CUDA and PyTorch matters: **CUDA** is NVIDIA's parallel computing platform — the hardware abstraction layer. **PyTorch** is the Python framework that packages neural network math specifically for CUDA core execution. PyTorch's C++ backend (ATen) calls the CUDA driver, which schedules work across thousands of cores.

```python
import torch

x = torch.randn(4096, 4096)
if torch.cuda.is_available():
    x = x.to("cuda")
```

### Autograd (Automatic Differentiation)

Training requires gradients — derivatives of the loss with respect to every parameter. Doing this manually for 8 billion parameters is impossible. Autograd solves this by building a **Dynamic Computation Graph** during the forward pass:

- Each operation is recorded as a node
- When `loss.backward()` is called, Autograd traverses the graph in reverse, applying the Chain Rule at each node
- Each parameter receives its `.grad` attribute with the exact direction and magnitude for weight updates

```python
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1   # Forward pass: builds graph
y.backward()              # Backward pass: computes gradients
print(x.grad)             # dy/dx = 2x + 3 = 7.0
```

During inference, Autograd is disabled via `torch.no_grad()` — no graph is built, and VRAM is reserved for the KV Cache instead.

### nn.Module Architecture

`nn.Module` is PyTorch's base class for building neural networks — a standardized blueprint for managing billions of parameters.

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

---

## 7. The Inference Script: Line by Line

The script answers one question: **"Given a text prompt, how do I get a response from Llama-3 8B on a GPU?"**

Flow: configure memory → load model → set inference mode → tokenize input → generate output → decode output.

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
| `bnb_4bit_compute_dtype=torch.bfloat16` | Expands weights back to BF16 during math operations for numerical stability |
| `bnb_4bit_quant_type="nf4"` | NormalFloat4 — designed for normally-distributed neural network weights |
| `bnb_4bit_use_double_quant=True` | Applies a second quantization on the quantization constants, saving ~0.4 extra bits per parameter |

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

The tokenizer converts text into Token IDs because the model only understands numbers:

```
"Hello world"  →  tokenizer  →  [15043, 3186]
```

| Parameter | What it does |
|---|---|
| `model_name` | Hugging Face Hub identifier — downloads the model on first run |
| `quantization_config` | Applies the 4-bit compression from Step 1 during loading |
| `device_map="auto"` | Automatically distributes model layers across available GPUs |

### Step 3 — Set to Evaluation Mode

```python
model.eval()
```

| Mode | When used | Behavior |
|---|---|---|
| `model.train()` | Training | Dropout randomly deactivates neurons; BatchNorm updates its statistics |
| `model.eval()` | Inference | Dropout disabled; BatchNorm frozen |

Dropout is a training technique that randomly silences neurons to prevent overfitting. During inference it must be off — otherwise outputs are non-deterministic on every call.

### Step 4 — Prepare Input

```python
prompt = "Explain the concept of Self-Attention in Transformers."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
```

| Part | What it does |
|---|---|
| `tokenizer(prompt, ...)` | Converts the string into Token IDs |
| `return_tensors="pt"` | Returns PyTorch tensors |
| `.to("cuda")` | Moves input tensors from CPU RAM to GPU VRAM — model and inputs must be on the same device |

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

`torch.no_grad()` disables Autograd for everything inside the block. Without it, the computation graph fills VRAM and causes OOM errors during generation.

| Parameter | What it does |
|---|---|
| `max_new_tokens=200` | Maximum tokens to generate (prompt tokens excluded) |
| `temperature=0.7` | Randomness control: `< 1.0` = more focused, `> 1.0` = more creative |
| `do_sample=True` | Enables sampling; required when `temperature != 1.0` |

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

## 8. Glossary

| Term | Definition |
|---|---|
| **Attention Score** | Weight measuring relevance between tokens |
| **Autograd** | PyTorch's automatic differentiation engine |
| **CUDA** | NVIDIA's parallel computing platform for GPU programming |
| **CUDA Core** | Individual parallel processor inside an NVIDIA GPU |
| **Decoder-Only** | Architecture for auto-regressive text generation |
| **KV Cache** | Stores previous token keys/values to avoid recalculation |
| **LoRA** | Low-Rank Adaptation — PEFT method for fine-tuning |
| **NF4** | 4-bit NormalFloat quantization format |
| **OOM** | Out of Memory error (CUDA out of memory) |
| **PEFT** | Parameter-Efficient Fine-Tuning |
| **Quantization** | Reducing precision of model weights to save memory |
| **RAG** | Retrieval-Augmented Generation |
| **Self-Attention** | Mechanism allowing tokens to attend to all other tokens simultaneously |
| **Tensor** | Multi-dimensional array (fundamental PyTorch data structure) |
| **Token** | Unit of text (word piece, subword) |
| **Tokenizer** | Converts text to Token IDs and vice versa |
| **VRAM** | Video RAM — GPU memory |
