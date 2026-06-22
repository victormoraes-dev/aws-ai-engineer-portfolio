# Week 1, Day 1 — Complete Review: Foundations of AI Engineering

*Study reference for the AWS Certified Machine Learning Engineer - Associate (MLA-C01)*

---

## Table of Contents

1. [Course Roadmap Overview](#1-course-roadmap-overview)
2. [Transformer Architecture & Self-Attention](#2-transformer-architecture--self-attention)
3. [The Inference Pipeline](#3-the-inference-pipeline)
4. [GPU VRAM Physics & Quantization](#4-gpu-vram-physics--quantization)
5. [RAG Architecture](#5-rag-architecture-retrieval-augmented-generation)
6. [PyTorch Learning Ladder](#6-pytorch-learning-ladder)
7. [Infrastructure as Code (Terraform)](#7-infrastructure-as-code-terraform)
8. [AWS Service Quotas](#8-aws-service-quotas)
9. [GPU Validation Script](#9-gpu-validation-script)
10. [Git & SSH Configuration](#10-git--ssh-configuration)
11. [Full Inference Code (Ready for GPU)](#11-full-inference-code-ready-for-gpu)
12. [Key Certification Takeaways (MLA-C01)](#12-key-certification-takeaways-mla-c01)
13. [Glossary](#13-glossary)
14. [Daily LinkedIn Post](#14-daily-linkedin-post-build-in-public)

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

### Step-by-Step Execution

**Step 1: Tokenization**
- The Hugging Face `AutoTokenizer` converts raw text into **Token IDs** (integers)
- Llama-3 uses a BPE tokenizer with ~32,000 tokens in vocabulary

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokens = tokenizer("Hello world", return_tensors="pt")
# tokens["input_ids"] = tensor([[15043, 3186]])
```

**Step 2: Embedding Lookup**
- Each Token ID is mapped to a dense vector of dimension **4096**

**Step 3: Decoder Layers**
- The input passes through **32 Transformer Decoder layers**
- Each layer contains: Masked Multi-Head Self-Attention + Feed-Forward Network (SwiGLU) + Residual connections + LayerNorm

**Step 4: Output Projection (LM Head)**
- The final hidden state passes through `nn.Linear(4096, 32000)` producing **logits** — raw scores for each token in the vocabulary

**Step 5: Softmax & Sampling**
- Logits are converted to probabilities via softmax
- The next token is selected (Greedy, Top-K, or Temperature sampling)

**Step 6: Auto-Regressive Loop**
- The predicted token is appended to the input sequence
- Steps 1-5 repeat until `<|end_of_text|>` is generated

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
input_ids = tokenizer("Hello world", return_tensors="pt").input_ids
output = model.generate(input_ids, max_new_tokens=100)
```

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

Uses the `bitsandbytes` library to compress model weights to 4-bit NormalFloat (NF4) precision:

```python
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**Result:** Model compresses from ~16 GB to ~4 GB, freeing **~12 GB of VRAM** for the KV Cache and long context windows.

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
        # __init__: Declare ALL building blocks (layers)
        self.layer1 = nn.Linear(4096, 4096)
        self.dropout = nn.Dropout(0.1)
        self.layer2 = nn.Linear(4096, 32000)

    def forward(self, x):
        # forward: Define the EXACT execution pipeline
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
```

**Analogy:**

| PyTorch Concept | Real-World Analogy |
|---|---|
| `nn.Module` | Architectural Blueprint |
| `__init__` | Materials list (2x4 lumber, wiring, drywall) |
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

**Critical Methods Before Inference:**

```python
# Thing 1: Disable Dropout and BatchNorm training behavior
model.eval()

# Thing 2: Disable Autograd to prevent building Computation Graph
with torch.no_grad():
    output = model.generate(input_ids, max_new_tokens=100)
```

| Method | What It Does | Why It's Needed |
|---|---|---|
| `model.eval()` | Disables Dropout, fixes BatchNorm | Without it, Dropout randomly deactivates neurons → non-deterministic garbage outputs |
| `torch.no_grad()` | Prevents building Computation Graph | Without it, Autograd wastes VRAM that should go to KV Cache → OOM crashes |

---

## 7. Infrastructure as Code (Terraform)

### Repository Structure (Monorepo)

```
ai-engineer-portfolio/
├── infrastructure/
│   ├── main.tf              # Terraform configuration
│   ├── variables.tf          # Input variables
│   └── outputs.tf            # Output values
├── notebooks/
│   ├── inference.ipynb       # PyTorch inference notebook
│   └── fine-tuning.ipynb     # LoRA fine-tuning notebook
├── data/
│   ├── raw/                  # Raw datasets
│   └── processed/            # Tokenized datasets
├── src/
│   ├── train.py              # Training script
│   └── inference.py          # Inference script
├── README.md                 # Project documentation
└── .gitignore
```

### Terraform Configuration

```hcl
provider "aws" {
  region = var.aws_region
}

resource "aws_sagemaker_notebook_instance" "ai_engineer_notebook" {
  name          = "ai-engineer-sprint"
  role_arn      = aws_iam_role.sagemaker_role.arn
  instance_type = var.instance_type  # "ml.g5.2xlarge" or "ml.t3.medium"

  lifecycle_config_name = aws_sagemaker_notebook_instance_lifecycle_configuration.stop_on_idle.name

  tags = {
    Name     = "AI Engineer Sprint"
    Project  = "Week-1-FineTuning"
    Ephemeral = "True"
  }
}
```

### IAM Execution Role

```hcl
resource "aws_iam_role" "sagemaker_role" {
  name = "sagemaker-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "s3_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}
```

### Deploy Commands

```bash
terraform init
terraform plan
terraform apply -auto-approve
terraform destroy  # Destroy after session to save costs
```

### Cost Management (AWS Budgets)

```hcl
resource "aws_budgets_budget" "ml_cost" {
  name              = "ml-engineer-monthly-budget"
  budget_type       = "COST"
  limit_amount      = "100"
  limit_unit        = "USD"
  time_unit         = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = ["your-email@example.com"]
  }
}
```

---

## 8. AWS Service Quotas

### The Problem

AWS restricts GPU instance quotas (`G and VT instance families`) by default. New accounts typically have a quota of **0** for `ml.g5.2xlarge`.

### The Solution

Request a quota increase via **AWS Service Quotas** console or AWS Support with a detailed technical justification.

### Sample Quota Increase Request

```
Subject: Service limit increase request - ml.g5.2xlarge for notebook instance usage

I am preparing for the AWS Certified Machine Learning Engineer - Associate (MLA-C01)
exam and building a professional MLOps portfolio.

1. The Workload:
Fine-tuning Llama-3 8B using 4-bit quantization (NF4 via BitsAndBytes) and
PEFT/LoRA adapters on a custom JSONL dataset.

2. Why ml.g5.2xlarge:
The instance provides a single NVIDIA A10G GPU with 24 GB VRAM. The 4-bit
quantized 8B model consumes ~6 GB VRAM, leaving remaining memory for KV Cache
and LoRA training weights.

3. Cost Control:
Infrastructure managed via Terraform (IaC). Instance is ephemeral — provisioned
for 2-4 hour daily sessions and destroyed after. AWS Budgets and billing alarms
are configured.
```

### GPU Instance Comparison for MLA-C01

| Instance | GPU | CUDA Cores | VRAM | Best For |
|---|---|---|---|---|
| `ml.g5.2xlarge` | NVIDIA A10G | 9,216 | 24 GB | Fine-tuning 7B-13B |
| `ml.p3.2xlarge` | NVIDIA V100 | 5,120 | 16 GB | Older gen, cheaper |
| `ml.p4d.24xlarge` | NVIDIA A100 | 6,912 | 40 GB | Large-scale training |
| `ml.p5.48xlarge` | NVIDIA H100 | 18,432 | 80 GB | Enterprise foundation models |

---

## 9. GPU Validation Script

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

## 10. Git & SSH Configuration

### Repository Initialization

```bash
git init
git add .
git commit -m "Initial commit: Week 1 Day 1 - Foundations"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/ai-engineer-portfolio.git
git push -u origin main
```

### SSH Key Configuration (Multiple GitHub Accounts)

```bash
# ~/.ssh/config
Host github.com-ai-engineer
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_ai_engineer
    IdentitiesOnly yes

# Clone with custom host
git clone git@github.com-ai-engineer:YOUR_USERNAME/ai-engineer-portfolio.git
```

---

## 11. Full Inference Code (Ready for GPU)

When the GPU quota is approved, run this complete script:

```python
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Step 1: Configure 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# Step 2: Load tokenizer and model
model_name = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"
)

# Step 3: Set to evaluation mode (CRITICAL!)
model.eval()

# Step 4: Prepare input
prompt = "Explain the concept of Self-Attention in Transformers."
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Step 5: Run inference (CRITICAL: disable Autograd)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )

# Step 6: Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

---

## 12. Key Certification Takeaways (MLA-C01)

### Must-Know Concepts

1. **Transformer Architecture:** Three types (Encoder, Decoder, Encoder-Decoder) and when to use each
2. **Memory Math:** `parameters × bytes_per_param = VRAM required`
3. **Quantization:** NF4 reduces memory 4x vs FP16. Cost optimization strategy
4. **RAG vs Fine-Tuning:** RAG for facts, Fine-Tuning for behavior/tone
5. **Infrastructure as Code:** Terraform/CloudFormation for ML infra. Tagging for cost allocation
6. **Autograd:** Training requires Computation Graph (more VRAM). Inference requires `torch.no_grad()`
7. **nn.Module:** `model.eval()` disables Dropout. Mandatory before inference
8. **Service Quotas:** GPU quotas restricted by default. Plan ahead
9. **AWS Budgets:** Configure billing alerts to prevent runaway costs
10. **Instance Selection:** Match GPU specs to model size and workload type

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

## 13. Glossary

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

*Document generated for study and review purposes. AWS MLA-C01 exam preparation.*