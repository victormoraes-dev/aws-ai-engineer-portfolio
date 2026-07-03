# AWS AI Engineer Portfolio — Victor Moraes

A 4-week accelerated program building production-grade AI systems while preparing for the **AWS Certified Machine Learning Engineer - Associate (MLA-C01)** certification.

Each project is fully documented with architecture decisions, real benchmark results, and line-by-line code walkthroughs.

---

## Projects

### Week 1 — LLaMA Fine-Tuning Pipeline

End-to-end pipeline for fine-tuning a large language model on a domain-specific legal dataset using AWS SageMaker.

| Day | Topic | What Was Built |
|-----|-------|----------------|
| [Day 1](apps/w01-llama-fine-tuning/w01-d01-transformer-and-self-attention/REVIEW.md) | Transformer Architecture & Self-Attention | Llama-3 8B inference script with 4-bit quantization on GPU |
| [Day 2](apps/w01-llama-fine-tuning/w01-d02-tokenization-and-embeddings/REVIEW.md) | Tokenization & Embeddings | Semantic embedding pipeline with t-SNE visualization and cosine similarity validation |
| [Day 3](apps/w01-llama-fine-tuning/w01-d03-prompt-engineering/REVIEW.md) | Prompt Engineering Baseline | Automated evaluation of 3 prompt strategies across 45 inferences using ROUGE-L and BERTScore |

**Stack:** Python · PyTorch · HuggingFace Transformers · AWS SageMaker · Groq API · sentence-transformers

---

### Week 2 — RAG System & Vector Databases

| Day | Topic | What Was Built |
|-----|-------|----------------|
| [Day 1](apps/w02-rag-system/w02-d01-rag-and-vector-database/REVIEW.md) | RAG Pipeline & Vector Databases | End-to-end document Q&A system: PDF ingestion → chunking → vector indexing → semantic retrieval → LLM answer generation |

**Stack:** Python · LangChain · ChromaDB · HuggingFace Embeddings · Groq API

---

## Infrastructure

| Component | Technology | Purpose |
|-----------|------------|---------|
| [SageMaker Notebook](infrastructure/terraform/sagemaker-notebook-instance/main.tf) | Terraform | `ml.g5.2xlarge` GPU instance (NVIDIA A10G, 24 GB VRAM) for model training |
| [Local LLM Runtime](infrastructure/docker/ollama-local-inference/) | Docker + Ollama | Local inference environment without cloud dependency |

---

## Repository Structure

```
apps/               # Projects — each with code, results, and a detailed REVIEW.md
infrastructure/     # Terraform (AWS) and Docker configurations
libs/               # Reusable AI/ML modules and agent utilities
packages/           # Shared tooling (config, types, UI components)
docs/               # Architecture Decision Records and guides
tools/              # Setup scripts and CLI utilities
```

---

## Key Skills Demonstrated

- **LLM Inference & Optimization** — 4-bit quantization (NF4), KV Cache management, VRAM budgeting
- **Fine-Tuning** — PEFT/LoRA on domain-specific datasets with AWS SageMaker
- **RAG Systems** — Document ingestion, chunking strategy, vector search, retrieval-augmented generation
- **Evaluation** — ROUGE-L, BERTScore, cosine similarity, prompt strategy benchmarking
- **AWS** — SageMaker, Bedrock, IAM, S3 — provisioned with Terraform
- **MLOps** — Reproducible pipelines, artifact versioning, metric-driven iteration

---

## Quick Start

```bash
git clone https://github.com/victormoraes-dev/aws-ai-engineer-portfolio.git
cd aws-ai-engineer-portfolio
./tools/scripts/setup.sh
```

See [docs/guides/CONTRIBUTING.md](docs/guides/CONTRIBUTING.md) for development conventions.

---

## License

[MIT](LICENSE) © Victor Moraes
