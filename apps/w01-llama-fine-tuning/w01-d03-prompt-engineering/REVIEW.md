# Week 1, Day 3 — Prompt Engineering Baseline

*Part of a 4-week accelerated program to build a production-grade AI engineering portfolio and earn the AWS Certified Machine Learning Engineer - Associate (MLA-C01) certification.*

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [The Problem: No Baseline, No Proof of Improvement](#2-the-problem-no-baseline-no-proof-of-improvement)
3. [The Infrastructure Pivot](#3-the-infrastructure-pivot)
4. [The Evaluation Dataset](#4-the-evaluation-dataset)
5. [The 3 Prompt Strategies](#5-the-3-prompt-strategies)
6. [How the Evaluation Was Measured](#6-how-the-evaluation-was-measured)
7. [Results](#7-results)
8. [The Evaluation Notebook](#8-the-evaluation-notebook)

---

## 1. What Was Built

Day 3 was about establishing a rigorous, metric-driven performance baseline before any fine-tuning happened. The target model was `meta.llama3-1-8b-instruct-v1:0` — the same Llama-3.1 8B used throughout the week.

Three prompt engineering strategies were tested against 15 domain-specific legal questions across 3 categories. Every response was scored automatically using two complementary metrics: ROUGE-L for structural overlap and BERTScore F1 for semantic similarity.

The deliverables were:
- `baseline_raw_outputs.csv` — all 45 model responses (15 questions × 3 strategies) side by side
- `baseline_summary.json` — aggregated scores, latency, token counts, and the LoRA opportunity gap
- A per-category breakdown identifying which legal domain benefits most from fine-tuning

---

## 2. The Problem: No Baseline, No Proof of Improvement

Fine-tuning a model without a baseline is like claiming a drug works without a control group. If Day 4's LoRA-trained model scores higher than the raw model, that improvement needs to be quantifiable — not just a subjective impression.

The baseline serves two purposes:
1. It establishes the current performance ceiling of prompt engineering alone
2. It quantifies the **LoRA opportunity** — the gap between zero-shot and few-shot performance that fine-tuning should match or exceed

---

## 3. The Infrastructure Pivot

The initial plan used AWS Bedrock's Converse API. The account had a cross-region inference quota of 0, which is not adjustable for new accounts.

Rather than waiting for a quota increase, the evaluation was migrated to **Groq's free API** (`llama-3.1-8b-instant`). Groq provides equivalent Llama 3.1 8B inference with a 30,000 tokens/min free tier — no credit card required. The model behavior is identical; only the API endpoint changed.

---

## 4. The Evaluation Dataset

| Category | Count | Focus |
|----------|-------|-------|
| **Contracts** | 6 | Contract interpretation, breach, enforceability, indemnification, non-compete, limitation of liability |
| **NDAs** | 4 | Confidentiality obligations, enforcement, duration, data processing agreements |
| **Intellectual Property** | 5 | Patents vs trade secrets, copyright registration, trademark infringement, open-source licensing, international branding |

Each question has a **ground-truth expected answer** written by a domain expert (1–3 explanatory sentences). The three categories enable per-domain performance analysis — detecting where prompt engineering helps most and where fine-tuning is required.

---

## 5. The 3 Prompt Strategies

### Strategy A — Zero-Shot Direct

Minimal instruction, no examples. Tests the model's raw domain knowledge without any structural guidance.

```
You are a legal assistant specializing in corporate law. Answer the following question concisely and accurately.

Question: {question}
Answer:
```

### Strategy B — Zero-Shot with Role

Adds a persona (senior corporate attorney) and explicit structure constraints (1–3 sentences). Tests whether role priming improves output quality and structural discipline.

```
You are a senior corporate attorney at a top-tier law firm. Provide a precise, technically accurate answer to the following legal question. Structure your answer in 1-3 sentences.

Question: {question}
Answer:
```

### Strategy C — Few-Shot (2 examples + target question)

Provides 2 in-domain examples before the target question. Tests the model's in-context learning capability — whether seeing examples of the expected format improves output.

```
You are a legal assistant specializing in corporate law. Answer legal questions concisely and accurately.

Example 1:
Q: What is consideration in contract law?
A: Consideration is something of value exchanged between parties to a contract...

Example 2:
Q: Can an employer terminate an at-will employee without cause?
A: Under at-will employment, either party may terminate the relationship at any time without cause...

Now answer the following:
Q: {question}
A:
```

The performance gap between Strategy A and Strategy C directly quantifies the LoRA opportunity — how much improvement in-context learning provides, which fine-tuning should match or exceed.

---

## 6. How the Evaluation Was Measured

### Metrics

| Metric | Role | What It Measures |
|--------|------|-----------------|
| **ROUGE-L** | Primary | Longest common subsequence — structural overlap between prediction and expected answer |
| **ROUGE-1** | Secondary | Unigram (single word) overlap — vocabulary alignment |
| **ROUGE-2** | Secondary | Bigram (word pair) overlap — local phrasing fluency |
| **BERTScore F1** | Secondary | Semantic similarity via embedding cosine similarity — captures paraphrases that ROUGE misses |
| **Latency** | Operational | Wall-clock time per inference call |
| **Token Consumption** | Cost | Output token count per response |

### The ROUGE vs BERTScore Distinction

```
ROUGE-L:   "Did you use the right words in the right ORDER?"
           Penalizes: reordering, missing key terms
           Best for: structural precision

BERTScore: "Did you capture the MEANING, even with different words?"
           Penalizes: semantic drift, hallucination
           Best for: meaning preservation
```

A high BERTScore with low ROUGE-L indicates the model **understands the concept** but **cannot express it with the correct legal structure** — a clear fine-tuning target.

### Pipeline Flow

```text
Load dataset (15 questions × 3 categories)
    ↓
For each question:
    Run Strategy A → generate prediction
    Run Strategy B → generate prediction
    Run Strategy C → generate prediction
    ↓
Score each prediction against ground-truth expected answer
    - ROUGE-1, ROUGE-2, ROUGE-L
    - BERTScore Precision, Recall, F1
    ↓
Aggregate results by strategy and by legal category
    ↓
Export full results matrix to CSV
    ↓
Compute Gap A → C (LoRA opportunity metric)
```

---

## 7. Results

### Overall Scores

| Strategy | ROUGE-L | ROUGE-1 | ROUGE-2 | BERTScore F1 | Latency (s) | Tokens |
|----------|---------|---------|---------|-------------|-------------|--------|
| **A — Zero-Shot Direct** | 0.1447 | 0.1913 | 0.0730 | 0.8611 | 1.10 | 198.0 |
| **B — Zero-Shot Role** | 0.2391 | 0.2811 | 0.1230 | 0.8873 | 1.43 | 96.4 |
| **C — Few-Shot** | 0.1986 | 0.2441 | 0.0982 | 0.8703 | 1.55 | 133.5 |

**Gap A → C (LoRA opportunity):** ROUGE-L: **+0.0539** | ROUGE-1: **+0.0527** | ROUGE-2: **+0.0252**

### Per-Category ROUGE-L Breakdown

| Category | Strategy A | Strategy B | Strategy C | Gap A → C |
|----------|-----------|-----------|-----------|-----------| 
| **Contracts** | 0.1862 | 0.2765 | 0.2088 | **+0.0226** |
| **NDAs** | 0.1045 | 0.1849 | 0.1893 | **+0.0848** |
| **Intellectual Property** | 0.1268 | 0.2372 | 0.1944 | **+0.0676** |

### BERTScore Breakdown

| Strategy | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| **A — Zero-Shot Direct** | 0.8277 | 0.8975 | 0.8611 |
| **B — Zero-Shot Role** | 0.8627 | 0.9137 | 0.8873 |
| **C — Few-Shot** | 0.8457 | 0.8969 | 0.8703 |

### Key Findings

1. **Strategy B (Role-primed) outperformed both A and C** across all metrics. The role persona combined with a hard length constraint (1–3 sentences) produced the highest structural quality (ROUGE-L 0.2391) and semantic quality (BERTScore 0.8873).

2. **Strategy C (Few-shot) underperformed B** because the 2 examples were from different legal subdomains than the target questions. Few-shot works best when examples are semantically similar to the target.

3. **High BERTScore, low ROUGE-L across all strategies** reveals that the model understands legal concepts but cannot express them with correct legal terminology and structure. This is the primary target for LoRA fine-tuning.

4. **NDA category has the largest gap** (+0.0848), indicating the highest ROI for fine-tuning — NDA answers require precise, formulaic structures that the raw model struggles with.

5. **LoRA fine-tuning strategy:** The weights should encode the structural discipline of Strategy B (concise, technically precise, 1–3 sentence answers) combined with the format awareness of Strategy C (Q/A pattern with legal terminology), with the training dataset weighted 3x toward NDA examples.

---

## 8. The Evaluation Notebook

**Notebook:** `code/metrics_baseline.ipynb`

**Dependencies:** `groq`, `evaluate` (HuggingFace), `rouge_score`, `bert-score`, `python-dotenv`

The notebook is structured in 10 cells, each with a single responsibility.

### Cell 1 — Configuration

```python
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_ID = "llama-3.1-8b-instant"
TEMPERATURE = 0.1
MAX_TOKENS = 256
```

`TEMPERATURE = 0.1` minimizes randomness — at near-zero temperature the model becomes near-deterministic, which is essential for reproducible metric comparisons across runs. `MAX_TOKENS = 256` caps output length so latency and token cost stay consistent across strategies.

### Cell 2 — Dataset

```python
DATASET = [
    {
        "id": "c01",
        "category": "Contracts",
        "question": "What are the essential elements of a valid contract?",
        "expected": "A valid contract generally requires offer, acceptance, consideration..."
    },
    # ... 14 more items
]
```

Each item carries four keys: `id` (unique identifier for CSV traceability), `category` (enables per-domain breakdown), `question` (the prompt input), and `expected` (the ground-truth reference answer that ROUGE and BERTScore compare against).

### Cell 3 — Prompt Templates

```python
def strategy_a(question: str) -> str:
    """Zero-Shot Direct — minimal instruction, no examples."""
    return f"""You are a legal assistant specializing in corporate law. 
        Answer the following question concisely and accurately.
        Question: {question}
        Answer:"""

def strategy_b(question: str) -> str:
    """Zero-Shot with Role — persona + structure constraints."""
    return f"""You are a senior corporate attorney at a top-tier law firm.
        Provide a precise, technically accurate answer to the following legal question. 
        Structure your answer in 1-3 sentences.
        Question: {question}
        Answer:"""

def strategy_c(question: str) -> str:
    """Few-Shot — 2 in-domain examples + target question."""
    return f"""You are a legal assistant specializing in corporate law.
        Answer legal questions concisely and accurately.
        
        Example 1:
        Q: What is consideration in contract law?
        A: Consideration is something of value exchanged between parties to a contract...
        
        Example 2:
        Q: Can an employer terminate an at-will employee without cause?
        A: Under at-will employment, either party may terminate the relationship...
        
        Now answer the following:
        Q: {question}
        A:"""

STRATEGIES = {
    "A - Zero-Shot Direct": strategy_a,
    "B - Zero-Shot Role": strategy_b,
    "C - Few-Shot": strategy_c,
}
```

Each strategy is a plain function that takes a question string and returns a complete prompt. The `STRATEGIES` dict maps human-readable names to functions, allowing the inference loop to iterate over all three without duplication.

### Cell 4 — Inference Engine

```python
import time

def query_groq(prompt: str) -> dict:
    start = time.time()

    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    latency = time.time() - start

    return {
        "output": response.choices[0].message.content,
        "latency_s": round(latency, 2),
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
    }
```

`time.time()` is called before and after the API request to capture wall-clock latency per inference call. The return dict preserves both the model output text and all token counts from `response.usage`, which are needed for cost analysis.

### Cell 5 — Run Full Baseline (45 Inferences)

```python
results = []

for item in DATASET:
    row = {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "expected": item["expected"],
    }

    for strategy_name, prompt_fn in STRATEGIES.items():
        prompt = prompt_fn(item["question"])
        result = query_groq(prompt)

        row[f"{strategy_name}_output"] = result["output"]
        row[f"{strategy_name}_latency"] = result["latency_s"]
        row[f"{strategy_name}_tokens"] = result["output_tokens"]

        time.sleep(0.3)  # Groq rate limit buffer

    results.append(row)
```

The outer loop iterates over the 15 dataset items; the inner loop runs all 3 strategies for each question. Each `row` holds the question metadata plus all 3 strategy outputs side-by-side — a wide format that makes ROUGE computation and CSV export straightforward. `time.sleep(0.3)` adds a 300ms delay between API calls to stay within Groq's free-tier rate limit.

### Cell 6 — Export Raw Outputs to CSV

```python
import csv

csv_path = "baseline_raw_outputs.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
```

This produces a 15-row CSV where each row is one question and each strategy's output is a separate column, enabling manual side-by-side inspection of model outputs before automated scoring.

### Cell 7 — Automated Scoring with ROUGE

```python
import evaluate

rouge = evaluate.load("rouge")

for strategy_name in STRATEGIES:
    predictions = [r[f"{strategy_name}_output"] for r in results]
    references = [r["expected"] for r in results]

    scores = rouge.compute(predictions=predictions, references=references)

    print(f"  ROUGE-1: {scores['rouge1']:.4f}")
    print(f"  ROUGE-2: {scores['rouge2']:.4f}")
    print(f"  ROUGE-L: {scores['rougeL']:.4f}")

    for cat in ["Contracts", "NDAs", "Intellectual Property"]:
        cat_items = [r for r in results if r["category"] == cat]
        cat_predictions = [r[f"{strategy_name}_output"] for r in cat_items]
        cat_refs = [r["expected"] for r in cat_items]
        cat_scores = rouge.compute(predictions=cat_predictions, references=cat_refs)
        print(f"    {cat:25s}: ROUGE-L = {cat_scores['rougeL']:.4f}")
```

The inner category loop re-filters `results` to only rows matching each category, then computes ROUGE-L on that subset — this is the per-domain breakdown that reveals which legal category benefits most from fine-tuning.

### Cell 8 — Summary Table

```python
summary = {}
for strategy_name in STRATEGIES:
    predictions = [r[f"{strategy_name}_output"] for r in results]
    references = [r["expected"] for r in results]
    scores = rouge.compute(predictions=predictions, references=references)

    avg_latency = sum(r[f"{strategy_name}_latency"] for r in results) / len(results)
    avg_tokens = sum(r[f"{strategy_name}_tokens"] for r in results) / len(results)

    summary[strategy_name] = {
        "rouge_l": scores["rougeL"],
        "rouge_1": scores["rouge1"],
        "rouge_2": scores["rouge2"],
        "avg_latency_s": round(avg_latency, 2),
        "avg_output_tokens": round(avg_tokens, 1),
    }

# Gap A -> C — the LoRA opportunity metric
gap_rouge_l = summary["C - Few-Shot"]["rouge_l"] - summary["A - Zero-Shot Direct"]["rouge_l"]
gap_rouge_1 = summary["C - Few-Shot"]["rouge_1"] - summary["A - Zero-Shot Direct"]["rouge_1"]
gap_rouge_2 = summary["C - Few-Shot"]["rouge_2"] - summary["A - Zero-Shot Direct"]["rouge_2"]
```

The Gap A → C calculation quantifies how much improvement in-context learning provides over zero-shot — the minimum target LoRA fine-tuning must surpass to justify the training cost.

### Cell 9 — Save Summary to JSON

```python
import datetime, json

output = {
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_ID,
    "temperature": TEMPERATURE,
    "max_tokens": MAX_TOKENS,
    "dataset_size": len(DATASET),
    "strategies": summary,
    "gaps": {
        "a_to_c_rouge_l": round(gap_rouge_l, 4),
        "a_to_c_rouge_1": round(gap_rouge_1, 4),
        "a_to_c_rouge_2": round(gap_rouge_2, 4),
    },
}

with open("baseline_summary.json", "w") as f:
    json.dump(output, f, indent=2)
```

All inference parameters are embedded in the JSON alongside the results — making the file self-describing and reproducible. The `gaps` block serializes the LoRA opportunity metric so downstream fine-tuning scripts can load this file and compare against the baseline without re-running inference.

### Cell 10 — BERTScore

```python
from bert_score import score as bertscore

for strat_name in STRATEGIES:
    predictions = [r[f"{strat_name}_output"] for r in results]
    references = [r["expected"] for r in results]
    
    P, R, F1 = bertscore(predictions, references, lang="en", verbose=False)
    
    print(f"  BERTScore Precision: {P.mean().item():.4f}")
    print(f"  BERTScore Recall:    {R.mean().item():.4f}")
    print(f"  BERTScore F1:        {F1.mean().item():.4f}")
```

`bertscore()` loads `roberta-large` and computes token-level cosine similarities between contextual embeddings of predictions and references. Unlike ROUGE, BERTScore captures semantic equivalence even when different words are used — the complement metric that distinguishes "understood the concept, wrong structure" from "completely wrong answer."
