# Week 1, Day 3 — Review: Prompt Engineering Baseline on AWS Bedrock

*Study reference for the AWS Certified Machine Learning Engineer - Associate (MLA-C01)*

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Evaluation Dataset](#2-the-evaluation-dataset)
3. [The 3 Prompt Strategies](#3-the-3-prompt-strategies)
4. [Automated Evaluation Pipeline](#4-automated-evaluation-pipeline)
5. [The Evaluation Script](#5-the-evaluation-script)
6. [Certification Alignment (MLA-C01)](#6-certification-alignment-mla-c01)

---

## 1. Overview

**Target model:** `meta.llama3-8b-instruct-v1:0` via Amazon Bedrock `Converse` API.

**Goal:** Establish a rigorous performance baseline by testing foundation models via AWS Bedrock using prompt engineering techniques before LoRA fine-tuning on Day 4. Without a baseline, improvement cannot be quantified.

The baseline covers **3 prompt strategies** tested against **15 domain-specific legal questions** across **3 categories**: Contracts (6), NDAs (4), Intellectual Property (5).

---

## 2. The Evaluation Dataset

| Category | Count | Focus |
|----------|-------|-------|
| **Contracts** | 6 | Contract interpretation, breach, enforceability |
| **NDAs** | 4 | Confidentiality obligations, scope, exceptions |
| **Intellectual Property** | 5 | Patents, copyrights, trademarks, trade secrets |

Each question has a **ground-truth expected answer** written by a domain expert (1–3 explanatory sentences). Categories enable per-domain performance analysis to detect where prompt engineering helps most and where fine-tuning may be required.

---

## 3. The 3 Prompt Strategies

### Strategy A — Zero-Shot Direct

Minimal instruction, no examples. Tests the model's **raw domain knowledge**.

```text
Question: {question}
Answer:
```

### Strategy B — Zero-Shot with Role

Adds a **persona** (senior corporate attorney) and structure constraints (1–3 sentences). Tests whether **role priming** improves output quality.

```text
You are a senior corporate attorney. Answer the following legal question in 1–3 clear, explanatory sentences.

Question: {question}
Answer:
```

### Strategy C — Few-Shot (2 examples + target question)

Provides **2 in-domain examples** before the target question. Tests the model's **in-context learning capability**.

```text
You are a senior corporate attorney. Answer each question in 1–3 clear, explanatory sentences.

Example 1:
Question: {example_question_1}
Answer: {example_answer_1}

Example 2:
Question: {example_question_2}
Answer: {example_answer_2}

Target Question: {question}
Answer:
```

**Key insight:** The performance gap between Strategy A and Strategy C directly quantifies the opportunity for LoRA fine-tuning on Day 4.

---

## 4. Automated Evaluation Pipeline

### Metrics

| Metric | Role | Rationale |
|--------|------|-----------|
| **ROUGE-L** | Primary | Measures the longest common subsequence between predicted and expected answer. Tolerates paraphrasing. |
| **BERTScore** | Secondary | Uses embeddings for semantic similarity. Higher correlation with human judgment. |
| **Latency** | Operational | Wall-clock time per inference call to Bedrock. |
| **Token Consumption** | Cost | Input and output token counts for cost analysis. |

### Pipeline Flow

```text
Load dataset (JSON)
    ↓
For each question:
    Run Strategy A → generate prediction
    Run Strategy B → generate prediction
    Run Strategy C → generate prediction
    ↓
Score each prediction against ground-truth expected answer
    ↓
Aggregate results by strategy and by legal category
    ↓
Export full results matrix to CSV
```

The baseline CSV becomes the "before" column in the Day 4 LoRA fine-tuning comparison. The delta (Fine-Tuned ROUGE-L − Baseline ROUGE-L) is the portfolio headline metric.

---

## 5. The Evaluation Script

**Script:** `code/wk1_day3_baseline_eval.py`

**Dependencies:** `boto3`, `evaluate` (HuggingFace), `rouge_score`, `bert-score`, `pandas`.

```python
import json
import boto3
import evaluate
import pandas as pd

# Configuration
MODEL_ID = "meta.llama3-8b-instruct-v1:0"
BEDROCK = boto3.client("bedrock-runtime", region_name="us-east-1")

# Dataset loader
def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)

# Prompt template factory
prompt_templates = {
    "A_zero_shot_direct": lambda q: f"Question: {q}\nAnswer:",
    "B_zero_shot_role": lambda q: (
        "You are a senior corporate attorney. Answer in 1–3 sentences.\n\n"
        f"Question: {q}\nAnswer:"
    ),
    "C_few_shot": lambda q, exs: build_few_shot_prompt(q, exs),
}

# Bedrock inference wrapper
def invoke_bedrock(prompt: str, model_id: str = MODEL_ID) -> dict:
    response = BEDROCK.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.3},
    )
    return response

# Scoring engine
def score_prediction(prediction: str, reference: str) -> dict:
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    rouge_result = rouge.compute(predictions=[prediction], references=[reference])
    bert_result = bertscore.compute(
        predictions=[prediction],
        references=[reference],
        lang="en",
        model_type="distilbert-base-uncased",
    )
    return {
        "rouge_l": rouge_result["rougeL"],
        "bert_score_f1": bert_result["f1"][0],
    }

# Results aggregator & CSV exporter
def run_baseline(dataset: list[dict]) -> pd.DataFrame:
    results = []
    for item in dataset:
        for strategy_name, template in prompt_templates.items():
            prediction = invoke_bedrock(template(item["question"]))
            scores = score_prediction(prediction, item["answer"])
            results.append({
                "id": item["id"],
                "category": item["category"],
                "strategy": strategy_name,
                "prediction": prediction,
                "expected": item["answer"],
                **scores,
            })
    return pd.DataFrame(results)

if __name__ == "__main__":
    dataset = load_dataset("../w01-d02-tokenization-and-embeddings/dataset/legal-faq-dataset.json")
    df = run_baseline(dataset)
    df.to_csv("wk1_day3_baseline_results.csv", index=False)
    print(df.groupby("strategy")[["rouge_l", "bert_score_f1"]].mean())
    print(df.groupby(["category", "strategy"])[["rouge_l"]].mean())
```

---

## 6. Certification Alignment (MLA-C01)

> **[CERTIFICATION FOCUS]** **Domain 2 (ML Model Development) — Task 2.3: Analyze model performance.** The baseline methodology maps directly to this domain: model evaluation techniques, metrics (ROUGE, BERTScore, F1, precision, recall), and establishing performance baselines before fine-tuning.
>
> The `boto3` `bedrock-runtime` client + `Converse` API + `inferenceConfig` pattern used in the script is the standard AWS SDK integration pattern tested in the exam for deploying and consuming foundation models.

---

*Study reference for AWS MLA-C01 exam preparation.*
