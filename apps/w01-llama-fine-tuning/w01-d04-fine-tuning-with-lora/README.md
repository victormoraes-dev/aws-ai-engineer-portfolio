# Module 1: QLoRA Fine-Tuning of Llama 3.1 8B on SageMaker

**Certification:** AWS Certified Machine Learning Engineer – Associate (MLA-C01)  
**Estimated Time:** 2–3 hours  
**Prerequisites:** Terraform-provisioned SageMaker Notebook Instance (ml.g5.2xlarge), S3 bucket with JSONL training data, baseline metrics  
**Services Covered:** Amazon SageMaker, Amazon S3, Hugging Face PEFT, bitsandbytes

---

## Step 1: Environment Verification

Verify that the notebook has GPU access and all required packages. The lifecycle config installed `transformers`, `peft`, `accelerate`, `bitsandbytes`, and `torch`. We confirm everything is available before loading an 8B parameter model.

### Check GPU and Package Versions

```python
# Cell: Verify GPU and package availability
import torch
import transformers
import peft
import bitsandbytes as bnb
import accelerate

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
if torch.cuda.is_available():
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"bitsandbytes: {bnb.__version__}")
print(f"accelerate: {accelerate.__version__}")
```

### Configure AWS Clients

```python
# Cell: Set up boto3 clients and S3 bucket reference
import boto3
import sagemaker
from pathlib import Path

s3_client = boto3.client("s3", region_name="us-east-1")
sagemaker_client = boto3.client("sagemaker", region_name="us-east-1")
session = sagemaker.Session()

bucket = "mla-c01-lora-123456789012"  # Replace with your bucket name from terraform output
prefix = "qlora-llama"
role = sagemaker.get_execution_role()

print(f"Bucket: {bucket}")
print(f"Role: {role}")
```

---

## Step 2: Inspect the Training Data

Load the JSONL dataset prepared on Day 3 and determine the task format. The `prompt` and `completion` fields define the supervised fine-tuning objective.

```python
# Cell: Download and inspect the JSONL training data
import json

train_local = Path("train.jsonl")
s3_client.download_file(bucket, "data/train.jsonl", str(train_local))

records = []
with open(train_local, "r") as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total training records: {len(records)}")
print(f"Keys in each record: {list(records[0].keys())}")

for i, rec in enumerate(records[:2]):
    print(f"\n--- Sample {i + 1} ---")
    print(f"Prompt ({len(rec['prompt'])} chars): {rec['prompt'][:150]}...")
    print(f"Completion ({len(rec['completion'])} chars): {rec['completion'][:150]}...")
```

### Define the Prompt Formatting Function

```python
# Cell: Define the chat template for Llama 3.1 instruction format
def format_chat_sample(record: dict) -> str:
    """Format a prompt-completion pair into Llama 3.1 chat template."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{record['prompt']}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{record['completion']}"
        "<|eot_id|>"
    )

# Test the formatter
print("Formatted sample:")
print(format_chat_sample(records[0]))
```

---

## Step 3: Load Llama 3.1 8B with 4-bit Quantization

This is the QLoRA setup. We load the model in **4-bit NormalFloat (NF4)** using bitsandbytes, which reduces the memory footprint from ~16 GB (FP16) to ~5.5 GB. This is what makes fine-tuning an 8B model feasible on a single A10G.

```python
# Cell: Load tokenizer and 4-bit quantized model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

model_id = "meta-llama/Meta-Llama-3.1-8B"

# 4-bit quantization configuration — NF4 dtype for optimal quality at 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print(f"Loading tokenizer: {model_id}")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

print(f"Model loaded on: {model.hf_device_map}")
print(f"Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
```

---

## Step 4: Apply LoRA with PEFT

Attach trainable low-rank adapters to the attention projection layers. We train only ~0.5% of the total parameters, keeping the base model frozen.

```python
# Cell: Configure and apply LoRA
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,                    # Rank — higher = more capacity, more memory
    lora_alpha=32,           # Scaling factor (alpha / r = 2)
    target_modules=[
        "q_proj", "v_proj", "k_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",  # MLP layers for Llama's gated MLP
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
```

---

## Step 5: Prepare the Dataset

Tokenize all training samples into a Hugging Face Dataset object with the Llama 3.1 chat template.

```python
# Cell: Tokenize the training dataset
from datasets import Dataset

# Format all records using the Llama 3.1 chat template
formatted_texts = [format_chat_sample(r) for r in records]
dataset = Dataset.from_dict({"text": formatted_texts})

print(f"Dataset size: {len(dataset)} samples")

# Tokenize function
def tokenize_function(examples: dict) -> dict:
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=1024,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

print(f"Tokenized dataset size: {len(tokenized_dataset)}")
print(f"Input IDs shape: {tokenized_dataset[0]['input_ids'][:10]}...")
```

---

## Step 6: Configure and Run Training

Use Hugging Face's `Trainer` with memory-efficient settings. Gradient checkpointing, 8-bit AdamW, and a small batch size make this fit on 24 GB VRAM.

```python
# Cell: Configure training arguments and run QLoRA fine-tuning
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./llama-3.1-8b-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,        # Effective batch size = 16
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    warmup_ratio=0.03,
    bf16=True,
    optim="paged_adamw_8bit",             # Memory-efficient optimizer for QLoRA
    report_to="none",
    save_total_limit=2,
    remove_unused_columns=False,
    max_grad_norm=0.3,
    lr_scheduler_type="cosine",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# Start training
print(f"Starting QLoRA fine-tuning of Llama 3.1 8B...")
print(f"VRAM before: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
trainer.train()
print(f"VRAM after: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

---

## Step 7: Save the LoRA Adapters and Log Metrics

Save only the LoRA adapter weights (~160 MB) — not the full 16 GB base model. This is the key benefit of PEFT: the adapters are tiny and portable.

```python
# Cell: Save LoRA adapters and training metrics
from datetime import datetime

save_path = Path("llama-3.1-8b-qlora-final")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# Extract final loss from training logs
log_history = trainer.state.log_history
final_loss = None
for log in reversed(log_history):
    if "loss" in log:
        final_loss = log["loss"]
        break

metrics = {
    "model": "meta-llama/Meta-Llama-3.1-8B",
    "method": "QLoRA",
    "quantization": "nf4",
    "lora_r": 16,
    "lora_alpha": 32,
    "epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size,
    "effective_batch_size": training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps,
    "final_loss": final_loss,
    "trainable_params": 41943040,
    "timestamp": datetime.now().isoformat(),
}

metrics_path = save_path / "training_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"LoRA adapters saved to: {save_path}")
adapter_size_mb = sum(f.stat().st_size for f in save_path.rglob("*")) / 1e6
print(f"Adapter size: {adapter_size_mb:.1f} MB")
print(f"Final loss: {final_loss:.4f}")
```

---

## Step 8: Compare Against Baseline

Load your pre-computed baseline metrics and evaluate the improvement from QLoRA fine-tuning.

```python
# Cell: Compare fine-tuned loss against baseline
baseline: dict[str, float] = {
    "baseline_loss": 2.87,      # Replace with your actual baseline from Day 3
    "perplexity": 17.64,        # Replace with your actual baseline from Day 3
}

final_perplexity = 2 ** (final_loss / 1.4427)
improvement = baseline["baseline_loss"] - final_loss

print("=" * 65)
print("BASELINE vs. QLoRA FINE-TUNED LLAMA 3.1 8B")
print("=" * 65)
print(f"{'Metric':<30} {'Baseline':<15} {'QLoRA Fine-Tuned':<15}")
print("-" * 60)
print(f"{'Training Loss':<30} {baseline['baseline_loss']:<15.4f} {final_loss:<15.4f}")
print(f"{'Perplexity':<30} {baseline['perplexity']:<15.2f} {final_perplexity:<15.2f}")
print(f"{'Improvement':<30} {'—':<15} {improvement:<+15.4f}")
print("=" * 65)

if improvement > 0:
    print(f"✅ QLoRA improved loss by {improvement:.4f} points.")
    print(f"   The model has successfully adapted to the target task.")
else:
    print("⚠️ No improvement detected. Consider:")
    print("   - Increasing epochs (try 5)")
    print("   - Increasing LoRA rank (try r=32)")
    print("   - Increasing the learning rate (try 5e-4)")
```

---

## Step 9: Upload Adapters to S3

Persist the trained LoRA adapters to S3 for future use, deployment, or sharing with teammates.

```python
# Cell: Upload LoRA adapters to S3
import tarfile

# Package the adapters into a tarball
adapter_tarball = Path("llama-3.1-8b-qlora-adapters.tar.gz")
with tarfile.open(adapter_tarball, "w:gz") as tar:
    tar.add(save_path, arcname=save_path.name)

# Upload to S3
s3_adapter_key = f"{prefix}/adapters/{adapter_tarball.name}"
s3_client.upload_file(str(adapter_tarball), bucket, s3_adapter_key)

adapter_s3_uri = f"s3://{bucket}/{s3_adapter_key}"
print(f"Adapters uploaded to: {adapter_s3_uri}")
print(f"Tarball size: {adapter_tarball.stat().st_size / 1e6:.1f} MB")
```

---

## End of Lab Summary

| Step | Achievement |
|------|-------------|
| **Step 1** | Verified GPU environment with CUDA, PyTorch, PEFT, and bitsandbytes |
| **Step 2** | Inspected JSONL training data and defined Llama 3.1 chat template |
| **Step 3** | Loaded Llama 3.1 8B with 4-bit NF4 quantization (~5.5 GB VRAM) |
| **Step 4** | Applied LoRA (r=16) to attention and MLP layers (0.5% trainable) |
| **Step 5** | Tokenized dataset with Llama 3.1 chat format at 1024 max length |
| **Step 6** | Ran QLoRA training directly on the notebook — no SageMaker Training Job needed |
| **Step 7** | Saved tiny LoRA adapter weights (~160 MB) and extracted final loss |
| **Step 8** | Compared fine-tuned loss against baseline with improvement delta |
| **Step 9** | Uploaded adapter tarball to S3 for persistence and deployment |

**What makes this efficient vs. the SageMaker Training Job approach:**

| Aspect | SageMaker Training Job | Direct Notebook (this lab) |
|--------|----------------------|---------------------------|
| Script packaging | Must write standalone `.py` entry point + `source_dir` | Write code directly in notebook cells |
| Debugging | Check CloudWatch Logs after failure | See errors instantly in the cell output |
| Iteration | Re-package, re-upload, re-launch (5+ min per cycle) | Edit cell, re-run (seconds) |
| GPU utilization | Job provisions a dedicated instance, warms up the container | Uses the already-running notebook GPU |
| Cost | Additional instance billed per-second | No extra cost — notebook GPU is already running |

**Next in the sprint plan:** Day 5 — Run SageMaker Clarify bias analysis on your training data and compare fairness metrics between the base model and the QLoRA fine-tuned model.

---

Would you like to proceed to Day 5's lab, or adjust any hyperparameters in this lab based on your dataset size and task?