# Week 1, Day 1 — Study Guide
## AWS Certified Machine Learning Engineer - Associate (MLA-C01)

---

## Section 1: Transformer Architecture and Self-Attention

Before Transformers, models like RNNs and LSTMs processed text sequentially — one word at a time, left to right. This created a fundamental bottleneck: they could not parallelize across GPU hardware, making training extremely slow on long sequences.

The Transformer, introduced in 2017 in the paper "Attention Is All You Need," solved this by processing the entire sequence simultaneously through a mechanism called Self-Attention. Self-Attention calculates a relevance weight, called an attention score, between every pair of tokens in a sequence at the same time. This allows the model to capture contextual relationships regardless of how far apart tokens are.

There are three Transformer architectures you must know for the exam. Encoder-only models, like BERT, are used for classification and named entity recognition. Decoder-only models, like Llama-3 and GPT-4, are optimized for auto-regressive text generation. Encoder-Decoder models, like T5 and BART, are used for translation and summarization.

Llama-3 uses a Decoder-only architecture. This means each token can only attend to itself and previous tokens, never future tokens. This is enforced by a triangular mask matrix that sets future positions to negative infinity before the softmax operation. This is called Causal or Masked Self-Attention. Llama-3 8B contains 32 decoder layers.

**Exam tip:** If a question asks which architecture is best for text generation, the answer is Decoder-only. If it asks about classification or sequence labeling, the answer is Encoder-only.

---

## Section 2: The Inference Pipeline

When you send a prompt to Llama-3, it goes through a six-step pipeline.

Step 1: Tokenization. The Hugging Face AutoTokenizer converts raw text into Token IDs, which are integers. Llama-3 uses a Byte Pair Encoding tokenizer with approximately 32,000 tokens in its vocabulary.

Step 2: Embedding Lookup. Each Token ID is mapped to a dense vector of dimension 4096. This vector is the numerical representation of the token's meaning.

Step 3: Decoder Layers. The embedding passes through 32 Transformer Decoder layers. Each layer contains Masked Multi-Head Self-Attention, a Feed-Forward Network using SwiGLU activation, residual connections, and Layer Normalization.

Step 4: Output Projection. The final hidden state passes through a linear layer called the LM Head, which maps from dimension 4096 to the vocabulary size of 32,000, producing raw scores called logits.

Step 5: Softmax and Sampling. The logits are converted to probabilities. The next token is selected using strategies like greedy decoding, top-k sampling, or temperature-based sampling.

Step 6: Auto-Regressive Loop. The predicted token is appended to the input sequence, and the entire process repeats from Step 1 until a special end-of-text token is generated.

**Exam tip:** The auto-regressive loop is what makes generation slow. Each new token requires a full forward pass through all 32 layers. This is why inference latency scales with output length.

---

## Section 3: GPU VRAM Physics and Quantization

Understanding GPU memory is critical for the MLA-C01 exam, especially for cost optimization questions.

An 8 billion parameter model like Llama-3 8B requires different amounts of VRAM depending on precision. At FP32, or 32-bit floating point, the model consumes approximately 32 gigabytes. At FP16, or 16-bit, it consumes approximately 16 gigabytes. At INT4, or 4-bit quantization, it consumes approximately 4 gigabytes.

The KV Cache is a memory buffer that stores the mathematical representations of all previous tokens to avoid recalculating them for each new token. It grows linearly with context length. The formula is: KV Cache Size approximately equals 2 times the number of layers times the hidden dimension times the sequence length times the precision bytes.

For Llama-3 8B with 32 layers and a hidden dimension of 4096, a 4,000-token context at FP16 consumes approximately 2 gigabytes. But a 32,000-token context consumes approximately 16 gigabytes. This is the primary cause of CUDA Out of Memory errors during long context generation.

The solution is 4-bit NormalFloat quantization, abbreviated NF4. Using the bitsandbytes library, the model weights are compressed from 16 gigabytes to approximately 4 gigabytes, freeing 12 gigabytes of VRAM for the KV Cache and longer context windows.

**Exam tip:** Quantization is a cost optimization strategy. By compressing the model, you can deploy on a smaller, cheaper GPU instance like the ml.g5.2xlarge instead of requiring a more expensive instance with more VRAM.

---

## Section 4: RAG Architecture

RAG stands for Retrieval-Augmented Generation. The problem it solves is that large documents cannot be fed entirely into an LLM's context window without causing Out of Memory errors.

The RAG pipeline has five steps. Step 1: Chunking. Split the document into small segments of 256 to 512 tokens each. Step 2: Embedding. Convert each chunk into a vector representation. Step 3: Storage. Store the embeddings in a Vector Database. Step 4: Retrieval. The user's query is embedded, and the vector database finds the 3 to 5 most similar chunks. Step 5: Generation. The query and the retrieved chunks are sent to the LLM as context.

For the AWS exam, you must know the relevant services. Amazon OpenSearch Serverless serves as a vector database for similarity search. Amazon Kendra is a managed enterprise search service with semantic understanding. Amazon Bedrock Knowledge Bases is a fully managed, serverless RAG service.

You must also know when to choose RAG versus Fine-Tuning. Choose RAG when you need the model to answer questions based on external, frequently changing, or private factual documents, and when reducing hallucinations is critical. Choose Fine-Tuning when you need to change the model's tone, style, or specific behavioral format. Both approaches can be combined.

---

## Section 5: PyTorch Foundations

### Level 1: Tensors and CUDA

PyTorch processes Tensors, which are multi-dimensional matrices. A scalar is a 0-dimensional tensor. A vector is 1-dimensional. A matrix is 2-dimensional. Anything 3-dimensional or higher is called a Tensor.

The reason we use GPUs is parallelism. A standard CPU has approximately 8 to 16 powerful cores that process math sequentially. The NVIDIA A10G GPU found in the ml.g5.2xlarge instance has 9,216 CUDA cores that process math in parallel. Neural networks require massive matrix multiplications, which GPUs are specifically designed to handle.

CUDA stands for Compute Unified Device Architecture. It is NVIDIA's proprietary parallel computing platform. PyTorch is a Python library that packages neural network math into Tensors and ships them to CUDA cores for execution via its C++ backend called ATen.

### Level 2: Autograd

Autograd is PyTorch's automatic differentiation engine. During the Forward Pass, when data moves from input to output prediction, Autograd builds a Dynamic Computation Graph, also called a Directed Acyclic Graph or DAG. Each mathematical operation is recorded as a node in this graph. Each node stores the operation type, the input tensors, and the gradient function, which is the mathematical derivative of that specific operation.

When you call loss.backward, Autograd traverses the graph in reverse, from the loss node back through every recorded operation, applying the Chain Rule from calculus at each node. Each parameter receives its gradient attribute, containing the exact direction and magnitude to adjust its weights.

**Critical exam concept:** Training workloads require Autograd enabled, which means the Computation Graph is built and stored in GPU VRAM. This doubles or triples memory consumption compared to inference. Inference workloads must disable Autograd using torch.no_grad, which prevents the graph from being built and frees VRAM for the KV Cache.

### Level 3: nn.Module

The nn.Module is PyTorch's base class for building neural networks. It provides a standardized blueprint. Every neural network in PyTorch is a Python class that inherits from nn.Module and implements two methods.

The first method is underscore underscore init underscore underscore. Here you declare what building blocks your network will use — linear layers, dropout layers, activation functions. This is where PyTorch registers all trainable parameters.

The second method is forward. Here you define the exact sequence of operations: the data flow from input to output.

Before running inference, you must call two critical methods. First, model.eval, which disables training-specific behaviors in layers like Dropout and BatchNorm. Without calling eval, Dropout will randomly deactivate neurons during inference, producing non-deterministic garbage outputs. Second, torch.no_grad, which prevents Autograd from building a Computation Graph, saving VRAM for the KV Cache.

**Exam tip:** If an exam question describes a scenario where a model produces inconsistent results during inference, the answer is likely that model.eval was not called. If the question describes an Out of Memory error during inference, the answer is likely that torch.no_grad was not used.

---

## Section 6: AWS Infrastructure and Service Quotas

### SageMaker Notebook Instances

Amazon SageMaker Notebook Instances are the primary compute environment for ML development on AWS. The ml.g5.2xlarge instance provides an NVIDIA A10G GPU with 24 gigabytes of VRAM and 9,216 CUDA cores. This is the appropriate instance for fine-tuning 7B to 13B parameter models.

### Service Quotas

AWS restricts GPU instance quotas by default. The G and VT instance families have a default quota of zero on new accounts to prevent unexpected billing. To provision an ml.g5.2xlarge, you must request a quota increase through the AWS Service Quotas console or AWS Support. The request should include a detailed technical justification, including the workload description, why the specific instance is required, and cost control measures.

### Cost Optimization

Three cost control mechanisms were implemented. First, Infrastructure as Code using Terraform, which makes all resources ephemeral and reproducible. Second, SageMaker Lifecycle Configurations that automatically stop idle notebook instances after a specified period. Third, AWS Budgets with monthly cost limits and email notifications at 80 percent threshold.

**Exam tip:** Cost optimization questions frequently appear in Domain 4. Know that SageMaker Lifecycle Configurations can auto-stop idle instances, AWS Budgets can send alerts, and Spot Instances can reduce training costs by up to 90 percent.

### Infrastructure as Code

Terraform is used to provision AWS resources declaratively. The configuration includes an IAM Role for SageMaker execution with least-privilege policies, the SageMaker Notebook Instance with lifecycle configuration, and an AWS Budget for cost monitoring. Terraform state files must never be committed to version control because they contain sensitive resource information.

---

## Section 7: Certification Domain Alignment

The concepts from Day 1 map to all four MLA-C01 exam domains.

Domain 1, Data Preparation, worth 28 percent, is aligned with the upcoming Day 2 work on tokenization and JSONL dataset formatting.

Domain 2, ML Model Development, worth 26 percent, is aligned with model selection concepts — understanding when to use Decoder-only architectures and how fine-tuning with PEFT and LoRA works.

Domain 3, Deployment and Orchestration, worth 22 percent, is aligned with SageMaker instance selection, Infrastructure as Code, and understanding deployment infrastructure requirements.

Domain 4, Monitoring, Maintenance, and Security, worth 24 percent, is aligned with cost optimization through quantization, Service Quotas management, AWS Budgets, and Lifecycle Configurations.

The passing score is 720 out of 1,000.

---

## Section 8: Self-Assessment Questions

Question 1: What is the primary cause of CUDA Out of Memory errors during long context generation?
Answer: The KV Cache, which grows linearly with context length.

Question 2: Why must you call model.eval before running inference?
Answer: To disable Dropout and BatchNorm training behaviors that would produce non-deterministic outputs.

Question 3: What does torch.no_grad do and why is it critical for inference?
Answer: It disables Autograd from building a Computation Graph, saving VRAM for the KV Cache.

Question 4: How much VRAM does an 8B model consume at 4-bit quantization versus FP16?
Answer: Approximately 4 gigabytes at 4-bit versus 16 gigabytes at FP16.

Question 5: What is the default GPU quota for new AWS accounts and how do you increase it?
Answer: The default quota is zero. You request an increase via the AWS Service Quotas console with a technical justification.

Question 6: When should you choose RAG over Fine-Tuning?
Answer: When you need the model to answer based on external, frequently changing, or private factual documents with reduced hallucination risk.

Question 7: How many CUDA cores does the NVIDIA A10G in the ml.g5.2xlarge have?
Answer: 9,216 CUDA cores.

Question 8: What are the three Transformer architectures and their primary use cases?
Answer: Encoder-only for classification, Decoder-only for text generation, Encoder-Decoder for translation and summarization.

---

*End of Week 1, Day 1 Audio Study Guide. Total estimated narration time: approximately 12 minutes.*