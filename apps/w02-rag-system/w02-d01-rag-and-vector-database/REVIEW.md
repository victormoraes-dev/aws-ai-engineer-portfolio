# Week 2, Day 1 — RAG Pipeline & Vector Databases

*Part of a 4-week accelerated program to build a production-grade AI engineering portfolio and earn the AWS Certified Machine Learning Engineer - Associate (MLA-C01) certification.*

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [The Problem: Documents Are Too Large for a Context Window](#2-the-problem-documents-are-too-large-for-a-context-window)
3. [The Architecture Decision](#3-the-architecture-decision)
4. [Environment Setup](#4-environment-setup)
5. [The Pipeline: Step by Step](#5-the-pipeline-step-by-step)
6. [Key Design Decisions](#6-key-design-decisions)

---

## 1. What Was Built

Week 2 shifted focus from fine-tuning a model to building a system around one. The deliverable was a working **Retrieval-Augmented Generation (RAG) pipeline** — a system that ingests a document, indexes it semantically, and answers questions about it using only the relevant sections as context.

The pipeline was built locally using open-source tools, but every component was chosen to map 1:1 to its AWS equivalent. The goal was to prove understanding of the fundamental mechanics of AI systems — not just how to configure managed services.

| Component | Lab Tool | AWS Equivalent |
|---|---|---|
| **LLM** | Groq API (`llama-3.1-8b-instant`) | Amazon Bedrock (Claude / Llama) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Amazon Titan Embeddings |
| **Vector Database** | ChromaDB | Amazon OpenSearch Serverless |
| **Orchestration** | LangChain (LCEL) | Agents & Knowledge Bases for Bedrock |
| **Data Storage** | Local PDF file | Amazon S3 |

---

## 2. The Problem: Documents Are Too Large for a Context Window

A 500-page PDF cannot be fed entirely into an LLM's context window. The KV Cache — the memory structure that stores all previous tokens during generation — would exhaust GPU VRAM before the model could process the full document.

RAG solves this by never sending the full document. Instead, it retrieves only the 3–5 most relevant sections at query time and sends those as context. The model answers based on what was retrieved, not what it memorized during training.

This approach has a second advantage: the knowledge base can be updated by re-indexing documents, without retraining the model.

---

## 3. The Architecture Decision

AWS was not used for the hands-on lab because of the same quota constraints encountered in Week 1. Rather than waiting, the pipeline was built with tools that are architecturally identical to the AWS stack — so the lab simultaneously serves as exam preparation.

The Groq API was chosen over running a local model (e.g., via Ollama) because it eliminates Docker dependency and cold-start latency while keeping the architecture identical. The same `llama-3.1-8b-instant` model used in the Week 1 prompt engineering baseline was reused here, making the two sessions directly comparable.

---

## 4. Environment Setup

The LLM is served via the Groq API — no local model runtime required.

```bash
pip install langchain-core langchain-groq langchain-huggingface langchain-community chromadb pypdf python-dotenv
```

Add the API key to a `.env` file at the project root:

```
GROQ_API_KEY=<your_groq_api_key>
```

---

## 5. The Pipeline: Step by Step

The notebook `w02-d01-local_rag.ipynb` ingests a document, chunks it, creates embeddings, stores them in a local vector database, and retrieves context to answer a question.

### Step 1 — Ingestion

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("machine-learning-engineer-associate-01.pdf")
docs = loader.load()
```

| Part | What it does |
|---|---|
| `PyPDFLoader` | Loads the local PDF document |
| `loader.load()` | Reads the document into memory as a list of `Document` objects |

On AWS, this step is replaced by uploading the file to S3 and pointing a Bedrock Knowledge Base at the bucket.

### Step 2 — Chunking

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
```

| Parameter | What it does |
|---|---|
| `chunk_size=1000` | Splits the document into segments of 1,000 characters |
| `chunk_overlap=200` | Keeps 200 characters of overlap between consecutive chunks to prevent loss of context at boundaries |

Chunk size is one of the most consequential decisions in a RAG system. Chunks that are too large waste context window space and dilute relevance. Chunks that are too small lose the surrounding context needed to answer questions accurately. The 200-character overlap ensures that a sentence split across two chunks is still fully represented in at least one of them.

### Step 3 — Embeddings & Vector Database

```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
```

| Part | What it does |
|---|---|
| `HuggingFaceEmbeddings` | Generates a 384-dimensional vector for each chunk using `all-MiniLM-L6-v2` |
| `Chroma.from_documents` | Stores the embedded chunks in a local ChromaDB vector database, persisted to disk |
| `as_retriever` | Configures the database to return the top `k=3` most semantically similar chunks at query time |

An important constraint: embeddings generated by different models cannot be compared. If the embedding model is changed after indexing, the entire dataset must be re-embedded. On AWS, this means re-running the Bedrock Knowledge Base sync job.

### Step 4 — LLM Setup

```python
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)
```

| Part | What it does |
|---|---|
| `ChatGroq` | LangChain-native Groq client, compatible with any LCEL chain |
| `llama-3.1-8b-instant` | Same model used in the Week 1 prompt engineering baseline |
| `temperature=0.1` | Low temperature for factual, deterministic answers grounded in retrieved context |

### Step 5 — Orchestration

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know."
    "\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

| Part | What it does |
|---|---|
| `system_prompt` | Instructs the LLM to strictly use the retrieved context and admit when it doesn't know |
| `RunnablePassthrough` | Passes the raw question string through as `input` without modification |
| `StrOutputParser` | Extracts the plain text string from the LLM response object |
| LCEL `\|` pipe | Composes retriever → prompt → LLM → parser into a single callable chain |

The LCEL pipe syntax replaces the deprecated `create_retrieval_chain` / `create_stuff_documents_chain` pattern with a cleaner, composable chain that is easier to extend and debug.

### Step 6 — Execution

```python
response = rag_chain.invoke("How could I prepare for the certification?")
print(response)
```

**Output:**
```
Based on the provided documents, it appears that the AWS Certified Machine Learning Engineer - Associate
(MLA-C01) certification is designed for individuals with at least 1 year of experience using Amazon
SageMaker and other AWS services for ML engineering. To prepare for the certification, I would recommend
the following steps:

1. Gain relevant experience: Ensure you have at least 1 year of experience using Amazon SageMaker and
   other AWS services for ML engineering.
2. Familiarize yourself with the exam guide: Study the exam guide provided in the documents...
3. Develop a strong understanding of ML algorithms and their use cases.
4. Practice with sample questions and hands-on labs.
...
```

When `rag_chain.invoke()` is called, the chain executes in sequence: the retriever embeds the question and fetches the top 3 matching chunks from ChromaDB, those chunks are injected into the system prompt as `{context}`, the full prompt is sent to the LLM, and the response is parsed and returned as a plain string.

---

## 6. Key Design Decisions

**Why ChromaDB over a managed vector store?** ChromaDB persists to disk and requires no infrastructure. It is architecturally equivalent to Amazon OpenSearch Serverless for the purposes of this lab — both store vectors and perform k-NN similarity search. The switch to OpenSearch on AWS requires changing only the vector store initialization line.

**Why `chunk_overlap=200`?** Without overlap, a sentence that falls at the boundary between two chunks is split in half. The retriever may return the chunk containing only the second half, missing the subject of the sentence. Overlap ensures boundary content is fully represented in at least one chunk.

**Why the same embedding model as Week 1?** Consistency. The `all-MiniLM-L6-v2` model was validated in Week 1 Day 2 with cosine similarity scores confirming it encodes domain semantics correctly. Reusing it here means the retrieval quality is already understood.

**Why `k=3`?** Returning 3 chunks balances context richness against context window consumption. Returning more chunks increases the chance of finding the right answer but also increases the risk of diluting the prompt with irrelevant content — and increases token cost on paid APIs.
