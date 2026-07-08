# Week 2, Day 2 — Advanced RAG Ingestion: Docling, Hybrid Chunking & Qdrant

*Part of a 4-week accelerated program to build a production-grade AI engineering portfolio and earn the AWS Certified Machine Learning Engineer - Associate (MLA-C01) certification.*

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [The Problem: Standard Parsers Destroy Document Structure](#2-the-problem-standard-parsers-destroy-document-structure)
3. [The Architecture Decision](#3-the-architecture-decision)
4. [Environment Setup](#4-environment-setup)
5. [The Pipeline: Step by Step](#5-the-pipeline-step-by-step)
6. [Key Design Decisions](#6-key-design-decisions)
7. [Code Strategy Updates](#7-code-strategy-updates)

---

## 1. What Was Built

Day 2 extended the RAG pipeline from Day 1 by replacing every component that would fail at production scale. The deliverable was a modular, class-based ingestion pipeline that parses PDFs with layout awareness, chunks them semantically, extracts structured metadata via LLM, and stores the results in a managed cloud vector database.

The pipeline was built locally using open-source tools, each chosen to map directly to its AWS equivalent.

| Component | Lab Tool | AWS Equivalent |
|---|---|---|
| **LLM** | Groq API (`llama-3.3-70b-versatile`) | Amazon Bedrock (Claude / Llama) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | Amazon Titan Embeddings |
| **PDF Parsing** | Docling | Amazon Textract |
| **Chunking** | HybridChunker (docling) | Bedrock Knowledge Base chunking |
| **Vector Database** | Qdrant Cloud | Amazon OpenSearch Serverless |
| **Orchestration** | Custom Python classes | Agents & Knowledge Bases for Bedrock |

---

## 2. The Problem: Standard Parsers Destroy Document Structure

The Day 1 pipeline used `PyPDFLoader` with `RecursiveCharacterTextSplitter`. This approach has a fundamental flaw: it treats a PDF as a flat stream of characters and splits at fixed intervals, with no awareness of whether a split lands inside a table, mid-sentence, or between a heading and its body paragraph.

When a chunk boundary cuts through a table row, the retrieved context is incomplete. The LLM receives half a table and cannot reconstruct the missing data — it either hallucinates or admits it doesn't know. For enterprise documents where tables carry the most precise information (pricing, specifications, compliance requirements), this is not acceptable.

A second problem emerged during development: the Groq API processes inference requests fast enough that a synchronous loop over 95 chunks triggers `HTTP 429 Too Many Requests` before the loop completes. A pipeline that crashes at chunk 40 of 95 is not production-ready.

---

## 3. The Architecture Decision

Amazon Bedrock was the original target for both embeddings and the LLM. The account had a cross-region inference quota of 0, which is not adjustable for new accounts. Rather than blocking the sprint, the architecture was decoupled: the embedding layer was replaced with a local HuggingFace model and the vector store was migrated to Qdrant Cloud.

Qdrant Cloud was chosen over a local ChromaDB instance (as used in Day 1) because it eliminates Docker dependency and provides a persistent, remotely accessible store — closer to the operational model of Amazon OpenSearch Serverless. The same `all-MiniLM-L6-v2` embedding model was kept to maintain comparability with Day 1 results.

For PDF parsing, `docling` was adopted because it produces a `DoclingDocument` object — a structured representation that preserves bounding boxes, layout elements, and heading hierarchy. The `HybridChunker` consumes this object directly, using the structural metadata to enforce chunk boundaries at logical document boundaries rather than character counts.

---

## 4. Environment Setup

```bash
pip install docling qdrant-client sentence-transformers langextract groq python-dotenv
```

Add credentials to a `.env` file at the project root:

```
GROQ_API_KEY=<your_groq_api_key>
QDRANT_URL=<your_qdrant_cluster_url>
QDRANT_API_KEY=<your_qdrant_api_key>
```

---

## 5. The Pipeline: Step by Step

The notebook `w02-d02-advanced_rag.ipynb` is structured as five modular classes — `PDFParser`, `SemanticChunker`, `MetadataExtractor`, `Embedder`, `VectorStore` — orchestrated by a `RAGPipeline` class. A `Chunk` dataclass is defined first as the shared data contract between all classes.

### Step 1 — Data Container

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import uuid

@dataclass
class Chunk:
    text: str
    index: int
    source_file: str
    page_numbers: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_point(self) -> Dict[str, Any]:
        return {
            "id": self.chunk_id,
            "vector": self.embedding,
            "payload": {
                "text": self.text,
                "index": self.index,
                "source_file": self.source_file,
                "page_numbers": self.page_numbers,
                "metadata": self.metadata,
            },
        }
```

| Part | What it does |
|---|---|
| `@dataclass` | Generates `__init__`, `__repr__`, and `__eq__` automatically — no boilerplate |
| `field(default_factory=dict)` | Creates a new dict per instance — avoids the mutable default argument bug |
| `chunk_id=field(default_factory=lambda: str(uuid.uuid4()))` | Assigns a random UUID at creation time; overwritten with a stable hash by `RAGPipeline._stable_chunk_id` before upsert |
| `to_point()` | Serializes the chunk into the exact structure Qdrant's `upsert` expects — keeps the `VectorStore` class free of `Chunk` internals |

The `to_point()` method is the key coupling point between the data model and the storage layer. By encapsulating the Qdrant payload structure inside `Chunk`, the `VectorStore.upsert_chunks` method reduces to `[c.to_point() for c in chunks]` — any future change to the payload schema only requires editing `Chunk`, not `VectorStore`.


### Step 2 — PDF Parsing with Docling

```python
from docling.document_converter import DocumentConverter
from pathlib import Path

class PDFParser:
    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def parse(self, pdf_path: Path):
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        result = self._converter.convert(str(pdf_path))
        return result.document
```

| Part | What it does |
|---|---|
| `DocumentConverter` | Docling's entry point — runs layout analysis on the PDF |
| `pdf_path.exists()` check | Fails fast with a clear error before Docling attempts conversion — avoids a cryptic internal exception |
| `result.document` | Returns a `DoclingDocument` object with structural metadata, not raw text |


Returning `result.document` instead of `result.document.export_to_markdown()` is the critical distinction from a naive implementation. The `DoclingDocument` object carries bounding box data, heading levels, and table structure. Discarding it by converting to Markdown at this stage would defeat the purpose of using Docling — the `HybridChunker` in the next step requires this object to perform layout-aware splitting.

### Step 3 — Semantic Chunking with HybridChunker

```python
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
from tqdm import tqdm

MAX_TOKENS = 512

class SemanticChunker:
    def __init__(
        self,
        tokenizer_model: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = 512,
    ) -> None:
        self._hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        hf_tokenizer = HuggingFaceTokenizer(
            tokenizer=self._hf_tokenizer,
            max_tokens=chunk_size,
        )
        self._chunker = HybridChunker(
            tokenizer=hf_tokenizer, max_tokens=chunk_size, merge_peers=True
        )

    def _truncate(self, text: str) -> str:
        tokens = self._hf_tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) <= MAX_TOKENS:
            return text
        return self._hf_tokenizer.decode(tokens[:MAX_TOKENS], skip_special_tokens=True)

    def chunk(self, document, source_file: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        for idx, dl_chunk in tqdm(enumerate(self._chunker.chunk(document)), desc="Chunking", unit="chunk"):
            text = dl_chunk.text if hasattr(dl_chunk, "text") else str(dl_chunk)
            chunks.append(
                Chunk(text=self._truncate(text.strip()), index=idx, source_file=source_file)
            )
        return chunks
```

| Part | What it does |
|---|---|
| `HuggingFaceTokenizer` | Docling's tokenizer wrapper — bridges the HuggingFace tokenizer to the chunker's token-counting interface |
| `tokenizer_model=EMBEDDING_MODEL_NAME` | Reuses the same model name constant as the `Embedder` class — guarantees token counts are measured in the same vocabulary the embedding model uses |
| `chunk_size: int = 512` | Strict 512-token ceiling — matches the maximum context window of `all-MiniLM-L6-v2`; chunks exceeding this cause indexing errors at embedding time |
| `max_tokens=chunk_size` | Hard ceiling per chunk, passed to both `HuggingFaceTokenizer` and `HybridChunker` to enforce the limit at two levels |
| `merge_peers=True` | Merges adjacent chunks that share the same structural parent (e.g., consecutive list items under the same heading) — reduces fragmentation |
| `_truncate()` | Safety net that re-encodes and hard-truncates any chunk that still exceeds 512 tokens after the chunker runs |
| `tqdm(enumerate(...), desc="Chunking")` | Progress bar on the chunking loop — prevents silent timeouts on large documents |
| `hasattr(dl_chunk, "text") else str(dl_chunk)` | Defensive fallback for chunk types that don't expose a `.text` attribute directly |

`HybridChunker` operates in two passes: first it uses the `DoclingDocument` structural metadata to identify natural split points (heading boundaries, paragraph ends, table edges), then it applies the `max_tokens` ceiling only when a natural boundary cannot be found within the limit. This means most chunks are semantically complete units rather than arbitrary slices — the opposite of `RecursiveCharacterTextSplitter`, which applies the size limit first and looks for boundaries second.

The `tokenizer_model` parameter being shared with `EMBEDDING_MODEL_NAME` is a deliberate coupling: if the embedding model is ever swapped, the chunker's token budget automatically recalibrates to the new model's vocabulary. A mismatch between the chunker's tokenizer and the embedding model's tokenizer would cause chunks that appear within budget to silently exceed the embedding model's context window at inference time.

### Step 4 — Metadata Extraction via LLM

```python
import json
import concurrent.futures
from groq import Groq
from tqdm import tqdm

class MetadataExtractor:
    _SYSTEM_PROMPT = (
        "You are a metadata extraction assistant. "
        "Return ONLY a JSON object with keys: "
        "summary (string), topics (list), entities (list), keywords (list)."
    )

    def __init__(
        self,
        api_key: Optional[str] = GROQ_API_KEY,
        model: str = LLM_MODEL_NAME,
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_workers: int = 10,
    ) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for metadata extraction.")
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_workers = max_workers
        self._client = Groq(api_key=api_key)

    @staticmethod
    def validate_api_key(api_key) -> None:
        """Validate the API key before starting a batch."""
        if not api_key:
            raise ValueError("GROQ_API_KEY is required for metadata extraction.")

    def extract(self, chunk: Chunk) -> Dict[str, Any]:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": chunk.text},
                ],
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                response_format={"type": "json_object"},
            )
            metadata = json.loads(response.choices[0].message.content)
        except Exception as exc:
            logger.warning("Metadata extraction failed for chunk %d: %s.", chunk.index, exc)
            metadata = {}

        metadata.setdefault("summary", "")
        metadata.setdefault("topics", [])
        metadata.setdefault("entities", [])
        metadata.setdefault("keywords", [])
        return metadata

    def extract_batch(self, chunks: list[Chunk]) -> None:
        """Extract metadata for all chunks in parallel, mutating chunk.metadata in place."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(self.extract, chunk): chunk for chunk in chunks}
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Metadata", unit="chunk"):
                chunk = futures[future]
                chunk.metadata = future.result()
```

| Part | What it does |
|---|---|
| `_SYSTEM_PROMPT` | Class-level constant that instructs the LLM to return only a JSON object with a fixed schema — no prose, no explanation |
| `response_format={"type": "json_object"}` | Activates Groq's JSON mode — the model is constrained to emit valid JSON, eliminating parse failures from conversational wrapping |
| `temperature=0.0` | Fully deterministic output — metadata extraction is a structured classification task, not a generative one |
| `max_workers=10` | Fires 10 concurrent Groq API calls — replaces the sequential `time.sleep` loop that triggered HTTP 429 errors on the free tier |
| `validate_api_key()` | Static method called before the batch starts — fails fast with a clear error rather than discovering a missing key mid-batch |
| `extract_batch()` | Dispatches all chunks to the thread pool and collects results via `as_completed` — each chunk failure is isolated and falls back to `{}` |
| `tqdm(as_completed(...), total=len(futures))` | Progress bar on the parallel metadata loop — prevents silent timeouts on large document batches |
| `except Exception` + `metadata = {}` | Catches any API or parse failure per chunk so a single bad call doesn't abort the entire ingestion run |
| `metadata.setdefault(...)` | Guarantees all four keys are always present in the returned dict — downstream code can access `chunk.metadata["topics"]` without defensive checks |

Using the Groq API directly instead of a `langextract` abstraction gives full control over the schema. The `_SYSTEM_PROMPT` defines exactly four keys — `summary`, `topics`, `entities`, `keywords` — which are richer and more retrieval-useful than the previous `title/authors/url` fields. At query time, these fields enable payload filtering: a search can be scoped to chunks where `entities` contains a specific organization, or ranked by keyword overlap before the vector similarity score is applied.

### Step 5 — Embedding

```python
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [vec.tolist() for vec in vectors]
```

| Part | What it does |
|---|---|
| `model_name=EMBEDDING_MODEL_NAME` | Shares the same constant as `SemanticChunker` — guarantees the chunker's token budget and the embedding model's vocabulary are always in sync |
| `get_embedding_dimension()` | Reads the vector size directly from the loaded model — no hardcoded magic number |
| `@property dimension` | Exposes `_dimension` as a read-only attribute so `VectorStore` can read it without being able to overwrite it |
| `if not texts: return []` | Guards against an empty chunk list — `model.encode([])` raises an error on some backends |
| `convert_to_numpy=True` | Returns a NumPy array instead of a PyTorch tensor — required before calling `.tolist()` |
| `batch_size=32` | Processes 32 texts per forward pass — balances GPU memory usage against throughput |
| `[vec.tolist() for vec in vectors]` | Converts each NumPy row into a plain Python list of floats, which is the format Qdrant's `PointStruct` expects |

The return type change from the previous version — a NumPy matrix to a `List[List[float]]` — is significant. Qdrant's client does not accept NumPy arrays in `PointStruct.vector`; passing one raises a serialization error at upsert time. Converting to plain Python floats here keeps the `VectorStore` class free of any NumPy dependency.

### Step 6 — Vector Storage with Qdrant Cloud

```python
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

class VectorStore:
    def __init__(
        self,
        url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = COLLECTION_NAME,
        vector_size: int = 384,
        distance: str = "Cosine",
    ) -> None:
        self._qdrant_models = qdrant_models
        self._collection_name = collection_name
        self._client = QdrantClient(url=url, api_key=api_key)
        self._ensure_collection(vector_size=vector_size, distance=distance)

    def _ensure_collection(self, vector_size: int, distance: str) -> None:
        existing_names = {c.name for c in self._client.get_collections().collections}
        if self._collection_name not in existing_names:
            self._client.create_collection(
                collection_name=self._collection_name,
                vectors_config=self._qdrant_models.VectorParams(
                    size=vector_size,
                    distance=distance,
                ),
            )

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        points = [c.to_point() for c in chunks if c.embedding is not None]
        if not points:
            return
        self._client.upsert(
            collection_name=self._collection_name,
            points=points,
            wait=True,
        )

    def query(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        must_filters = []
        if filters:
            for key, value in filters.items():
                must_filters.append(
                    self._qdrant_models.FieldCondition(
                        key=key,
                        match=self._qdrant_models.MatchValue(value=value),
                    )
                )
        query_filter = (
            self._qdrant_models.Filter(must=must_filters) if must_filters else None
        )
        results = self._client.query_points(
            collection_name=self._collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        return [
            {"id": str(p.id), "score": float(p.score), "payload": p.payload or {}}
            for p in results.points
        ]
```

**Initialization parameters and what they configure inside Qdrant:**

| Parameter | Value | What it configures inside Qdrant |
|---|---|---|
| `url` | Qdrant Cloud cluster endpoint | The HTTPS address of the remote Qdrant node — all API calls (upsert, query, collection management) are sent here over gRPC or REST |
| `api_key` | Secret token | Authenticates every request to the cluster — Qdrant Cloud rejects requests without a valid key |
| `collection_name` | `"rag_collection"` | The logical namespace for this dataset — equivalent to a table in a relational database; multiple collections can coexist in the same cluster |
| `vector_size` | `384` | The dimensionality of every vector stored in this collection — must match the output dimension of `all-MiniLM-L6-v2` exactly; Qdrant rejects upserts where the vector length differs |
| `distance` | `"Cosine"` | The similarity function Qdrant uses to rank results at query time — cosine similarity measures the angle between two vectors, making it scale-invariant (a long document and a short one with the same meaning score equally) |

**How Qdrant works during a query:**

When `query_points` is called, Qdrant does not scan all 95 vectors one by one. It uses an **HNSW index** (Hierarchical Navigable Small World graph) built automatically when vectors are upserted. HNSW organizes vectors as nodes in a multi-layer graph where each node is connected to its nearest neighbors. At query time, the search starts at the top layer (coarse navigation) and descends through layers of increasing density until it reaches the approximate nearest neighbors — achieving sub-millisecond retrieval even at millions of vectors, at the cost of a small approximation error.

The `distance: str = "Cosine"` parameter is baked into the HNSW index at collection creation time and cannot be changed afterwards. Switching from Cosine to Euclidean distance would require dropping the collection and re-indexing all vectors.

**How payload filtering interacts with vector search:**

The `query_filter` built from the `filters` dict is applied as a **pre-filter** before the HNSW traversal. Qdrant first identifies which points satisfy the payload conditions (e.g., `source_file == "doc_A.pdf"`), then restricts the HNSW search to only those points. This is more efficient than post-filtering (running the full vector search and discarding non-matching results afterwards), but it requires that the filtered fields are indexed in Qdrant's payload index — otherwise Qdrant falls back to a full scan of the filtered subset.

| Method | What it does |
|---|---|
| `_ensure_collection` | Idempotent setup — creates the collection only if it doesn't exist, safe to re-run |
| `c.to_point()` | Delegates payload serialization to `Chunk.to_point()` — `VectorStore` has no knowledge of `Chunk` internals |
| `wait=True` | Blocks until Qdrant confirms the upsert is persisted and the HNSW index is updated — prevents a race condition where a query immediately after ingest returns stale results |
| `query_filter` | Translates a plain Python dict into Qdrant's `Filter` + `FieldCondition` structure for pre-filtered vector search |
| `query_points` | Triggers HNSW approximate nearest-neighbor search — returns points ranked by cosine similarity to `query_vector` |

### Step 7 — Orchestration

```python
import hashlib, uuid

class RAGPipeline:
    def __init__(
        self,
        chunk_size: int = 512,
        enable_metadata_extraction: bool = True,
    ) -> None:
        self.parser = PDFParser()
        self.chunker = SemanticChunker(chunk_size=chunk_size)
        self.embedder = Embedder()
        self.store = VectorStore(vector_size=self.embedder.dimension)
        self.enable_metadata_extraction = enable_metadata_extraction
        if enable_metadata_extraction:
            MetadataExtractor.validate_api_key(GROQ_API_KEY)
        self.extractor = MetadataExtractor() if enable_metadata_extraction else None

    def _stable_chunk_id(self, chunk: Chunk) -> str:
        digest = hashlib.sha256(
            f"{chunk.source_file}:{chunk.index}:{chunk.text}".encode("utf-8")
        ).hexdigest()
        return str(uuid.UUID(digest[:32].ljust(32, "0")))

    def ingest(self, pdf_path: Path) -> list[Chunk]:
        document = self.parser.parse(pdf_path)
        chunks = self.chunker.chunk(document, source_file=pdf_path.name)

        if not chunks:
            return []

        for chunk in chunks:
            chunk.chunk_id = self._stable_chunk_id(chunk)

        if self.extractor is not None:
            logging.info("Extracting metadata for %d chunks (parallel)...", len(chunks))
            MetadataExtractor.validate_api_key(GROQ_API_KEY)
            self.extractor.extract_batch(chunks)

        embeddings = self.embedder.embed([c.text for c in chunks])
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        self.store.upsert_chunks(chunks)
        return chunks

    def query(self, query_text: str, top_k: int = 5, filters: dict | None = None) -> list[dict]:
        query_vector = self.embedder.embed([query_text])[0]
        return self.store.query(query_vector=query_vector, top_k=top_k, filters=filters)
```

| Part | What it does |
|---|---|
| `chunk_size: int = 512` | Strict 512-token default — aligns with the embedding model's context window |
| `validate_api_key()` at init | Fails fast at pipeline construction if `GROQ_API_KEY` is missing — avoids discovering the error mid-batch |
| `validate_api_key()` at ingest | Second guard before the parallel metadata batch — catches key rotation or env changes between init and ingest |
| `_stable_chunk_id` | Generates a deterministic UUID from content hash — re-ingesting the same document upserts rather than duplicates |
| `if not chunks: return []` | Early exit guard — prevents the embedding and upsert steps from running on an empty list |
| `extract_batch(chunks)` | Parallel metadata extraction via `ThreadPoolExecutor` — replaces the sequential `time.sleep` loop |
| `enable_metadata_extraction=False` | Allows the pipeline to run without the LLM step for fast re-indexing or cost-sensitive environments |
| `query()` | Embeds the query text and delegates to `VectorStore.query` — the caller never handles raw vectors |

The `_stable_chunk_id` hash is the key to idempotent ingestion. Without it, re-running the pipeline on the same document would insert duplicate points into Qdrant. With it, Qdrant's `upsert` operation recognizes the same ID and overwrites the existing point — making the pipeline safe to re-run after a crash or a metadata schema change.

### Step 8 — Execution

```python
target_pdf = Path("./machine-learning-engineer-associate-01.pdf")

pipeline = RAGPipeline(chunk_size=512, enable_metadata_extraction=True)
ingested_chunks = pipeline.ingest(pdf_path=target_pdf)
```

**Output:**
```
Ingestion complete. Successfully processed and stored 95 chunks.
```

With the document indexed, the pipeline can be queried with natural language:

```python
results = pipeline.query(
    query_text="What are the requirements for the AWS Machine Learning Engineer certification?",
    top_k=3
)

for i, result in enumerate(results, start=1):
    score = result["score"]
    payload = result["payload"]
    print(f"--- Result {i} (Similarity Score: {score:.4f}) ---")
    print(f"Source File: {payload.get('source_file', 'Unknown')}")
    print(f"Content Preview: {payload.get('text', '')}\n")
```

**Output:**
```
--- Result 1 (Similarity Score: 0.7429) ---
Source File: machine-learning-engineer-associate-01.pdf
Content Preview: The AWS Certified Machine Learning Engineer - Associate (MLA-C01) exam
validates a candidate's ability to build, operationalize, deploy, and maintain
machine learning (ML) solutions and pipelines by using the AWS Cloud...

--- Result 2 (Similarity Score: 0.6841) ---
...
```

When `pipeline.query()` is called, the chain executes in sequence: the query text is embedded into a 384-dimensional vector using the same `all-MiniLM-L6-v2` model used during ingestion, Qdrant computes cosine similarity between the query vector and all 95 stored chunk vectors, and the top-k results are returned ranked by score. The similarity score of 0.7429 for the top result confirms the retrieval is semantically accurate — the chunk directly describes the certification requirements.

---

## 6. Key Design Decisions

**Why Docling over PyPDFLoader?** `PyPDFLoader` extracts raw text with no structural awareness — a table becomes a sequence of whitespace-separated strings, and a heading is indistinguishable from body text. Docling runs a layout analysis pass that identifies structural elements before extraction, giving the chunker the information it needs to split at logical boundaries.

**Why Qdrant Cloud over local ChromaDB?** ChromaDB persists to a local directory, which means the index is lost if the container is removed and cannot be shared across machines. Qdrant Cloud provides a persistent, remotely accessible store with the same API surface — closer to the operational model of Amazon OpenSearch Serverless and requiring zero infrastructure management.

**Why content-hashed chunk IDs?** Auto-generated IDs (e.g., sequential integers or random UUIDs) make re-ingestion destructive — every run appends duplicates. A SHA-256 hash of `source_file + index + text` produces the same ID for the same content every time, so `upsert` is idempotent. This is the same principle behind S3 object versioning and DynamoDB conditional writes.

**Why parallel metadata extraction instead of a sequential loop with `time.sleep`?** A synchronous loop over 95 chunks with a 2-second delay takes over 3 minutes and still risks HTTP 429 errors if the delay is miscalibrated. `ThreadPoolExecutor` with 10 workers fires concurrent requests and lets the OS scheduler handle I/O wait — the same wall-clock time yields 10x the throughput. Individual chunk failures are isolated via `try/except` inside `extract()`, so a single API error doesn't abort the batch.

**Why `validate_api_key` called at both init and ingest?** Calling it only at `MetadataExtractor.__init__` means a missing key raises inside the constructor, which is correct. Calling it explicitly before `extract_batch` in `RAGPipeline.ingest` adds a second guard that catches key rotation or environment changes between pipeline construction and the actual batch run — a common failure mode in long-running notebook sessions.
