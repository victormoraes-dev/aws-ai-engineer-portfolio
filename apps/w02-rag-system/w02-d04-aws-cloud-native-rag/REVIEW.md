# Week 2, Day 4 — Cloud-Native RAG Pipeline with Amazon Bedrock Knowledge Base

*Part of a 4-week accelerated program to build a production-grade AI engineering portfolio and earn the AWS Certified Machine Learning Engineer - Associate (MLA-C01) certification.*

---

## Table of Contents

1. [What Was Built](#1-what-was-built)
2. [The Problem](#2-the-problem)
3. [The Architecture Decision](#3-the-architecture-decision)
4. [AWS Services Used in the Solution](#4-aws-services-used-in-the-solution)
5. [Environment Setup](#5-environment-setup)
6. [The Implementation: Step by Step](#6-the-implementation-step-by-step)
7. [Key Design Decisions](#7-key-design-decisions)
8. [Issues Encountered and Resolutions](#8-issues-encountered-and-resolutions)

---

## 1. What Was Built

A fully automated, cloud-native Retrieval-Augmented Generation (RAG) pipeline on AWS using Infrastructure as Code (Terraform). The pipeline automatically ingests PDF documents uploaded to S3, generates vector embeddings via Amazon Titan Text Embeddings V2, stores them in Amazon OpenSearch Serverless, and supports hybrid retrieval with reranking via Amazon Rerank V1 and answer generation via Anthropic Claude 3.5 Sonnet V2. An event-driven Lambda function triggers Bedrock Knowledge Base ingestion jobs automatically on S3 object creation — no manual sync required.

| Component | Lab Tool | AWS Equivalent |
|---|---|---|
| **Document Storage** | S3 Bucket with event notifications | Amazon S3 |
| **Vector Store** | OpenSearch Serverless VECTORSEARCH collection | Amazon OpenSearch Serverless |
| **Embeddings** | Titan Text Embeddings V2 (1024-dim vectors) | Amazon Bedrock |
| **Reranking** | Amazon Rerank V1 | Amazon Bedrock |
| **Generation** | Claude 3.5 Sonnet V2 | Amazon Bedrock |
| **Auto-Ingestion** | Lambda function triggered by S3 events | AWS Lambda |
| **Knowledge Base** | Bedrock Knowledge Base with OpenSearch storage | Amazon Bedrock |
| **Infrastructure** | Terraform with AWS + OpenSearch providers | Terraform / IaC |

---

## 2. The Problem

Building a production-grade RAG pipeline requires orchestrating multiple AWS services with complex IAM trust relationships, SigV4 authentication for OpenSearch Serverless, and event-driven ingestion workflows. The naive approach of manually syncing documents through the Bedrock console does not scale and introduces human latency. Additionally, OpenSearch Serverless has a unique authentication model that differs from managed OpenSearch Service — it uses `aoss` as the SigV4 service name instead of `es`, and requires data access policies (not just IAM) to grant index-level permissions.

The pipeline must handle:
- **Automatic ingestion** — documents uploaded to S3 should trigger embedding generation and vector indexing without manual intervention
- **Fine-grained access control** — both the Bedrock KB role and the Terraform caller identity need permissions on the OpenSearch collection and indices
- **Model access** — three different foundation models (embeddings, reranking, generation) each require IAM `InvokeModel` permissions and on-demand quota allocation
- **Composite ID handling** — Terraform's `aws_bedrockagent_data_source` resource returns a composite ID (`kb_id,ds_id`) that must be split when passing to the Lambda environment

---

## 3. The Architecture Decision

**Why OpenSearch Serverless over managed OpenSearch Service?**

OpenSearch Serverless provides a fully managed vector search collection type (`VECTORSEARCH`) with automatic scaling, pay-per-use pricing, and native k-NN vector search support. Managed OpenSearch Service requires cluster provisioning, shard management, and capacity planning — overkill for a RAG knowledge base that needs vector similarity search.

**Why Bedrock Knowledge Base over custom LangChain pipeline?**

Bedrock Knowledge Base abstracts the ingestion pipeline (chunking, embedding, indexing) into a managed service. The alternative — a custom Lambda that calls Titan Embeddings, chunks documents, and writes to OpenSearch — would require significantly more code and maintenance. The KB also provides built-in hybrid retrieval and reranking APIs.

**Why event-driven Lambda over manual sync?**

The Bedrock Knowledge Base does not auto-sync when new documents are added to the S3 data source. Without the Lambda, a user would need to manually trigger ingestion through the console or CLI after every upload. The Lambda closes this gap by calling `StartIngestionJob` on S3 `ObjectCreated` events.

---

## 4. AWS Services Used in the Solution

Each service below is provisioned via Terraform and serves a specific role in the RAG pipeline. The Terraform resource blocks are referenced to show exactly how each service is instantiated.

---

### 4.1 Amazon S3 (Simple Storage Service)

**Purpose:** Stores source PDF documents that feed the RAG pipeline. Also configured with event notifications to trigger the Lambda function on object upload.

**Why S3?** S3 is the native data source for Bedrock Knowledge Bases. It provides durable, encrypted object storage with built-in event notification capabilities that eliminate the need for a separate message broker. The `inclusion_prefixes` on the KB data source and the `filter_prefix` on the S3 notification are aligned to the same `documents/` path, ensuring only PDFs in that prefix trigger ingestion.

**Terraform provisioning:**

```hcl
# S3 bucket for source documents
resource "aws_s3_bucket" "documents" {
  bucket = "${var.project_name}-${random_string.this.result}"
  tags   = local.common_tags
}

# Server-side encryption (SSE-S3)
resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block all public access
resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# S3 event notification → Lambda
resource "aws_s3_bucket_notification" "documents_notification" {
  bucket = aws_s3_bucket.documents.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.auto_ingestion.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "documents/"
    filter_suffix       = ".pdf"
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke]
}
```

| Resource | What It Provisions | Why It's Needed |
|---|---|---|
| `aws_s3_bucket.documents` | The document storage bucket | Bedrock KB reads PDFs from this bucket during ingestion |
| `aws_s3_bucket_server_side_encryption_configuration` | SSE-S3 encryption | Security best practice — encrypts documents at rest |
| `aws_s3_bucket_public_access_block` | Blocks public access | Prevents accidental data exposure |
| `aws_s3_bucket_notification` | S3 → Lambda trigger | Event-driven architecture — no polling required |

---

### 4.2 Amazon OpenSearch Serverless

**Purpose:** Acts as the vector database. Stores 1024-dimensional embeddings generated by Titan Text Embeddings V2 in a k-NN index, enabling semantic similarity search during RAG retrieval.

**Why OpenSearch Serverless?** Unlike managed OpenSearch Service (which requires cluster sizing, node count, and EBS volume configuration), OpenSearch Serverless is auto-scaling and serverless. The `VECTORSEARCH` collection type is specifically optimized for vector similarity search with FAISS engine support. The pipeline uses three security layers:

1. **Encryption policy** — AWS-owned KMS key for data at rest
2. **Network policy** — Public access (lab environment; would be VPC-only in production)
3. **Data access policy** — Resource-based permissions granting both the KB role and the Terraform caller access to collection and index operations

**Terraform provisioning:**

```hcl
# Encryption policy — AWS-owned key
resource "aws_opensearchserverless_security_policy" "encryption" {
  name        = "${var.collection_name}-encr"
  type        = "encryption"
  policy = jsonencode({
    Rules = [{
      Resource     = ["collection/${var.collection_name}"]
      ResourceType = "collection"
    }]
    AWSOwnedKey = true
  })
}

# Network policy — public access for lab
resource "aws_opensearchserverless_security_policy" "network" {
  name        = "${var.collection_name}-net"
  type        = "network"
  policy = jsonencode([{
    AllowFromPublic = true
    Rules = [{
      Resource     = ["collection/${var.collection_name}"]
      ResourceType = "collection"
    }]
  }])
}

# VECTORSEARCH collection
resource "aws_opensearchserverless_collection" "main" {
  name        = var.collection_name
  type        = "VECTORSEARCH"
  depends_on  = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network
  ]
}

# Data access policy — dual principals
resource "aws_opensearchserverless_access_policy" "data" {
  name        = "${var.collection_name}-data"
  type        = "data"
  policy = jsonencode([{
    Rules = [
      {
        Resource     = ["collection/${var.collection_name}"]
        ResourceType = "collection"
        Permission = [
          "aoss:CreateCollectionItems",
          "aoss:DeleteCollectionItems",
          "aoss:UpdateCollectionItems",
          "aoss:DescribeCollectionItems"
        ]
      },
      {
        Resource     = ["index/${var.collection_name}/*"]
        ResourceType = "index"
        Permission = [
          "aoss:CreateIndex", "aoss:DeleteIndex",
          "aoss:UpdateIndex", "aoss:DescribeIndex",
          "aoss:ReadDocument", "aoss:WriteDocument"
        ]
      }
    ]
    Principal = [
      local.caller_arn,
      aws_iam_role.kb_role.arn
    ]
  }])
  depends_on = [aws_opensearchserverless_collection.main]
}
```

| Resource | What It Provisions | Why It's Needed |
|---|---|---|
| `aws_opensearchserverless_security_policy.encryption` | KMS encryption policy | Required before collection creation — data at rest |
| `aws_opensearchserverless_security_policy.network` | Network access policy | Required before collection creation — controls who can reach the endpoint |
| `aws_opensearchserverless_collection.main` | The vector search collection | Hosts the k-NN index where embeddings are stored |
| `aws_opensearchserverless_access_policy.data` | Data access policy (resource-based) | Grants KB role + caller identity permission to create/read/write indices — IAM alone is not sufficient for OpenSearch Serverless |

---

### 4.3 OpenSearch Serverless Vector Index (via OpenSearch Provider)

**Purpose:** Defines the k-NN index schema inside the OpenSearch Serverless collection. Maps the vector field, text field, and metadata field that Bedrock KB expects during ingestion and retrieval.

**Why a separate provider?** The `aws` Terraform provider can create the OpenSearch Serverless collection and security policies, but it cannot create indices inside the collection. The `opensearch-project/opensearch` provider connects directly to the collection endpoint using SigV4 authentication and creates the index with the correct k-NN mappings.

**Terraform provisioning:**

```hcl
# OpenSearch provider configured for Serverless
provider "opensearch" {
  url                   = aws_opensearchserverless_collection.main.collection_endpoint
  healthcheck           = false
  aws_region            = var.region
  aws_signature_service = "aoss"
  sign_aws_requests     = true
}

# k-NN vector index
resource "opensearch_index" "bedrock_kb" {
  name                           = var.index_name
  number_of_shards               = 2
  number_of_replicas             = 0
  index_knn                      = true
  index_knn_algo_param_ef_search = 512

  mappings = jsonencode({
    properties = {
      "bedrock-knowledge-base-default-vector" = {
        type      = "knn_vector"
        dimension = 1024
        method = {
          name       = "hnsw"
          engine     = "faiss"
          parameters = { m = 16, ef_construction = 512 }
          space_type = "l2"
        }
      }
      "AMAZON_BEDROCK_METADATA" = {
        type  = "text"
        index = false
      }
      "AMAZON_BEDROCK_TEXT_CHUNK" = {
        type  = "text"
        index = true
      }
    }
  })

  force_destroy = true
  depends_on = [
    aws_opensearchserverless_collection.main,
    time_sleep.aoss_index_propagation
  ]
}
```

| Resource | What It Provisions | Why It's Needed |
|---|---|---|
| `provider "opensearch"` | SigV4-authenticated connection to the Serverless endpoint | The `aws_signature_service = "aoss"` setting is critical — defaults to `es` which causes 403 |
| `opensearch_index.bedrock_kb` | The k-NN index with FAISS/HNSW mappings | Bedrock KB writes embeddings to this index during ingestion and reads from it during retrieval |

---

### 4.4 Amazon Bedrock Knowledge Base

**Purpose:** Manages the RAG ingestion pipeline — reads PDFs from S3, chunks them, calls Titan Embeddings V2 to generate vectors, and writes them to the OpenSearch Serverless index. Also provides the `Retrieve` and `RetrieveAndGenerate` APIs for query-time retrieval.

**Why Bedrock KB?** Building a custom ingestion pipeline would require: (1) a PDF parser, (2) a chunking strategy, (3) embedding API calls with retry logic, (4) OpenSearch document writes, and (5) metadata management. Bedrock KB handles all of this as a managed service. The KB also supports hybrid search (keyword + vector) and integrates natively with the Rerank API.

**Terraform provisioning:**

```hcl
# Knowledge Base
resource "aws_bedrockagent_knowledge_base" "kb" {
  name     = var.kb_name
  role_arn = aws_iam_role.kb_role.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${var.region}::foundation-model/amazon.titan-embed-text-v2:0"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"
    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.main.arn
      vector_index_name = var.index_name
      field_mapping {
        metadata_field = "AMAZON_BEDROCK_METADATA"
        text_field     = "AMAZON_BEDROCK_TEXT_CHUNK"
        vector_field   = "bedrock-knowledge-base-default-vector"
      }
    }
  }
}

# S3 data source
resource "aws_bedrockagent_data_source" "s3_source" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.kb.id
  name              = var.data_source_name

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn         = aws_s3_bucket.documents.arn
      inclusion_prefixes = ["documents/"]
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "FIXED_SIZE"
      fixed_size_chunking_configuration {
        max_tokens         = var.chunk_max_tokens
        overlap_percentage = var.chunk_overlap_percentage
      }
    }
  }
}
```

| Resource | What It Provisions | Why It's Needed |
|---|---|---|
| `aws_bedrockagent_knowledge_base.kb` | The KB with embedding model + vector store config | Orchestrates the entire ingestion pipeline — chunking, embedding, indexing |
| `aws_bedrockagent_data_source.s3_source` | S3 data source with chunking config | Tells the KB where to read documents and how to chunk them |

---

### 4.5 Amazon Bedrock Foundation Models

**Purpose:** Three models serve distinct roles in the RAG pipeline:

| Model | Model ID | Path | Purpose |
|---|---|---|---|
| Titan Text Embeddings V2 | `amazon.titan-embed-text-v2:0` | Ingestion | Converts PDF text chunks into 1024-dimensional vectors |
| Amazon Rerank V1 | `amazon.rerank-v1:0` | Query | Reranks retrieved chunks by relevance before LLM generation |
| Claude 3.5 Sonnet V2 | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Query | Generates the final natural-language answer from reranked context |

**Why these specific models?** Titan Embeddings V2 produces 1024-dim vectors (matching the index `dimension = 1024`), supports up to 8K input tokens, and is optimized for text similarity tasks. Rerank V1 provides cross-encoder reranking that significantly improves retrieval precision over pure vector similarity. Claude 3.5 Sonnet V2 offers strong reasoning and instruction-following capabilities for answer generation.

**Terraform provisioning (IAM policy granting access):**

```hcl
resource "aws_iam_policy" "bedrock_model_access" {
  name = "${var.project_name}-bedrock-model-access"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "InvokeEmbeddingModel"
        Effect    = "Allow"
        Actions   = ["bedrock:InvokeModel"]
        Resources = ["arn:aws:bedrock:${var.region}::foundation-model/amazon.titan-embed-text-v2:0"]
      },
      {
        Sid       = "InvokeLLMModel"
        Effect    = "Allow"
        Actions   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
        Resources = ["arn:aws:bedrock:${var.region}::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"]
      },
      {
        Sid       = "InvokeRerankModel"
        Effect    = "Allow"
        Actions   = ["bedrock:InvokeModel"]
        Resources = ["arn:aws:bedrock:${var.region}::foundation-model/amazon.rerank-v1:0"]
      },
      {
        Sid       = "BedrockRuntime"
        Effect    = "Allow"
        Actions   = ["bedrock:Retrieve", "bedrock:RetrieveAndGenerate"]
        Resources = ["*"]
      },
      {
        Sid       = "RerankAPI"
        Effect    = "Allow"
        Actions   = ["bedrock-agent-runtime:Rerank"]
        Resources = ["*"]
      }
    ]
  })
}
```

| Statement | What It Grants | Why It's Needed |
|---|---|---|
| `InvokeEmbeddingModel` | `bedrock:InvokeModel` on Titan Embeddings V2 | KB role calls this during ingestion to generate vectors |
| `InvokeLLMModel` | `bedrock:InvokeModel` + `InvokeModelWithResponseStream` on Claude 3.5 | Query path — generates answers; streaming for real-time responses |
| `InvokeRerankModel` | `bedrock:InvokeModel` on Rerank V1 | Some implementations call the model directly via InvokeModel |
| `BedrockRuntime` | `bedrock:Retrieve` + `RetrieveAndGenerate` | KB retrieval and generation APIs |
| `RerankAPI` | `bedrock-agent-runtime:Rerank` | Dedicated reranking API (separate from InvokeModel) |

---

### 4.6 AWS Lambda

**Purpose:** Event-driven auto-ingestion trigger. When a PDF is uploaded to S3, the Lambda function calls `StartIngestionJob` on the Bedrock Knowledge Base, eliminating the need for manual sync.

**Why Lambda?** Bedrock Knowledge Bases do not auto-sync when new documents are added to the S3 data source. Without this Lambda, every document upload would require a manual `aws bedrock-agent start-ingestion-job` CLI call. Lambda provides sub-second response to S3 events, scales automatically with upload volume, and integrates natively with S3 event notifications without requiring an event bus or message queue.

**Terraform provisioning:**

```hcl
# Lambda function
resource "aws_lambda_function" "auto_ingestion" {
  function_name    = local.lambda_function_name
  role             = aws_iam_role.auto_ingestion_lambda_role.arn
  handler          = "lambda_function.lambda_handler"
  runtime          = "python3.11"
  timeout          = 300
  memory_size      = 128

  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256

  environment {
    variables = {
      KNOWLEDGE_BASE_ID = aws_bedrockagent_knowledge_base.kb.id
      DATA_SOURCE_ID    = aws_bedrockagent_data_source.s3_source.data_source_id
      REGION            = var.region
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.auto_ingestion_lambda_logs,
    aws_iam_role.auto_ingestion_lambda_role,
    aws_iam_role_policy.auto_ingestion_lambda_policy
  ]
}

# S3 permission to invoke Lambda
resource "aws_lambda_permission" "allow_s3_invoke" {
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.auto_ingestion.function_name
  principal      = "s3.amazonaws.com"
  source_account = data.aws_caller_identity.current.account_id
  source_arn     = aws_s3_bucket.documents.arn
}
```

| Resource | What It Provisions | Why It's Needed |
|---|---|---|
| `aws_lambda_function.auto_ingestion` | The Python 3.11 Lambda function | Executes `StartIngestionJob` on S3 events |
| `aws_lambda_permission.allow_s3_invoke` | S3 → Lambda invoke permission | S3 needs explicit permission to invoke the Lambda function |
| `DATA_SOURCE_ID` env var | Uses `.data_source_id` (not `.id`) | `.id` returns composite `kb_id,ds_id` which fails API validation |

---

### 4.7 AWS IAM (Identity and Access Management)

**Purpose:** Defines trust relationships and permissions for three distinct roles in the pipeline:

| Role | Trusted By | Purpose |
|---|---|---|
| `aws-rag-kb-role` | `bedrock.amazonaws.com` | Bedrock KB assumes this role to read S3, call Titan Embeddings, and write to OpenSearch |
| `aws-rag-query-role` | `lambda.amazonaws.com` | Query Lambda assumes this role to call Bedrock Retrieve/Generate and Rerank APIs |
| `aws-rag-lab-auto-ingestion-lambda-role` | `lambda.amazonaws.com` | Ingestion Lambda assumes this role to call `StartIngestionJob` and write CloudWatch logs |

**Why three separate roles?** Following the principle of least privilege, each component gets only the permissions it needs. The KB role needs S3 read + OpenSearch write + embedding model access. The ingestion Lambda role needs only `StartIngestionJob` + CloudWatch Logs. The query role needs `Retrieve`, `RetrieveAndGenerate`, `Rerank`, and LLM model access. Combining these into a single role would violate security best practices and make auditing difficult.

**Terraform provisioning (KB role as example):**

```hcl
# KB Role — trust bedrock.amazonaws.com
resource "aws_iam_role" "kb_role" {
  name = "${var.project_name}-kb-role-${random_string.this.result}"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = { Service = "bedrock.amazonaws.com" }
      Action = "sts:AssumeRole"
    }]
  })
}

# KB inline policy — S3 + OpenSearch
resource "aws_iam_role_policy" "kb_permissions" {
  name = "${var.project_name}-kb-permissions"
  role = aws_iam_role.kb_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "S3Access"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*"
        ]
      },
      {
        Sid    = "OpenSearchAccess"
        Effect = "Allow"
        Action = ["aoss:APIAccessAll"]
        Resource = "arn:aws:aoss:${var.region}:${data.aws_caller_identity.current.account_id}:collection/*"
      }
    ]
  })
}

# Lambda role — bedrock:StartIngestionJob + CloudWatch Logs
resource "aws_iam_role_policy" "auto_ingestion_lambda_policy" {
  name = "${var.project_name}-auto-ingestion-lambda-policy"
  role = aws_iam_role.auto_ingestion_lambda_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:*"
      },
      {
        Sid    = "BedrockIngestion"
        Effect = "Allow"
        Action = ["bedrock:StartIngestionJob", "bedrock:GetIngestionJob"]
        Resource = [
          "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:knowledge-base/*",
          "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:data-source/*"
        ]
      }
    ]
  })
}
```

| Resource | What It Provisions | Why It's Needed |
|---|---|---|
| `aws_iam_role.kb_role` | KB execution role with bedrock trust | Bedrock assumes this role to perform ingestion and retrieval |
| `aws_iam_role_policy.kb_permissions` | S3 read + OpenSearch API access | KB needs to read PDFs from S3 and write vectors to OpenSearch |
| `aws_iam_role.auto_ingestion_lambda_role` | Lambda execution role | Lambda assumes this to call Bedrock APIs |
| `aws_iam_role_policy.auto_ingestion_lambda_policy` | `StartIngestionJob` + CloudWatch Logs | Lambda needs to trigger ingestion and log execution |
| `aws_iam_policy.bedrock_model_access` | `InvokeModel` on all three models | Attached to KB role and query role — grants model invocation |

---

### 4.8 Amazon CloudWatch Logs

**Purpose:** Collects execution logs from the Lambda function for debugging and monitoring.

**Why CloudWatch Logs?** Lambda automatically sends stdout/stderr to CloudWatch Logs, but the log group must exist with the correct naming convention (`/aws/lambda/<function-name>`) and retention policy. Pre-creating the log group via Terraform ensures consistent retention settings and avoids the default "never expire" behavior.

**Terraform provisioning:**

```hcl
resource "aws_cloudwatch_log_group" "auto_ingestion_lambda_logs" {
  name              = "/aws/lambda/${local.lambda_function_name}"
  retention_in_days = 7
  tags              = local.common_tags
}
```

| Resource | What It Provisions | Why It's Needed |
|---|---|---|
| `aws_cloudwatch_log_group` | Log group with 7-day retention | Stores Lambda execution logs — used to verify ingestion job triggers and diagnose errors |

---

### 4.9 Terraform Providers

**Purpose:** Infrastructure as Code provisioning for all AWS resources.

**Why two providers?** The `hashicorp/aws` provider manages AWS-native resources (S3, Lambda, IAM, Bedrock, OpenSearch Serverless collection/policies). The `opensearch-project/opensearch` provider connects directly to the OpenSearch Serverless endpoint to create the k-NN index — something the AWS provider cannot do.

```hcl
# AWS provider
provider "aws" {
  region = var.region
}

# OpenSearch provider — configured for Serverless
provider "opensearch" {
  url                   = aws_opensearchserverless_collection.main.collection_endpoint
  healthcheck           = false
  aws_region            = var.region
  aws_signature_service = "aoss"
  sign_aws_requests     = true
}
```

| Provider | What It Manages | Why It's Needed |
|---|---|---|
| `hashicorp/aws` | All AWS resources (S3, Lambda, IAM, Bedrock, OpenSearch Serverless) | Primary IaC provider for AWS infrastructure |
| `opensearch-project/opensearch` | The k-NN index inside the collection | The AWS provider cannot create indices — only the OpenSearch provider can |

---

## 5. Environment Setup

```bash
# Terraform providers
terraform init

# AWS CLI for verification
pip install awscli

# Set AWS credentials (default credential chain)
export AWS_REGION=us-east-1
```

```env
# Lambda environment variables (set via Terraform)
KNOWLEDGE_BASE_ID=<10-char KB ID>
DATA_SOURCE_ID=<10-char data source ID>
REGION=us-east-1
```

---

## 6. The Implementation: Step by Step

### Step 1 — OpenSearch Serverless Collection with Security Policies

```hcl
resource "aws_opensearchserverless_security_policy" "encryption" {
  name        = "${var.collection_name}-encr"
  type        = "encryption"
  policy = jsonencode({
    Rules = [{
      Resource     = ["collection/${var.collection_name}"]
      ResourceType = "collection"
    }]
    AWSOwnedKey = true
  })
}

resource "aws_opensearchserverless_security_policy" "network" {
  name        = "${var.collection_name}-net"
  type        = "network"
  policy = jsonencode([{
    AllowFromPublic = true
    Rules = [{
      Resource     = ["collection/${var.collection_name}"]
      ResourceType = "collection"
    }]
  }])
}

resource "aws_opensearchserverless_collection" "main" {
  name        = var.collection_name
  type        = "VECTORSEARCH"
  depends_on  = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network
  ]
}
```

Encryption and network policies must exist before the collection can be created. The collection will remain in `CREATING` status until both policies are attached.

---

### Step 2 — Data Access Policy with Dual Principals

```hcl
resource "aws_opensearchserverless_access_policy" "data" {
  name        = "${var.collection_name}-data"
  type        = "data"
  policy = jsonencode([{
    Rules = [
      {
        Resource     = ["collection/${var.collection_name}"]
        ResourceType = "collection"
        Permission = [
          "aoss:CreateCollectionItems",
          "aoss:DeleteCollectionItems",
          "aoss:UpdateCollectionItems",
          "aoss:DescribeCollectionItems"
        ]
      },
      {
        Resource     = ["index/${var.collection_name}/*"]
        ResourceType = "index"
        Permission = [
          "aoss:CreateIndex", "aoss:DeleteIndex",
          "aoss:UpdateIndex", "aoss:DescribeIndex",
          "aoss:ReadDocument", "aoss:WriteDocument"
        ]
      }
    ]
    Principal = [
      local.caller_arn,
      aws_iam_role.kb_role.arn
    ]
  }])
  depends_on = [aws_opensearchserverless_collection.main]
}
```

**Critical lesson:** The OpenSearch Terraform provider authenticates as the **caller identity**. If the caller is not listed as a Principal, the provider receives `403 Forbidden` when attempting to create the `opensearch_index` resource. Both principals are required.

---

### Step 3 — OpenSearch Provider Configuration for Serverless

```hcl
provider "opensearch" {
  url                   = aws_opensearchserverless_collection.main.collection_endpoint
  healthcheck           = false
  aws_region            = var.region
  aws_signature_service = "aoss"
  sign_aws_requests     = true
}
```

**Critical lesson:** The provider defaults to `es` as the SigV4 service name. OpenSearch Serverless requires `aoss`. Without this setting, all requests are rejected with `403 authorization_exception`.

---

### Step 4 — Vector Index with k-NN Mappings

```hcl
resource "opensearch_index" "bedrock_kb" {
  name                           = var.index_name
  number_of_shards               = 2
  number_of_replicas             = 0
  index_knn                      = true
  index_knn_algo_param_ef_search = 512

  mappings = jsonencode({
    properties = {
      "bedrock-knowledge-base-default-vector" = {
        type      = "knn_vector"
        dimension = 1024
        method = {
          name       = "hnsw"
          engine     = "faiss"
          parameters = { m = 16, ef_construction = 512 }
          space_type = "l2"
        }
      }
      "AMAZON_BEDROCK_METADATA" = {
        type  = "text"
        index = false
      }
      "AMAZON_BEDROCK_TEXT_CHUNK" = {
        type  = "text"
        index = true
      }
    }
  })

  force_destroy = true
  depends_on = [
    aws_opensearchserverless_collection.main,
    time_sleep.aoss_index_propagation
  ]
}
```

The `time_sleep.aoss_index_propagation` resource includes a `triggers` block tied to the data access policy version, forcing the sleep to re-run whenever the policy changes.

---

### Step 5 — Bedrock Knowledge Base with OpenSearch Storage

```hcl
resource "aws_bedrockagent_knowledge_base" "kb" {
  name     = var.kb_name
  role_arn = aws_iam_role.kb_role.arn

  knowledge_base_configuration {
    type = "VECTOR"
    vector_knowledge_base_configuration {
      embedding_model_arn = "arn:aws:bedrock:${var.region}::foundation-model/amazon.titan-embed-text-v2:0"
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"
    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.main.arn
      vector_index_name = var.index_name
      field_mapping {
        metadata_field = "AMAZON_BEDROCK_METADATA"
        text_field     = "AMAZON_BEDROCK_TEXT_CHUNK"
        vector_field   = "bedrock-knowledge-base-default-vector"
      }
    }
  }
}
```

---

### Step 6 — S3 Data Source with Inclusion Prefixes

```hcl
resource "aws_bedrockagent_data_source" "s3_source" {
  knowledge_base_id = aws_bedrockagent_knowledge_base.kb.id
  name              = var.data_source_name

  data_source_configuration {
    type = "S3"
    s3_configuration {
      bucket_arn         = aws_s3_bucket.documents.arn
      inclusion_prefixes = ["documents/"]
    }
  }

  vector_ingestion_configuration {
    chunking_configuration {
      chunking_strategy = "FIXED_SIZE"
      fixed_size_chunking_configuration {
        max_tokens         = var.chunk_max_tokens
        overlap_percentage = var.chunk_overlap_percentage
      }
    }
  }
}
```

**Critical lesson:** The Bedrock API rejects an empty `inclusion_prefixes = []` with `ValidationException: Member must have length greater than or equal to 1`.

---

### Step 7 — Event-Driven Lambda Auto-Ingestion

```python
import json
import logging
import os
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
DATA_SOURCE_ID = os.environ.get("DATA_SOURCE_ID", "")
REGION = os.environ.get("REGION", "us-east-1")

bedrock_agent_client = boto3.client("bedrock-agent", region_name=REGION)


def start_ingestion_job() -> str:
    if not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
        raise ValueError(
            "Missing required environment variables: "
            "KNOWLEDGE_BASE_ID and DATA_SOURCE_ID must be set."
        )

    response = bedrock_agent_client.start_ingestion_job(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId=DATA_SOURCE_ID,
        description="Auto-triggered ingestion job from S3 ObjectCreated event.",
    )

    ingestion_job = response.get("ingestionJob", {})
    job_id = ingestion_job.get("ingestionJobId", "")

    if not job_id:
        raise RuntimeError("StartIngestionJob response did not contain an ingestionJobId.")

    logger.info("Ingestion job started successfully. jobId=%s", job_id)
    return job_id


def lambda_handler(event, context):
    logger.info("Received S3 event: %s", json.dumps(event))

    try:
        s3_objects = []
        for record in event.get("Records", []):
            s3_info = record.get("s3", {})
            bucket = s3_info.get("bucket", {}).get("name", "")
            key = s3_info.get("object", {}).get("key", "")
            if bucket and key:
                s3_objects.append({"bucket": bucket, "key": key})

        if not s3_objects:
            return {"statusCode": 200, "body": json.dumps("No valid S3 objects found.")}

        job_id = start_ingestion_job()

        return {
            "statusCode": 200,
            "jobId": job_id,
            "processedObjects": s3_objects,
            "body": json.dumps(
                f"Started Bedrock ingestion job {job_id} for {len(s3_objects)} object(s)."
            ),
        }

    except ClientError as exc:
        error_code = exc.response.get("Error", {}).get("Code", "Unknown")
        error_message = exc.response.get("Error", {}).get("Message", str(exc))
        logger.error("ClientError: code=%s, message=%s", error_code, error_message)
        return {
            "statusCode": 500,
            "body": json.dumps(f"ClientError: {error_code} - {error_message}"),
        }
```

**Critical lesson:** The `DATA_SOURCE_ID` environment variable must use `aws_bedrockagent_data_source.s3_source.data_source_id` — NOT `.id`. The `.id` attribute returns a composite ID (`knowledge_base_id,data_source_id`), which fails the API's regex validation `[0-9a-zA-Z]{10}`.

---

### Step 8 — S3 Event Notification to Lambda

```hcl
resource "aws_s3_bucket_notification" "documents_notification" {
  bucket = aws_s3_bucket.documents.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.auto_ingestion.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "documents/"
    filter_suffix       = ".pdf"
  }

  depends_on = [aws_lambda_permission.allow_s3_invoke]
}
```

---

## 7. Key Design Decisions

**Why `aws_signature_service = "aoss"` instead of the default `es`?**

OpenSearch Serverless uses a different SigV4 service name (`aoss`) than managed OpenSearch Service (`es`). The Terraform OpenSearch provider defaults to `es`. Without explicitly setting `aws_signature_service = "aoss"`, all API calls to the Serverless collection endpoint are signed with the wrong service name and rejected with `403 authorization_exception`. This is the single most common failure point when provisioning OpenSearch Serverless indices via Terraform.

**Why both `local.caller_arn` and `aws_iam_role.kb_role.arn` in the data access policy?**

OpenSearch Serverless uses a dual-layer access model: IAM policies (identity-based) AND data access policies (resource-based). The Terraform OpenSearch provider authenticates as the caller identity — so the caller must be a Principal in the data access policy to create indices. The KB role must also be a Principal so Bedrock can read/write vectors during ingestion and retrieval. Removing either principal breaks a different part of the pipeline.

**Why `data_source_id` instead of `id` for the Lambda environment variable?**

The `aws_bedrockagent_data_source` Terraform resource returns a composite `id` in the format `knowledge_base_id,data_source_id` (e.g., `WI9IXIPNK4,WMSEULILFZ`). The Bedrock `StartIngestionJob` API expects only the 10-character data source ID. The `data_source_id` attribute returns just the data source ID, satisfying the API's regex constraint `[0-9a-zA-Z]{10}`.

**Why `inclusion_prefixes = ["documents/"]` instead of `[]`?**

The Bedrock `CreateDataSource` API rejects empty arrays with `ValidationException: Member must have length greater than or equal to 1`. Using `["documents/"]` aligns with the S3 notification `filter_prefix`, ensuring the data source ingests the same objects that trigger the Lambda.

**Why `aoss:APIAccessAll` instead of `aoss:APIInvokeAll`?**

`aoss:APIInvokeAll` is not a valid OpenSearch Serverless IAM action. The correct action for full API access is `aoss:APIAccessAll`. This typo causes silent failures where the KB role can assume the role but cannot write vectors to the collection.

**Why a `time_sleep` with `triggers` instead of a static sleep?**

A static `time_sleep` resource only runs once during initial creation. When the data access policy is updated (e.g., adding a new principal), the OpenSearch Serverless API needs propagation time before the changes take effect. By using a `triggers` block tied to `policy_version`, the sleep re-runs whenever the policy changes, preventing race conditions where the `opensearch_index` resource attempts creation before the updated policy has propagated.

---

## 8. Issues Encountered and Resolutions

| # | Error | Root Cause | Fix |
|---|---|---|---|
| 1 | `403 Forbidden authorization_exception` on `opensearch_index` | Caller identity missing from data access policy Principals | Added `local.caller_arn` to Principal list |
| 2 | `403 Forbidden` persisted after fix #1 | OpenSearch provider signing requests with `es` instead of `aoss` | Set `aws_signature_service = "aoss"` on the provider block |
| 3 | `ValidationException: inclusionPrefixes must have length >= 1` | Empty array `[]` rejected by Bedrock API | Changed to `["documents/"]` |
| 4 | `AccessDeniedException: bedrock:StartIngestionJob` | Lambda role missing Bedrock ingestion permissions | Added `bedrock:StartIngestionJob` and `bedrock:GetIngestionJob` to Lambda IAM policy |
| 5 | `ValidationException: dataSourceId failed regex [0-9a-zA-Z]{10}` | Terraform `.id` returns composite `kb_id,ds_id` | Changed to `.data_source_id` attribute |
| 6 | `429 ThrottlingException` on all model invocations | New AWS account has applied quota = 0 for all on-demand Bedrock models | Opened AWS Support case to set applied quota to default values |

---

## Models Used in the RAG Pipeline

| Path | Model | Model ID | Purpose |
|---|---|---|---|
| **Ingestion** | Titan Text Embeddings V2 | `amazon.titan-embed-text-v2:0` | Generates 1024-dim vector embeddings from PDF chunks |
| **Query — Retrieval** | Amazon Rerank V1 | `amazon.rerank-v1:0` | Reranks retrieved chunks by relevance before LLM generation |
| **Query — Generation** | Claude 3.5 Sonnet V2 | `anthropic.claude-3-5-sonnet-20241022-v2:0` | Generates the final answer from reranked context |

---

## Pending: AWS Support Case

The pipeline infrastructure is 100% provisioned and correct. The only remaining blocker is the account-level Bedrock on-demand quota initialization (applied value = 0 on new accounts). An AWS Support case has been prepared requesting the applied quota be set to default values for all three models. Once resolved, the full pipeline will be operational: **S3 upload → Lambda trigger → Bedrock ingestion → Titan embeddings → OpenSearch indexing → hybrid retrieval → reranking → Claude generation**.