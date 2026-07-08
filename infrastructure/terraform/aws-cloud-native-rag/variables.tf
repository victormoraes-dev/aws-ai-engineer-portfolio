# =============================================================================
# File: variables.tf
# =============================================================================
# Input variables for the RAG pipeline. Defaults are tuned for a lab
# environment on a fresh AWS account. Override via terraform.tfvars or
# -var flags as needed.
# =============================================================================

variable "region" {
  description = "AWS region for all resources. Bedrock and OpenSearch Serverless must be available in this region."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short project identifier used as a prefix for resource names and tags."
  type        = string
  default     = "aws-rag"
}

variable "environment" {
  description = "Environment label (e.g., lab, dev, prod). Used in naming and tags."
  type        = string
  default     = "lab"
}

variable "pdf_file_path" {
  description = "Local path to the PDF file to upload to S3 and ingest into the Knowledge Base."
  type        = string
  default     = "machine-learning-engineer-associate-01.pdf"
}

variable "collection_name" {
  description = "Name of the OpenSearch Serverless VECTORSEARCH collection."
  type        = string
  default     = "aws-rag-collection"
}

variable "kb_name" {
  description = "Name of the Bedrock Knowledge Base."
  type        = string
  default     = "aws-rag-knowledge-base"
}

variable "data_source_name" {
  description = "Name of the Bedrock Knowledge Base S3 data source."
  type        = string
  default     = "s3-pdf-source"
}

variable "embedding_model_id" {
  description = "Bedrock foundation model ID used for generating text embeddings."
  type        = string
  default     = "amazon.titan-embed-text-v2:0"
}

variable "llm_model_id" {
  description = "Bedrock foundation model ID used for generation (LLM)."
  type        = string
  default     = "anthropic.claude-3-5-sonnet-20241022-v2:0"
}

variable "rerank_model_id" {
  description = "Bedrock foundation model ID used for cross-encoder re-ranking at query time."
  type        = string
  default     = "amazon.rerank-v1:0"
}

variable "index_name" {
  description = "Name of the vector index inside the OpenSearch Serverless collection."
  type        = string
  default     = "bedrock-knowledge-base-default-index"
}

variable "chunk_max_tokens" {
  description = "Maximum number of tokens per chunk for fixed-size chunking."
  type        = number
  default     = 1000
}

variable "chunk_overlap_percentage" {
  description = "Overlap percentage between adjacent chunks for fixed-size chunking."
  type        = number
  default     = 20
}

variable "company_name" {
  description = "Company name for the Anthropic model access use-case form."
  type        = string
  default     = "MLA-C01 Study Lab"
}

variable "company_website" {
  description = "Company website for the Anthropic model access use-case form."
  type        = string
  default     = "https://example.com"
}

variable "intended_users" {
  description = "Number of intended users for the Anthropic use-case form."
  type        = string
  default     = "1"
}

variable "industry_option" {
  description = "Industry for the Anthropic use-case form."
  type        = string
  default     = "Technology"
}

variable "use_cases" {
  description = "Use-case description for the Anthropic use-case form."
  type        = string
  default     = "AI Engineering certification study and RAG pipeline development"
}

variable "tags" {
  description = "Additional tags to apply to all taggable resources, merged with common_tags."
  type        = map(string)
  default = {
    Project     = "aws-rag"
    ManagedBy   = "terraform"
    Environment = "lab"
  }
}

# -----------------------------------------------------------------------------
# Variable: Lambda deployment package path
# -----------------------------------------------------------------------------
# Path to the zipped Lambda function deployment package. The user zips the
# lambda_function.py file into lambda_function.zip (default) before running
# terraform apply.
variable "lambda_source_path" {
  description = "Path to the zipped Lambda function deployment package."
  type        = string
  default     = "lambda_function.zip"
}

variable "lambda_source_file" {
  default = "../../../apps/w02-rag-system/w02-d04-aws-cloud-native-rag/code/lambda_function.py"
}