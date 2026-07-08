# =============================================================================
# File: locals.tf
# =============================================================================
# Computed values shared across resources: unique names with random suffixes,
# model ARNs constructed from region + model ID, the caller identity used in
# OpenSearch data-access policies, and the common tag map.
# =============================================================================

data "aws_caller_identity" "current" {}

data "aws_partition" "current" {}

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_file = var.lambda_source_file
  output_path = "${path.module}/lambda_function_payload.zip"
}

resource "random_string" "this" {
  length  = 8
  upper   = false
  special = false
}


locals {
  # Unique resource names with a random suffix to avoid collisions on a
  # fresh account that may have been used for prior labs.
  bucket_name = "${var.project_name}-${var.environment}-${random_string.this.result}"
  role_name   = "${var.project_name}-kb-role-${random_string.this.result}"

  # Common tags merged with user-supplied tags.
  common_tags = merge(
    var.tags,
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Module      = "advanced-rag"
    }
  )

  # Caller identity used in OpenSearch data-access policy principals.
  caller_account_id = data.aws_caller_identity.current.account_id
  caller_arn        = "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:root"

  # Bedrock foundation model ARNs constructed from region and model ID.
  # Format: arn:aws:bedrock:<region>::foundation-model/<model-id>
  embedding_model_arn = "arn:${data.aws_partition.current.partition}:bedrock:${var.region}::foundation-model/${var.embedding_model_id}"
  llm_model_arn       = "arn:${data.aws_partition.current.partition}:bedrock:${var.region}::foundation-model/${var.llm_model_id}"
  rerank_model_arn    = "arn:${data.aws_partition.current.partition}:bedrock:${var.region}::foundation-model/${var.rerank_model_id}"

  # OpenSearch Serverless collection ARN and index ARN.
  collection_arn       = "arn:${data.aws_partition.current.partition}:aoss:${var.region}:${data.aws_caller_identity.current.account_id}:collection/${var.collection_name}"
  collection_index_arn = "arn:${data.aws_partition.current.partition}:aoss:${var.region}:${data.aws_caller_identity.current.account_id}:collection/${var.collection_name}"

  lambda_function_name = "${var.project_name}-${var.environment}-auto-ingestion"
}
