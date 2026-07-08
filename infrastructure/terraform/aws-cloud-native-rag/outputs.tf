# =============================================================================
# File: outputs.tf
# =============================================================================
# Output values for use in downstream scripts, query-time notebooks, and
# exam reference. Includes the new Bedrock model access IAM policy ARN.
# =============================================================================

output "knowledge_base_id" {
  description = "ID of the Bedrock Knowledge Base."
  value       = aws_bedrockagent_knowledge_base.kb.id
}

output "knowledge_base_arn" {
  description = "ARN of the Bedrock Knowledge Base."
  value       = aws_bedrockagent_knowledge_base.kb.arn
}

output "data_source_id" {
  description = "ID of the Bedrock Knowledge Base S3 data source."
  value       = aws_bedrockagent_data_source.s3_source.data_source_id
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket storing the source PDF."
  value       = aws_s3_bucket.documents.bucket
}

output "collection_arn" {
  description = "ARN of the OpenSearch Serverless VECTORSEARCH collection."
  value       = aws_opensearchserverless_collection.main.arn
}

output "collection_endpoint" {
  description = "Endpoint URL of the OpenSearch Serverless collection."
  value       = aws_opensearchserverless_collection.main.collection_endpoint
}

output "collection_id" {
  description = "ID of the OpenSearch Serverless collection."
  value       = aws_opensearchserverless_collection.main.id
}

output "iam_role_arn" {
  description = "ARN of the Bedrock Knowledge Base IAM role."
  value       = aws_iam_role.kb_role.arn
}

output "embedding_model_arn" {
  description = "ARN of the Bedrock embedding model (Titan Text Embeddings v2)."
  value       = local.embedding_model_arn
}

output "llm_model_arn" {
  description = "ARN of the Bedrock LLM model (Claude 3.5 Sonnet)."
  value       = local.llm_model_arn
}

output "rerank_model_arn" {
  description = "ARN of the Bedrock rerank model used for cross-encoder re-ranking."
  value       = local.rerank_model_arn
}

output "region" {
  description = "AWS region where all resources are provisioned."
  value       = var.region
}

output "model_access_policy_arn" {
  description = "ARN of the IAM policy granting least-privilege Bedrock model access (October 2025 model)."
  value       = aws_iam_policy.bedrock_model_access.arn
}

output "rag_query_role_arn" {
  description = "ARN of the IAM role for query-time Bedrock access."
  value       = aws_iam_role.rag_query_role.arn
}
