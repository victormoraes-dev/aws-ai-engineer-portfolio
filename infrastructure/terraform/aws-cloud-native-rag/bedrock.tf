# =============================================================================
# File: bedrock.tf
# =============================================================================
# Amazon Bedrock Knowledge Base and its S3 data source. The Knowledge Base
# orchestrates parsing, fixed-size chunking, and embedding of the PDF, writing
# vectors into the OpenSearch Serverless collection. Re-ranking and LLM
# generation happen at query time via the Bedrock runtime APIs.
# =============================================================================

# Bedrock Knowledge Base configured for vector storage in OpenSearch Serverless.
# Field mappings match the default index schema expected by Bedrock.
resource "aws_bedrockagent_knowledge_base" "kb" {
  name     = var.kb_name
  role_arn = aws_iam_role.kb_role.arn
  tags     = local.common_tags

  knowledge_base_configuration {
    type = "VECTOR"

    vector_knowledge_base_configuration {
      embedding_model_arn = local.embedding_model_arn
    }
  }

  storage_configuration {
    type = "OPENSEARCH_SERVERLESS"

    opensearch_serverless_configuration {
      collection_arn    = aws_opensearchserverless_collection.main.arn
      vector_index_name = var.index_name

      field_mapping {
        vector_field   = "bedrock-knowledge-base-default-vector"
        text_field     = "AMAZON_BEDROCK_TEXT_CHUNK"
        metadata_field = "AMAZON_BEDROCK_METADATA"
      }
    }
  }

  depends_on = [
    time_sleep.iam_propagation,
    time_sleep.aoss_policy_propagation,
    aws_opensearchserverless_collection.main,
    aws_iam_role_policy_attachment.kb_bedrock_model_access,
    opensearch_index.bedrock_kb
  ]
}

# S3 data source for the Knowledge Base. Ingests all objects under the bucket
# root (inclusion_prefixes = []). Fixed-size chunking with configurable token
# count and overlap percentage.
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
