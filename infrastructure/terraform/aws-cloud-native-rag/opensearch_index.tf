# =============================================================================
# File: opensearch_index.tf
# =============================================================================
# Creates the vector index inside the OpenSearch Serverless collection.
# Bedrock Knowledge Base requires this index to exist BEFORE KB creation.
# The index uses knn_vector type for hybrid search (k-NN + BM25).
# =============================================================================

resource "opensearch_index" "bedrock_kb" {
  name                           = var.index_name
  number_of_shards               = 2
  number_of_replicas             = 0
  index_knn                      = true
  index_knn_algo_param_ef_search = 512

  # Titan Text Embeddings v2 default dimension = 1024
  # (v1 uses 1536 — adjust if using a different embedding model)
  mappings = jsonencode({
    properties = {
      "bedrock-knowledge-base-default-vector" = {
        type      = "knn_vector"
        dimension = 1024
        method = {
          name   = "hnsw"
          engine = "faiss"
          parameters = {
            m               = 16
            ef_construction = 512
          }
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
