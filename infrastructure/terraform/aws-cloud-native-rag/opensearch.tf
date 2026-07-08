# =============================================================================
# File: opensearch.tf
# =============================================================================
# OpenSearch Serverless VECTORSEARCH collection with security policies.
# =============================================================================

resource "aws_opensearchserverless_security_policy" "encryption" {
  name        = "${var.collection_name}-encr"
  type        = "encryption"
  description = "Encryption policy for ${var.collection_name}"

  policy = jsonencode({
    Rules = [
      {
        Resource     = ["collection/${var.collection_name}"]
        ResourceType = "collection"
      }
    ]
    AWSOwnedKey = true
  })
}

# -----------------------------------------------------------------------------
# Network security policy
# -----------------------------------------------------------------------------
resource "aws_opensearchserverless_security_policy" "network" {
  name        = "${var.collection_name}-net"
  type        = "network"
  description = "Network policy for ${var.collection_name}"

  policy = jsonencode([
    {
      AllowFromPublic = true
      Rules = [
        {
          Resource     = ["collection/${var.collection_name}"]
          ResourceType = "collection"
        }
      ]
    }
  ])
}

# -----------------------------------------------------------------------------
# OpenSearch Serverless VECTORSEARCH collection
# -----------------------------------------------------------------------------
resource "aws_opensearchserverless_collection" "main" {
  name        = var.collection_name
  description = "Vector store for ${var.project_name} Advanced RAG pipeline"
  type        = "VECTORSEARCH"

  depends_on = [
    aws_opensearchserverless_security_policy.encryption,
    aws_opensearchserverless_security_policy.network
  ]

  lifecycle {
    prevent_destroy = true
  }

  tags = local.common_tags
}

# -----------------------------------------------------------------------------
# Data access policy
# -----------------------------------------------------------------------------
resource "aws_opensearchserverless_access_policy" "data" {
  name        = "${var.collection_name}-data"
  type        = "data"
  description = "Data access policy for ${var.collection_name}"

  policy = jsonencode([
    {
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
            "aoss:CreateIndex",
            "aoss:DeleteIndex",
            "aoss:UpdateIndex",
            "aoss:DescribeIndex",
            "aoss:ReadDocument",
            "aoss:WriteDocument"
          ]
        }
      ]
      Principal = [
        local.caller_arn,
        aws_iam_role.kb_role.arn
      ]
    }
  ])

  depends_on = [
    aws_opensearchserverless_collection.main
  ]
}
