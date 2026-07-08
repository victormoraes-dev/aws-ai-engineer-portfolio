# =============================================================================
# File: iam.tf
# =============================================================================
# IAM roles and policies for the Bedrock Knowledge Base and for Bedrock model
# access. Includes the October 2025 simplified model-access IAM policy that
# replaces the retired console Model Access page.
# =============================================================================

# ---------------------------------------------------------------------------
# Bedrock Knowledge Base service role
# ---------------------------------------------------------------------------

# Trust policy allowing the Bedrock service to assume this role.
data "aws_iam_policy_document" "kb_trust" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["bedrock.amazonaws.com"]
    }
  }
}

# IAM role assumed by Bedrock Knowledge Base to read S3 objects and write
# vectors to the OpenSearch Serverless collection.
resource "aws_iam_role" "kb_role" {
  name               = local.role_name
  assume_role_policy = data.aws_iam_policy_document.kb_trust.json
  tags               = local.common_tags
}

# Inline permissions for the KB role: read objects from the S3 bucket and
# invoke the OpenSearch Serverless collection APIs (ingestion + retrieval).
data "aws_iam_policy_document" "kb_permissions" {
  statement {
    sid    = "S3ReadDataSource"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = [
      aws_s3_bucket.documents.arn,
      "${aws_s3_bucket.documents.arn}/*"
    ]
  }

  statement {
    sid    = "OpenSearchServerlessAccess"
    effect = "Allow"
    actions = [
      "aoss:APIAccessAll"
    ]
    resources = [
      "arn:${data.aws_partition.current.partition}:aoss:${var.region}:${data.aws_caller_identity.current.account_id}:collection/*"
    ]
  }
}

# Attach the inline permissions document to the KB role.
resource "aws_iam_role_policy" "kb_permissions" {
  name   = "${local.role_name}-permissions"
  role   = aws_iam_role.kb_role.id
  policy = data.aws_iam_policy_document.kb_permissions.json
}

# Wait 15 seconds after IAM role creation to allow credentials to propagate
# across AWS regions before the Knowledge Base attempts to assume the role.
resource "time_sleep" "iam_propagation" {
  depends_on      = [aws_iam_role_policy.kb_permissions]
  create_duration = "15s"
}

# ---------------------------------------------------------------------------
# 6b. Bedrock model access IAM policy (October 2025 simplified model access)
# ---------------------------------------------------------------------------
# As of October 2025, AWS simplified Bedrock model access. Models are enabled
# by default with correct IAM permissions. The old Model Access page and
# PutFoundationModelEntitlement API are retired. Access is now controlled via
# standard IAM policies granting bedrock:InvokeModel on specific model ARNs.
#
# Reference:
#   - https://docs.aws.amazon.com/bedrock/latest/userguide/security_iam_id-based-policy-examples.html
#   - https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html
# ---------------------------------------------------------------------------

# Managed IAM policy granting least-privilege access to the specific Bedrock
# foundation models used by this RAG pipeline. Model ARNs are explicit (no
# wildcards) to enforce least privilege.
data "aws_iam_policy_document" "bedrock_model_access" {
  statement {
    sid    = "InvokeEmbeddingModel"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel"
    ]
    resources = [
      local.embedding_model_arn
    ]
  }

  statement {
    sid    = "InvokeLLMModel"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ]
    resources = [
      local.llm_model_arn
    ]
  }

  statement {
    sid    = "KnowledgeBaseRetrieve"
    effect = "Allow"
    actions = [
      "bedrock:Retrieve",
      "bedrock:RetrieveAndGenerate"
    ]
    resources = ["*"]
  }

  statement {
    sid    = "RerankAPI"
    effect = "Allow"
    actions = [
      "bedrock-agent-runtime:Rerank"
    ]
    resources = ["*"]
  }

  statement {
    sid    = "ListAndGetFoundationModels"
    effect = "Allow"
    actions = [
      "bedrock:ListFoundationModels",
      "bedrock:GetFoundationModel"
    ]
    resources = ["*"]
  }

  statement {
    sid    = "MarketplaceModelAutoEnablement"
    effect = "Allow"
    actions = [
      "aws-marketplace:Subscribe",
      "aws-marketplace:Unsubscribe",
      "aws-marketplace:ViewSubscriptions"
    ]
    resources = ["*"]
  }
}

# Managed policy object that can be attached to multiple principals.
resource "aws_iam_policy" "bedrock_model_access" {
  name        = "${var.project_name}-bedrock-model-access"
  description = "Least-privilege Bedrock model access for the Advanced RAG pipeline (October 2025 model access model)"
  policy      = data.aws_iam_policy_document.bedrock_model_access.json
  tags        = local.common_tags
}

# Attach the Bedrock model access policy to the Knowledge Base role so the KB
# can invoke the embedding model during ingestion.
resource "aws_iam_role_policy_attachment" "kb_bedrock_model_access" {
  role       = aws_iam_role.kb_role.name
  policy_arn = aws_iam_policy.bedrock_model_access.arn
}

# IAM role that the query-time application (Lambda, EC2, etc.) assumes
# to invoke Bedrock models and query the Knowledge Base
data "aws_iam_policy_document" "rag_query_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "rag_query_role" {
  name               = "${var.project_name}-query-role"
  assume_role_policy = data.aws_iam_policy_document.rag_query_assume_role.json
  tags               = local.common_tags
}

# Attach the Bedrock model access policy to the query role
resource "aws_iam_role_policy_attachment" "rag_query_bedrock_access" {
  role       = aws_iam_role.rag_query_role.name
  policy_arn = aws_iam_policy.bedrock_model_access.arn
}

# -----------------------------------------------------------------------------
# IAM Role: Lambda execution role
# -----------------------------------------------------------------------------
# Trust policy allows the Lambda service to assume this role. An inline policy
# grants the minimum permissions required to read objects from the documents
# bucket, start and monitor Bedrock ingestion jobs, invoke the Titan embedding
# model, and emit logs to CloudWatch.
resource "aws_iam_role" "auto_ingestion_lambda_role" {
  name = "${var.project_name}-${var.environment}-auto-ingestion-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${var.environment}-auto-ingestion-lambda-role"
  })
}

# -----------------------------------------------------------------------------
# IAM Role Policy: inline permissions for the Lambda execution role
# -----------------------------------------------------------------------------
# Grants:
#   - s3:GetObject on the documents bucket and bucket/*
#   - bedrock-agent:StartIngestionJob on the KB ARN
#   - bedrock-agent:GetIngestionJob on the KB ARN
#   - bedrock:InvokeModel on the Titan embedding model ARN
#   - logs:CreateLogGroup, logs:CreateLogStream, logs:PutLogEvents
resource "aws_iam_role_policy" "auto_ingestion_lambda_policy" {
  name = "${var.project_name}-${var.environment}-auto-ingestion-lambda-policy"
  role = aws_iam_role.auto_ingestion_lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ReadDocumentsFromS3"
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = [
          aws_s3_bucket.documents.arn,
          "${aws_s3_bucket.documents.arn}/*"
        ]
      },
      {
        Sid    = "StartAndGetIngestionJob"
        Effect = "Allow"
        Action = [
          "bedrock:StartIngestionJob",
          "bedrock:GetIngestionJob"
        ]
        Resource = [
          aws_bedrockagent_knowledge_base.kb.arn,
          "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:data-source/*"
        ]
      },
      {
        Sid    = "InvokeEmbeddingModel"
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel"
        ]
        Resource = local.embedding_model_arn
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

# AOSS data access policy propagation sleep (30s)
# Gives the OpenSearch Serverless data access policy time to propagate
# before the Bedrock Knowledge Base attempts to access the collection.
resource "time_sleep" "aoss_policy_propagation" {
  create_duration = "60s"

  depends_on = [
    aws_opensearchserverless_access_policy.data,
    aws_opensearchserverless_collection.main
  ]
}

# Propagation sleep for index creation — re-runs when data access policy changes
# The triggers block forces re-creation whenever the policy version changes,
# ensuring propagation delay before the opensearch_index resource attempts creation.
resource "time_sleep" "aoss_index_propagation" {
  create_duration = "30s"

  triggers = {
    policy_version = aws_opensearchserverless_access_policy.data.policy_version
  }

  depends_on = [
    aws_opensearchserverless_access_policy.data
  ]
}