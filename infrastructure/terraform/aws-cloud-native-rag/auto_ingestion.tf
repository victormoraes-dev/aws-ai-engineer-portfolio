# -----------------------------------------------------------------------------
# File: auto_ingestion.tf
# -----------------------------------------------------------------------------
# This file adds the event-driven auto-ingestion layer to the existing RAG
# infrastructure.
#
# Flow:
#   S3 upload (console / web app / CLI)
#     -> S3 event notification (s3:ObjectCreated:*)
#       -> Lambda function (lambda_function.py)
#         -> Bedrock StartIngestionJob on the Knowledge Base data source
#
# Non-technical users can now simply upload a PDF to the documents S3 bucket
# and ingestion starts automatically - no CLI commands, no manual sync step.
#
# Exam relevance (MLA-C01 - Domain 3):
#   - Event-driven ML workflows
#   - Lambda orchestration of ML/data pipelines
#   - S3 event notifications triggering downstream processing
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# CloudWatch Log Group: Lambda function logs (14-day retention)
# -----------------------------------------------------------------------------
# Pre-created so the Lambda function does not auto-create a log group with
# infinite retention. Retention is set to 14 days to control cost.
resource "aws_cloudwatch_log_group" "auto_ingestion_lambda_logs" {
  name              = "/aws/lambda/${local.lambda_function_name}"
  retention_in_days = 14

  tags = merge(local.common_tags, {
    Name = "${var.project_name}-${var.environment}-auto-ingestion-lambda-logs"
  })
}

# -----------------------------------------------------------------------------
# Lambda Function: event-driven auto-ingestion orchestrator
# -----------------------------------------------------------------------------
# Triggered by S3 event notifications on the documents bucket. Reads the
# uploaded object metadata, then calls Bedrock StartIngestionJob to sync the
# new content into the Knowledge Base data source.
resource "aws_lambda_function" "auto_ingestion" {
  filename         = data.archive_file.lambda_zip.output_path
  function_name    = local.lambda_function_name
  role             = aws_iam_role.auto_ingestion_lambda_role.arn
  handler          = "lambda_function.lambda_handler"
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  runtime          = "python3.11"
  timeout          = 300

  environment {
    variables = {
      KNOWLEDGE_BASE_ID = aws_bedrockagent_knowledge_base.kb.id
      DATA_SOURCE_ID    = aws_bedrockagent_data_source.s3_source.data_source_id
      REGION            = var.region
    }
  }

  depends_on = [
    aws_iam_role_policy.auto_ingestion_lambda_policy,
    aws_cloudwatch_log_group.auto_ingestion_lambda_logs
  ]
}

# -----------------------------------------------------------------------------
# Lambda Permission: allow S3 to invoke the Lambda function
# -----------------------------------------------------------------------------
# Required so the S3 service principal is authorized to invoke the Lambda
# function via the bucket notification configuration.
resource "aws_lambda_permission" "allow_s3_invoke" {
  statement_id   = "AllowExecutionFromS3Bucket"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.auto_ingestion.function_name
  principal      = "s3.amazonaws.com"
  source_arn     = aws_s3_bucket.documents.arn
  source_account = data.aws_caller_identity.current.account_id
}

# -----------------------------------------------------------------------------
# S3 Bucket Notification: trigger Lambda on object creation
# -----------------------------------------------------------------------------
# Sends an event to the Lambda function whenever a new object is created under
# the "documents/" prefix in the documents bucket. This is the entry point of
# the event-driven ingestion workflow.
resource "aws_s3_bucket_notification" "documents_notification" {
  bucket = aws_s3_bucket.documents.id

  lambda_function {
    lambda_function_arn = aws_lambda_function.auto_ingestion.arn
    events              = ["s3:ObjectCreated:*"]
    filter_prefix       = "documents/"
    filter_suffix       = ".pdf"
  }

  depends_on = [
    aws_lambda_permission.allow_s3_invoke,
    aws_lambda_function.auto_ingestion
  ]
}

# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------
output "auto_ingestion_lambda_arn" {
  description = "ARN of the event-driven auto-ingestion Lambda function."
  value       = aws_lambda_function.auto_ingestion.arn
}

output "s3_notification_status" {
  description = "Indicates that the S3 bucket notification for auto-ingestion is configured."
  value       = "configured"
}
