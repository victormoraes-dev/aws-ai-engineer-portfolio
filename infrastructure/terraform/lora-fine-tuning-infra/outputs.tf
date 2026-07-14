
# ──────────────────────────────────────────────
# Outputs
# ──────────────────────────────────────────────

output "s3_bucket" {
  description = "S3 bucket for training data and artifacts"
  value       = aws_s3_bucket.loRa_ml_artifacts.bucket
}

output "region" {
  description = "AWS region"
  value       = var.aws_region
}
