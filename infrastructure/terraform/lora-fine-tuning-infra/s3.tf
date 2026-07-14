
# ──────────────────────────────────────────────
# S3 Bucket for training data and model artifacts
# ──────────────────────────────────────────────

resource "aws_s3_bucket" "loRa_ml_artifacts" {
  bucket        = "${var.project_name}-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_versioning" "loRa_ml_artifacts" {
  bucket = aws_s3_bucket.loRa_ml_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "loRa_ml_artifacts" {
  bucket = aws_s3_bucket.loRa_ml_artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "loRa_ml_artifacts" {
  bucket = aws_s3_bucket.loRa_ml_artifacts.id

  rule {
    id     = "expire-model-artifacts"
    status = "Enabled"

    filter {
      prefix = "output/"
    }

    expiration {
      days = 30
    }
  }
}
