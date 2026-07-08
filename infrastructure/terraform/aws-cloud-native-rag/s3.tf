# =============================================================================
# File: s3.tf
# =============================================================================
# Amazon S3 bucket for storing the source PDF. Uses SSE-S3 (AES256) with an
# AWS-owned key for the lab. Public access is fully blocked and object
# ownership is enforced to the bucket owner.
# =============================================================================

# S3 bucket with a unique name. Tags identify the project and environment.
resource "aws_s3_bucket" "documents" {
  bucket = local.bucket_name
  tags   = local.common_tags
}

# Server-side encryption using SSE-S3 (AES256) with an AWS-owned key.
# This avoids the need to provision a customer-managed KMS key for the lab.
resource "aws_s3_bucket_server_side_encryption_configuration" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block all public access to the bucket. Required for least-privilege lab.
resource "aws_s3_bucket_public_access_block" "documents" {
  bucket                  = aws_s3_bucket.documents.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enforce bucket-owner ownership of all uploaded objects (ACLs disabled).
resource "aws_s3_bucket_ownership_controls" "documents" {
  bucket = aws_s3_bucket.documents.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}
