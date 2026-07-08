# =============================================================================
# File: versions.tf
# =============================================================================
# Terraform and provider version constraints plus the default AWS provider
# configuration. The AWS provider is pinned to >= 5.50.0 to ensure support for
# OpenSearch Serverless resources, Bedrock Knowledge Base resources, and the
# latest IAM/data-source behaviors.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.50.0"
    }
    null = {
      source  = "hashicorp/null"
      version = ">= 3.2.0"
    }
    time = {
      source  = "hashicorp/time"
      version = ">= 0.9.0"
    }
    random = {
      source  = "hashicorp/random"
      version = ">= 3.5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = ">= 2.4.0"
    }
    opensearch = {
      source  = "opensearch-project/opensearch"
      version = ">= 2.2.0"
    }
  }
}

provider "aws" {
  region = var.region
}

provider "opensearch" {
  url                   = aws_opensearchserverless_collection.main.collection_endpoint
  healthcheck           = false
  aws_region            = var.region
  aws_signature_service = "aoss"
  sign_aws_requests     = true
}
