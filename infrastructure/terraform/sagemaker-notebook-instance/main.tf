data "aws_caller_identity" "current" {}
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.3.0"
}

provider "aws" {
  region = "us-east-1"
}

# Data source for the current AWS account and caller identity
locals {
  tags = {
    Environment = "dev"
    ManagedBy   = "terraform"
  }
}

# IAM execution role for SageMaker Notebook Instance
resource "aws_iam_role" "sagemaker_execution_role" {
  name = "sagemaker-notebook-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })

  tags = local.tags
}

resource "aws_iam_role_policy" "sagemaker_execution_policy" {
  name = "sagemaker-notebook-execution-policy"
  role = aws_iam_role.sagemaker_execution_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = "*"
      },
      {
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

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# KMS key for SageMaker notebook encryption at rest
resource "aws_kms_key" "sagemaker_notebook" {
  description             = "KMS key for SageMaker notebook instance encryption"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "Enable IAM User Permissions"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "Allow SageMaker to use the key"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = [
          "kms:Encrypt",
          "kms:Decrypt",
          "kms:ReEncrypt*",
          "kms:GenerateDataKey*",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })

  tags = local.tags
}

# SageMaker Notebook Instance
resource "aws_sagemaker_notebook_instance" "main" {
  name                   = "ml-g5-2xlarge-notebook"
  role_arn               = aws_iam_role.sagemaker_execution_role.arn
  instance_type          = "ml.g5.2xlarge"
  volume_size            = 50
  kms_key_id             = aws_kms_key.sagemaker_notebook.arn
  direct_internet_access = "Enabled"
  instance_metadata_service_configuration {
    minimum_instance_metadata_service_version = "2"
  }

  tags = local.tags
}

output "sagemaker_notebook_instance_name" {
  description = "Name of the SageMaker Notebook Instance"
  value       = aws_sagemaker_notebook_instance.main.name
}

output "sagemaker_notebook_instance_url" {
  description = "URL of the SageMaker Notebook Instance"
  value       = aws_sagemaker_notebook_instance.main.url
}

output "sagemaker_execution_role_arn" {
  description = "ARN of the SageMaker execution IAM role"
  value       = aws_iam_role.sagemaker_execution_role.arn
}
