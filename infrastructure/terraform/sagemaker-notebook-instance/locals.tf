# Data source for the current AWS account and caller identity
locals {
  tags = {
    Environment = "lab"
    ManagedBy   = "terraform"
  }
}

data "aws_caller_identity" "current" {}
