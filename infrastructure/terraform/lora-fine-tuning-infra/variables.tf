# ──────────────────────────────────────────────
# Variables
# ──────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "mla-c01-lora"
}

variable "notebook_instance_name" {
  description = "Name of the already-provisioned SageMaker Notebook Instance"
  type        = string
  default     = "ml-g5-2xlarge-notebook"
}
