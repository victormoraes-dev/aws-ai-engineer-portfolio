# ──────────────────────────────────────────────
# Locals
# ──────────────────────────────────────────────

data "aws_caller_identity" "current" {}

locals {
  tags = {
    Environment = "lab"
    ManagedBy   = "terraform"
    Project     = "mla-c01-lora"
  }
}
