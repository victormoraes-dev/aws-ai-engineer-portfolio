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
