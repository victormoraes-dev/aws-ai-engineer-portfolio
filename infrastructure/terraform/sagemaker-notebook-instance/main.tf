# SageMaker Notebook Instance
resource "aws_sagemaker_notebook_instance" "main" {
  name                   = "ml-g5-2xlarge-notebook"
  role_arn               = aws_iam_role.sagemaker_execution_role.arn
  instance_type          = "ml.g5.2xlarge"
  volume_size            = 200
  kms_key_id             = aws_kms_key.sagemaker_notebook.arn
  direct_internet_access = "Enabled"
  lifecycle_config_name  = aws_sagemaker_notebook_instance_lifecycle_configuration.lifecycle_configuration_lab.name
  instance_metadata_service_configuration {
    minimum_instance_metadata_service_version = "2"
  }

  tags = local.tags
}

resource "aws_sagemaker_notebook_instance_lifecycle_configuration" "lifecycle_configuration_lab" {
  name = "sagemaker-instance-lifecycle-config"

  on_start = base64encode(<<-EOT
    #!/bin/bash
    set -e

    sudo -u ec2-user -i <<'EOF'
    source ~/anaconda3/envs/pytorch//bin/activate JupyterSystemEnv

    pip install --quiet --upgrade \
      boto3 \
      sagemaker \
      datasets \
      sentence-transformers \
      transformers \
      peft \
      accelerate \
      torch \
      torchvision \
      langchain-aws \
      langchain-core \
      bitsandbytes \
      scikit-learn \
      matplotlib \
      pandas \
      trl \
      nn

    source ~/anaconda3/envs/pytorch/bin/deactivate
    EOF
    EOT
  )
}
