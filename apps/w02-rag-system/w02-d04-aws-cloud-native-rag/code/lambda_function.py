# =============================================================================
# AWS Lambda: Event-Driven Auto-Ingestion for Amazon Bedrock Knowledge Base
# -----------------------------------------------------------------------------
# WHAT THIS LAMBDA DOES:
#   Automatically starts an Amazon Bedrock Knowledge Base ingestion job whenever
#   a new document is uploaded to the S3 bucket configured as the KB data source.
#   This implements the event-driven auto-ingestion pattern for a cloud-native
#   Retrieval-Augmented Generation (RAG) pipeline.
#
# TRIGGER:
#   S3 event notification on the "s3:ObjectCreated:*" event type. The S3 bucket
#   must be configured to send ObjectCreated events to this Lambda function.
#
# REQUIRED ENVIRONMENT VARIABLES:
#   KNOWLEDGE_BASE_ID : The identifier of the Bedrock Knowledge Base (e.g., "ABCDE12345")
#   DATA_SOURCE_ID    : The identifier of the KB data source to ingest from (e.g., "FGHIJ67890")
#   REGION            : AWS region where the Bedrock Knowledge Base lives (e.g., "us-east-1")
#
# REQUIRED IAM PERMISSIONS (Lambda execution role):
#   - s3:GetObject            (on the source bucket ARN, e.g., arn:aws:s3:::my-rag-bucket/*)
#   - bedrock-agent:StartIngestionJob
#   - bedrock-agent:GetIngestionJob
#   - bedrock:InvokeModel     (for the embedding model used by the Knowledge Base)
#
# HOW TO DEPLOY:
#   1. Create a deployment directory and add this file (lambda_function.py).
#   2. Install dependencies into the directory:
#        pip install -t . boto3
#   3. Package the function:
#        zip -r function.zip .
#   4. Create the Lambda function (Python 3.11+) and attach the execution role
#      with the permissions listed above.
#   5. Add an S3 trigger on the source bucket for event type "ObjectCreated".
#   6. Set the environment variables KNOWLEDGE_BASE_ID, DATA_SOURCE_ID, REGION.
#
# EXAM RELEVANCE:
#   MLA-C01 Domain 3 — Event-driven ML workflows, Lambda for ML orchestration.
#   Demonstrates decoupled, event-driven ingestion for a RAG pipeline using S3
#   notifications and Bedrock Knowledge Base APIs.
# =============================================================================

import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import ClientError

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# Configuration from environment variables
# -----------------------------------------------------------------------------
KNOWLEDGE_BASE_ID: str = os.environ.get("KNOWLEDGE_BASE_ID", "")
DATA_SOURCE_ID: str = os.environ.get("DATA_SOURCE_ID", "")
REGION: str = os.environ.get("REGION", "us-east-1")

# -----------------------------------------------------------------------------
# Boto3 client (created once at cold-start for reuse across invocations)
# -----------------------------------------------------------------------------
bedrock_agent_client = boto3.client("bedrock-agent", region_name=REGION)


def _extract_s3_objects(event: dict[str, Any]) -> list[dict[str, str]]:
    """
    Extract bucket name and object key from an S3 event notification.

    Handles both single-record and batch (multiple records) S3 events.

    Args:
        event: The S3 event payload delivered to the Lambda handler.

    Returns:
        A list of dictionaries, each containing 'bucket' and 'key' keys.
    """
    records: list[dict[str, str]] = []
    for record in event.get("Records", []):
        s3_info: dict[str, Any] = record.get("s3", {})
        bucket: str = s3_info.get("bucket", {}).get("name", "")
        key: str = s3_info.get("object", {}).get("key", "")
        if bucket and key:
            records.append({"bucket": bucket, "key": key})
        else:
            logger.warning("Skipping malformed S3 record: %s", json.dumps(record))
    return records


def start_ingestion_job() -> str:
    """
    Start a Bedrock Knowledge Base ingestion job for the configured data source.

    This function only starts the job; it does NOT poll for completion because
    Lambda has a 15-minute timeout and ingestion may take significantly longer
    for large document sets.

    Returns:
        The ingestion job ID as a string.

    Raises:
        ClientError: If the Bedrock Agent API call fails.
        ValueError: If required configuration is missing.
    """
    if not KNOWLEDGE_BASE_ID or not DATA_SOURCE_ID:
        raise ValueError(
            "Missing required environment variables: "
            "KNOWLEDGE_BASE_ID and DATA_SOURCE_ID must be set."
        )

    logger.info(
        "Starting Bedrock ingestion job for knowledgeBaseId=%s, dataSourceId=%s",
        KNOWLEDGE_BASE_ID,
        DATA_SOURCE_ID,
    )

    response = bedrock_agent_client.start_ingestion_job(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        dataSourceId=DATA_SOURCE_ID,
        description="Auto-triggered ingestion job from S3 ObjectCreated event.",
    )

    ingestion_job: dict[str, Any] = response.get("ingestionJob", {})
    job_id: str = ingestion_job.get("ingestionJobId", "")

    if not job_id:
        raise RuntimeError("StartIngestionJob response did not contain an ingestionJobId.")

    logger.info("Ingestion job started successfully. jobId=%s", job_id)
    return job_id


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    AWS Lambda entry point for the S3-triggered Bedrock KB ingestion workflow.

    Extracts all S3 objects from the event (supporting batch uploads), starts a
    single Bedrock Knowledge Base ingestion job, and returns the job ID. The job
    is not polled for completion.

    Args:
        event: The S3 event notification payload.
        context: The AWS Lambda runtime context object.

    Returns:
        A dictionary containing:
            - statusCode: HTTP-style status code (200 on success, 500 on error).
            - jobId: The Bedrock ingestion job ID (on success).
            - processedObjects: List of bucket/key pairs that triggered the job.
            - body: A JSON-serializable message string.
    """
    logger.info("Received S3 event: %s", json.dumps(event))

    try:
        s3_objects: list[dict[str, str]] = _extract_s3_objects(event)

        if not s3_objects:
            logger.warning("No valid S3 objects found in event; nothing to ingest.")
            return {
                "statusCode": 200,
                "body": json.dumps("No valid S3 objects found in event."),
                "processedObjects": [],
            }

        logger.info("Processing %d S3 object(s) from event.", len(s3_objects))
        for obj in s3_objects:
            logger.info("Object uploaded -> bucket=%s, key=%s", obj["bucket"], obj["key"])

        job_id: str = start_ingestion_job()

        return {
            "statusCode": 200,
            "jobId": job_id,
            "processedObjects": s3_objects,
            "body": json.dumps(
                f"Started Bedrock ingestion job {job_id} for {len(s3_objects)} object(s)."
            ),
        }

    except ClientError as exc:
        error_code: str = exc.response.get("Error", {}).get("Code", "Unknown")
        error_message: str = exc.response.get("Error", {}).get("Message", str(exc))
        logger.error(
            "ClientError starting ingestion job: code=%s, message=%s",
            error_code,
            error_message,
        )
        return {
            "statusCode": 500,
            "body": json.dumps(f"ClientError: {error_code} - {error_message}"),
        }

    except ValueError as exc:
        logger.error("Configuration error: %s", str(exc))
        return {
            "statusCode": 500,
            "body": json.dumps(f"Configuration error: {exc}"),
        }

    except Exception as exc:  # noqa: BLE001 - top-level Lambda safety net
        logger.exception("Unexpected error during ingestion job start: %s", str(exc))
        return {
            "statusCode": 500,
            "body": json.dumps(f"Unexpected error: {exc}"),
        }
