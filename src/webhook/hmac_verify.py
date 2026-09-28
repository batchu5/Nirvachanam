"""HMAC-SHA256 signature verification for GitHub webhooks.

GitHub sends an `X-Hub-Signature-256` header with every webhook delivery.
We verify it against our webhook secret to ensure the payload is authentic
and hasn't been tampered with in transit.

See: https://docs.github.com/en/webhooks/using-webhooks/validating-webhook-deliveries
"""

import hashlib
import hmac

from fastapi import HTTPException, Request

from src.config import settings


async def verify_github_signature(request: Request) -> bytes:
    """FastAPI dependency that verifies the GitHub webhook HMAC signature.

    Reads the raw request body, computes HMAC-SHA256 with the configured
    webhook secret, and compares it to the X-Hub-Signature-256 header
    using constant-time comparison.

    Returns:
        The raw request body bytes (for downstream parsing).

    Raises:
        HTTPException(403): If signature is missing or invalid.
    """
    signature_header = request.headers.get("X-Hub-Signature-256")
    if not signature_header:
        raise HTTPException(status_code=403, detail="Missing X-Hub-Signature-256 header")

    body = await request.body()

    if not verify_signature(body, settings.github_webhook_secret, signature_header):
        raise HTTPException(status_code=403, detail="Invalid webhook signature")

    return body


def verify_signature(payload: bytes, secret: str, signature_header: str) -> bool:
    """Verify HMAC-SHA256 signature.

    Args:
        payload: Raw request body bytes.
        secret: The webhook secret configured in the GitHub App.
        signature_header: Value of X-Hub-Signature-256 header (e.g. "sha256=abc123...").

    Returns:
        True if the signature is valid.
    """
    if not signature_header.startswith("sha256="):
        return False

    expected_signature = signature_header[7:]  # Strip "sha256=" prefix

    computed = hmac.new(
        key=secret.encode("utf-8"),
        msg=payload,
        digestmod=hashlib.sha256,
    ).hexdigest()

    return hmac.compare_digest(computed, expected_signature)
