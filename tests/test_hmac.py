"""Tests for HMAC webhook signature verification."""

import json

from src.webhook.hmac_verify import verify_signature
from tests.conftest import SAMPLE_PR_PAYLOAD, compute_webhook_signature


class TestVerifySignature:
    """Test the HMAC-SHA256 verification logic."""

    def test_valid_signature_passes(self):
        """A correctly signed payload should pass verification."""
        secret = "test-secret-123"
        body = json.dumps(SAMPLE_PR_PAYLOAD).encode()
        sig = compute_webhook_signature(SAMPLE_PR_PAYLOAD, secret)

        assert verify_signature(body, secret, sig) is True

    def test_invalid_signature_rejected(self):
        """A payload with wrong signature should be rejected."""
        body = json.dumps(SAMPLE_PR_PAYLOAD).encode()

        assert verify_signature(body, "correct-secret", "sha256=deadbeef123456") is False

    def test_wrong_secret_rejected(self):
        """Signing with wrong secret should fail verification."""
        body = json.dumps(SAMPLE_PR_PAYLOAD).encode()
        sig = compute_webhook_signature(SAMPLE_PR_PAYLOAD, "wrong-secret")

        assert verify_signature(body, "correct-secret", sig) is False

    def test_missing_sha256_prefix_rejected(self):
        """Signature without 'sha256=' prefix should be rejected."""
        body = json.dumps(SAMPLE_PR_PAYLOAD).encode()

        assert verify_signature(body, "secret", "invalid-format-no-prefix") is False

    def test_empty_body(self):
        """Empty body should still produce a valid HMAC (for edge case handling)."""
        secret = "test-secret"
        body = b""
        sig = compute_webhook_signature("", secret)

        assert verify_signature(body, secret, sig) is True

    def test_different_payloads_different_signatures(self):
        """Two different payloads should produce different signatures."""
        secret = "test-secret"
        sig1 = compute_webhook_signature({"a": 1}, secret)
        sig2 = compute_webhook_signature({"b": 2}, secret)

        assert sig1 != sig2
