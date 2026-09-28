"""Tests for webhook endpoint + debounce logic.

These test the FastAPI webhook handler with mocked Redis.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from tests.conftest import (
    SAMPLE_CLOSED_PAYLOAD,
    SAMPLE_PR_PAYLOAD,
    SAMPLE_SYNC_PAYLOAD,
    compute_webhook_signature,
)


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app, raise_server_exceptions=False)


def _make_headers(payload: dict, secret: str = "dev-secret", event: str = "pull_request"):
    """Build valid webhook headers."""
    body = json.dumps(payload)
    return {
        "X-Hub-Signature-256": compute_webhook_signature(payload, secret),
        "X-GitHub-Event": event,
        "Content-Type": "application/json",
    }


class TestWebhookEndpoint:
    """Test the POST /webhook/github endpoint."""

    @patch("src.webhook.handler.get_redis_pool")
    @patch("src.webhook.handler.handle_push_webhook")
    def test_valid_pr_opened(self, mock_debounce, mock_redis, client):
        """Valid PR opened event should return 202 and enqueue."""
        mock_redis.return_value = AsyncMock()
        mock_debounce.return_value = None

        headers = _make_headers(SAMPLE_PR_PAYLOAD)
        body = json.dumps(SAMPLE_PR_PAYLOAD)

        response = client.post("/webhook/github", content=body, headers=headers)

        assert response.status_code == 202
        assert "enqueued" in response.text.lower() or response.status_code == 202

    @patch("src.webhook.handler.get_redis_pool")
    @patch("src.webhook.handler.handle_push_webhook")
    def test_synchronize_event_accepted(self, mock_debounce, mock_redis, client):
        """PR synchronize (new push) should also be accepted."""
        mock_redis.return_value = AsyncMock()
        mock_debounce.return_value = None

        headers = _make_headers(SAMPLE_SYNC_PAYLOAD)
        body = json.dumps(SAMPLE_SYNC_PAYLOAD)

        response = client.post("/webhook/github", content=body, headers=headers)
        assert response.status_code == 202

    def test_missing_signature_rejected(self, client):
        """Request without X-Hub-Signature-256 should get 403."""
        body = json.dumps(SAMPLE_PR_PAYLOAD)
        headers = {
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        }

        response = client.post("/webhook/github", content=body, headers=headers)
        assert response.status_code == 403

    def test_invalid_signature_rejected(self, client):
        """Request with wrong signature should get 403."""
        body = json.dumps(SAMPLE_PR_PAYLOAD)
        headers = {
            "X-Hub-Signature-256": "sha256=invalid_signature_here",
            "X-GitHub-Event": "pull_request",
            "Content-Type": "application/json",
        }

        response = client.post("/webhook/github", content=body, headers=headers)
        assert response.status_code == 403

    def test_non_pr_event_ignored(self, client):
        """Non pull_request events should return 202 (ignored)."""
        headers = _make_headers(SAMPLE_PR_PAYLOAD, event="push")
        body = json.dumps(SAMPLE_PR_PAYLOAD)

        response = client.post("/webhook/github", content=body, headers=headers)
        assert response.status_code == 202
        assert "ignored" in response.text.lower()

    @patch("src.webhook.handler.get_redis_pool")
    @patch("src.webhook.handler.handle_push_webhook")
    def test_closed_pr_ignored(self, mock_debounce, mock_redis, client):
        """Closed PR action should be ignored (not enqueued)."""
        mock_redis.return_value = AsyncMock()

        headers = _make_headers(SAMPLE_CLOSED_PAYLOAD)
        body = json.dumps(SAMPLE_CLOSED_PAYLOAD)

        response = client.post("/webhook/github", content=body, headers=headers)
        assert response.status_code == 202
        assert "ignored" in response.text.lower()
        mock_debounce.assert_not_called()


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_check(self, client):
        """Health endpoint should return 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
