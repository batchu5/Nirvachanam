"""Shared test fixtures — sample payloads, fake Redis, etc."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest


# ---------------------------------------------------------------------------
# Sample GitHub webhook payloads
# ---------------------------------------------------------------------------

SAMPLE_PR_PAYLOAD = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "title": "Fix null pointer in user service",
        "head": {"sha": "abc123def456", "ref": "fix/null-pointer"},
        "base": {"sha": "main789xyz", "ref": "main"},
        "user": {"login": "testuser"},
    },
    "repository": {"full_name": "testorg/testrepo"},
    "installation": {"id": 12345},
}

SAMPLE_SYNC_PAYLOAD = {
    **SAMPLE_PR_PAYLOAD,
    "action": "synchronize",
    "pull_request": {
        **SAMPLE_PR_PAYLOAD["pull_request"],
        "head": {"sha": "newsha999", "ref": "fix/null-pointer"},
    },
}

SAMPLE_CLOSED_PAYLOAD = {
    **SAMPLE_PR_PAYLOAD,
    "action": "closed",
}


# ---------------------------------------------------------------------------
# Sample unified diff
# ---------------------------------------------------------------------------

SAMPLE_DIFF = """\
diff --git a/src/auth/login.py b/src/auth/login.py
index 1234567..abcdefg 100644
--- a/src/auth/login.py
+++ b/src/auth/login.py
@@ -10,6 +10,8 @@ def authenticate(username: str, password: str) -> bool:
     if not username or not password:
         raise ValueError("Username and password required")
 
+    # BUG: SQL injection vulnerability
+    query = f"SELECT * FROM users WHERE username = '{username}'"
     connection = get_db_connection()
     result = connection.execute(query)
 
@@ -25,4 +27,5 @@ def logout(session_id: str) -> None:
     sessions.pop(session_id, None)
     logger.info(f"User logged out: {session_id}")
+    # TODO: invalidate JWT token
 """

SAMPLE_DIFF_WITH_SECRETS = """\
diff --git a/config.py b/config.py
--- a/config.py
+++ b/config.py
@@ -1,3 +1,5 @@
 import os
 
+API_KEY = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
+AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
 DATABASE_URL = os.getenv("DB_URL")
"""

SAMPLE_EMPTY_DIFF = ""

SAMPLE_LOCKFILE_DIFF = """\
diff --git a/package-lock.json b/package-lock.json
index 1111111..2222222 100644
--- a/package-lock.json
+++ b/package-lock.json
@@ -1,3 +1,3 @@
 {
-  "version": "1.0.0"
+  "version": "1.0.1"
 }
"""


# ---------------------------------------------------------------------------
# Helper to compute HMAC signature
# ---------------------------------------------------------------------------

def compute_webhook_signature(payload: dict | str, secret: str = "dev-secret") -> str:
    """Compute X-Hub-Signature-256 header value for a webhook payload."""
    if isinstance(payload, dict):
        body = json.dumps(payload).encode()
    else:
        body = payload.encode()
    sig = hmac.new(
        key=secret.encode(),
        msg=body,
        digestmod=hashlib.sha256,
    ).hexdigest()
    return f"sha256={sig}"
