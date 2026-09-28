"""Milestone 1 Demo Script — POST a sample webhook and verify the flow.

Run with:
    .\.venv\Scripts\Activate.ps1; python scripts/demo_m1.py
"""

import hashlib
import hmac
import json
import sys

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WEBHOOK_SECRET = "2d063722215e99aa60a33a1d1ed1112149d060c6"
API_URL = "http://localhost:8000"

# Sample PR webhook payload (mimics what GitHub sends)
SAMPLE_PAYLOAD = {
    "action": "opened",
    "number": 42,
    "pull_request": {
        "number": 42,
        "title": "feat: add user authentication module",
        "user": {"login": "demo-developer"},
        "head": {
            "sha": "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
            "ref": "feature/auth-module",
        },
        "base": {"ref": "main"},
        "diff_url": "https://github.com/demo-org/demo-repo/pull/42.diff",
        "html_url": "https://github.com/demo-org/demo-repo/pull/42",
    },
    "repository": {"full_name": "demo-org/demo-repo"},
    "installation": {"id": 12345},
}


def compute_signature(payload_bytes: bytes, secret: str) -> str:
    """Compute the HMAC-SHA256 signature like GitHub does."""
    mac = hmac.new(secret.encode(), payload_bytes, hashlib.sha256)
    return f"sha256={mac.hexdigest()}"


def main():
    print("=" * 60)
    print("  Milestone 1 Demo: Webhook -> Parse -> Enqueue")
    print("=" * 60)
    print()

    # Step 1: Health check
    print("[1/4] Checking API health...")
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5)
        health = r.json()
        print(f"  [OK] Health: {health}")
    except Exception as e:
        print(f"  [FAIL] API not reachable: {e}")
        print("  -> Make sure 'docker-compose up -d' is running.")
        sys.exit(1)

    # Step 2: Test HMAC rejection
    print()
    print("[2/4] Testing HMAC rejection (invalid signature)...")
    r = httpx.post(
        f"{API_URL}/webhook/github",
        content=b'{"action":"opened"}',
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": "sha256=invalid",
        },
        timeout=5,
    )
    if r.status_code == 403:
        print(f"  [OK] Correctly rejected: {r.status_code} {r.json()}")
    else:
        print(f"  [FAIL] Expected 403, got {r.status_code}: {r.text}")

    # Step 3: Test non-PR event ignored
    print()
    print("[3/4] Testing non-PR event (push) is ignored...")
    push_payload = json.dumps({"action": "push"}).encode()
    sig = compute_signature(push_payload, WEBHOOK_SECRET)
    r = httpx.post(
        f"{API_URL}/webhook/github",
        content=push_payload,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": sig,
        },
        timeout=5,
    )
    print(f"  [OK] Response: {r.status_code} {r.text}")

    # Step 4: Post a valid PR webhook
    print()
    print("[4/4] Posting valid PR webhook (opened)...")
    payload_bytes = json.dumps(SAMPLE_PAYLOAD).encode()
    signature = compute_signature(payload_bytes, WEBHOOK_SECRET)
    print(f"  Payload: PR #{SAMPLE_PAYLOAD['number']} '{SAMPLE_PAYLOAD['pull_request']['title']}'")
    print(f"  Repo: {SAMPLE_PAYLOAD['repository']['full_name']}")
    print(f"  SHA: {SAMPLE_PAYLOAD['pull_request']['head']['sha'][:8]}...")
    print(f"  Signature: {signature[:30]}...")

    r = httpx.post(
        f"{API_URL}/webhook/github",
        content=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "X-GitHub-Event": "pull_request",
            "X-Hub-Signature-256": signature,
        },
        timeout=10,
    )

    if r.status_code in (200, 202):
        print(f"  [OK] Accepted: {r.status_code} {r.text}")
    else:
        print(f"  [WARN]  Response: {r.status_code} {r.text}")

    # Summary
    print()
    print("=" * 60)
    print("  Demo complete!")
    print("=" * 60)
    print()
    print("Next steps to verify:")
    print("  1. Check API logs:  docker logs pr_review_agent-api-1 --tail 20")
    print("  2. Check Redis:     docker exec pr_review_agent-redis-1 redis-cli KEYS '*'")
    print("  3. Check Worker:    docker logs pr_review_agent-worker-1 --tail 10")


if __name__ == "__main__":
    main()
