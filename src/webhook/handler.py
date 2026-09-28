"""GitHub webhook handler — receives PR events, parses, and enqueues for review.

Flow (§5):
  1. Verify HMAC signature
  2. Filter to actionable events (opened, synchronize, reopened)
  3. Extract PRMetadata from payload
  4. Call handle_push_webhook() to debounce + enqueue
  5. Return 202 immediately
"""

import json

import structlog
from fastapi import APIRouter, Depends, Header, Request, Response

from src.models.schemas import WebhookPullRequestPayload
from src.queue.debounce import handle_push_webhook
from src.queue.redis_pool import get_redis_pool
from src.webhook.hmac_verify import verify_github_signature

logger = structlog.get_logger()

router = APIRouter(prefix="/webhook", tags=["webhook"])

# PR actions we care about — ignore closed, edited, labeled, etc.
ACTIONABLE_PR_ACTIONS = {"opened", "synchronize", "reopened"}


@router.post("/github", status_code=202)
async def github_webhook(
    request: Request,
    body: bytes = Depends(verify_github_signature),
    x_github_event: str = Header(default=""),
):
    """Receive GitHub webhook events.

    - Verifies HMAC signature (via dependency)
    - Filters to `pull_request` events with actionable actions
    - Parses payload → PRMetadata
    - Enqueues deferred review job via arq (with debounce)
    - Returns 202 Accepted immediately (async processing)
    """
    # --- Filter to events we handle ---
    if x_github_event != "pull_request":
        logger.debug("webhook.ignored_event", event=x_github_event)
        return Response(status_code=202, content="Event ignored")

    # --- Parse payload ---
    try:
        payload_dict = json.loads(body)
        payload = WebhookPullRequestPayload(**payload_dict)
    except Exception:
        logger.exception("webhook.parse_error")
        return Response(status_code=400, content="Invalid payload")

    # --- Filter to actionable PR actions ---
    if payload.action not in ACTIONABLE_PR_ACTIONS:
        logger.debug(
            "webhook.ignored_action",
            action=payload.action,
            repo=payload.repository.full_name,
            pr=payload.number,
        )
        return Response(status_code=202, content="Action ignored")

    # --- Extract metadata ---
    pr_meta = payload.to_pr_metadata()

    logger.info(
        "webhook.received",
        repo=pr_meta.repo,
        pr=pr_meta.pr_number,
        action=payload.action,
        head_sha=pr_meta.head_sha,
        author=pr_meta.author,
    )

    # --- Enqueue with debounce ---
    redis = await get_redis_pool()
    await handle_push_webhook(
        redis=redis,
        repo=pr_meta.repo,
        pr=pr_meta.pr_number,
        sha=pr_meta.head_sha,
    )

    logger.info(
        "webhook.enqueued",
        repo=pr_meta.repo,
        pr=pr_meta.pr_number,
        sha=pr_meta.head_sha,
    )

    return Response(
        status_code=202,
        content=f"Review enqueued for {pr_meta.repo}#{pr_meta.pr_number}",
    )
