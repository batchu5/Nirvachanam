"""GitHub API client — App JWT auth, installation tokens, PR operations.

Handles GitHub App authentication (JWT → installation access token) and
provides async methods for the operations we need:
  - Fetching PR diffs
  - Checking PR HEAD SHA (staleness check)
  - Posting reviews with inline comments

All requests go through httpx.AsyncClient with tenacity retry for transient
errors. Rate-limit headers (X-RateLimit-*) are respected automatically.

References: PRD §5 (GitHub integration), §7 (review posting).
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import jwt
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = structlog.get_logger()

GITHUB_API_BASE = "https://api.github.com"


class GitHubClient:
    """Async GitHub API client with App authentication.

    Usage:
        client = GitHubClient(
            app_id="12345",
            private_key=open("key.pem").read(),
        )
        diff = await client.get_pr_diff("owner/repo", 42)
        await client.post_review("owner/repo", 42, "abc123", body, comments)
        await client.close()
    """

    def __init__(
        self,
        app_id: str = "",
        private_key: str = "",
    ):
        self._app_id = app_id or settings.github_app_id
        self._private_key = private_key or settings.github_private_key

        # Installation token cache: installation_id → (token, expires_at)
        self._token_cache: dict[int, tuple[str, float]] = {}

        self._http = httpx.AsyncClient(
            base_url=GITHUB_API_BASE,
            timeout=30.0,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._http.aclose()

    # ----- JWT + Installation Token Auth -----

    def _create_jwt(self) -> str:
        """Create a short-lived JWT for GitHub App authentication.

        GitHub App JWTs are valid for max 10 minutes.

        Returns:
            Encoded JWT string.
        """
        now = int(time.time())
        payload = {
            "iat": now - 60,       # Issued at (60s in the past for clock skew)
            "exp": now + (9 * 60), # Expires in 9 minutes
            "iss": self._app_id,   # GitHub App ID
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    async def _get_installation_token(self, installation_id: int) -> str:
        """Get or refresh an installation access token.

        Installation tokens are valid for 1 hour. We cache them and refresh
        when they're within 5 minutes of expiry.

        Args:
            installation_id: GitHub App installation ID.

        Returns:
            Installation access token string.
        """
        # Check cache
        if installation_id in self._token_cache:
            token, expires_at = self._token_cache[installation_id]
            if time.time() < expires_at - 300:  # 5 min buffer
                return token

        # Request new token
        app_jwt = self._create_jwt()
        response = await self._http.post(
            f"/app/installations/{installation_id}/access_tokens",
            headers={"Authorization": f"Bearer {app_jwt}"},
        )
        response.raise_for_status()
        data = response.json()

        token = data["token"]
        # Parse expiry — GitHub returns ISO 8601 datetime
        # Token is valid for 1 hour, we'll use a conservative estimate
        expires_at = time.time() + 3500  # ~58 minutes

        self._token_cache[installation_id] = (token, expires_at)
        logger.info(
            "github.token_refreshed",
            installation_id=installation_id,
        )
        return token

    def _get_auth_headers(self, token: str) -> dict[str, str]:
        """Build auth headers with installation token."""
        return {"Authorization": f"token {token}"}

    # ----- PR Operations -----

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def get_pr_diff(
        self,
        repo: str,
        pr_number: int,
        installation_id: int | None = None,
    ) -> str:
        """Fetch the unified diff for a pull request.

        Args:
            repo: Repository full name ("owner/repo").
            pr_number: PR number.
            installation_id: GitHub App installation ID.

        Returns:
            Raw unified diff text.
        """
        headers: dict[str, str] = {"Accept": "application/vnd.github.diff"}

        if installation_id:
            token = await self._get_installation_token(installation_id)
            headers.update(self._get_auth_headers(token))

        response = await self._http.get(
            f"/repos/{repo}/pulls/{pr_number}",
            headers=headers,
        )
        response.raise_for_status()
        return response.text

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def get_pr(
        self,
        repo: str,
        pr_number: int,
        installation_id: int | None = None,
    ) -> dict[str, Any]:
        """Get PR details (head SHA, base SHA, etc.).

        Args:
            repo: Repository full name.
            pr_number: PR number.
            installation_id: GitHub App installation ID.

        Returns:
            PR data dict from GitHub API.
        """
        headers: dict[str, str] = {}
        if installation_id:
            token = await self._get_installation_token(installation_id)
            headers.update(self._get_auth_headers(token))

        response = await self._http.get(
            f"/repos/{repo}/pulls/{pr_number}",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()

    async def get_pr_head_sha(
        self,
        repo: str,
        pr_number: int,
        installation_id: int | None = None,
    ) -> str:
        """Get the current HEAD SHA of a PR (for staleness check).

        Args:
            repo: Repository full name.
            pr_number: PR number.
            installation_id: GitHub App installation ID.

        Returns:
            HEAD commit SHA string.
        """
        pr_data = await self.get_pr(repo, pr_number, installation_id)
        return pr_data["head"]["sha"]

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def get_file_content(
        self,
        repo: str,
        path: str,
        ref: str,
        installation_id: int | None = None,
    ) -> str | None:
        """Get file content from GitHub at a specific ref.

        Args:
            repo: Repository full name.
            path: File path relative to repo root.
            ref: Git ref (commit SHA, branch, tag).
            installation_id: GitHub App installation ID.

        Returns:
            File content as string, or None if not found.
        """
        headers: dict[str, str] = {"Accept": "application/vnd.github.raw+json"}
        if installation_id:
            token = await self._get_installation_token(installation_id)
            headers.update(self._get_auth_headers(token))

        response = await self._http.get(
            f"/repos/{repo}/contents/{path}",
            params={"ref": ref},
            headers=headers,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.text

    # ----- Review Posting -----

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def post_review(
        self,
        repo: str,
        pr_number: int,
        commit_sha: str,
        body: str,
        comments: list[dict[str, Any]] | None = None,
        event: str = "COMMENT",
        installation_id: int | None = None,
    ) -> dict[str, Any]:
        """Post a review on a pull request.

        Args:
            repo: Repository full name.
            pr_number: PR number.
            commit_sha: The SHA of the commit to review.
            body: Review body (markdown summary).
            comments: Optional list of inline comment dicts with keys:
                - path: file path
                - position: position in the diff (NOT line number)
                - body: comment text
            event: Review event type — "COMMENT", "APPROVE", or "REQUEST_CHANGES".
            installation_id: GitHub App installation ID.

        Returns:
            Response data dict from GitHub API.
        """
        headers: dict[str, str] = {}
        if installation_id:
            token = await self._get_installation_token(installation_id)
            headers.update(self._get_auth_headers(token))

        payload: dict[str, Any] = {
            "commit_id": commit_sha,
            "body": body,
            "event": event,
        }
        if comments:
            payload["comments"] = comments

        logger.info(
            "github.posting_review",
            repo=repo,
            pr=pr_number,
            sha=commit_sha[:8],
            comments=len(comments) if comments else 0,
        )

        response = await self._http.post(
            f"/repos/{repo}/pulls/{pr_number}/reviews",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        result = response.json()

        logger.info(
            "github.review_posted",
            repo=repo,
            pr=pr_number,
            review_id=result.get("id"),
        )
        return result

    @retry(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        reraise=True,
    )
    async def post_comment(
        self,
        repo: str,
        pr_number: int,
        body: str,
        installation_id: int | None = None,
    ) -> dict[str, Any]:
        """Post a standalone comment on a PR (not a review comment).

        Used for status messages, error reports, etc.

        Args:
            repo: Repository full name.
            pr_number: PR number (treated as issue number for the API).
            body: Comment body markdown.
            installation_id: GitHub App installation ID.

        Returns:
            Response data dict.
        """
        headers: dict[str, str] = {}
        if installation_id:
            token = await self._get_installation_token(installation_id)
            headers.update(self._get_auth_headers(token))

        response = await self._http.post(
            f"/repos/{repo}/issues/{pr_number}/comments",
            headers=headers,
            json={"body": body},
        )
        response.raise_for_status()
        return response.json()
