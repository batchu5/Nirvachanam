"""Enumerations used across the review pipeline."""

from enum import StrEnum


class Severity(StrEnum):
    """Finding severity levels — maps to GitHub review comment styling."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Category(StrEnum):
    """Agent categories — each specialist agent owns one category."""
    BUG = "bug"
    SECURITY = "security"
    STYLE = "style"
    TEST = "test"


class ReviewStatus(StrEnum):
    """Review lifecycle states stored in the `reviews` table."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEGRADED = "degraded"
