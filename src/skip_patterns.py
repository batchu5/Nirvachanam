"""Shared skip patterns for files that should never be sent to the LLM.

Kept in a dependency-free module so both the diff parser and the
deterministic file-selection stage use one list and one matcher.
"""

from __future__ import annotations

import fnmatch

DEFAULT_SKIP_PATTERNS: list[str] = [
    # Lock files & minified assets
    "*.lock",
    "*.min.js",
    "*.min.css",
    "*.map",
    # Generated code
    "*.generated.*",
    "*.pb.go",
    "*_pb2.py",
    "*_pb2_grpc.py",
    # Vendor / dependency directories
    "vendor/*",
    "node_modules/*",
    ".venv/*",
    "venv/*",
    # Snapshots & fixtures
    "__snapshots__/*",
    # Binary-like assets
    "*.svg",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.bmp",
    "*.webp",
    # Fonts
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.otf",
    # Compiled / cache
    "*.pyc",
    "__pycache__/*",
    "*.class",
    "*.o",
    "*.so",
    "*.dll",
    "*.exe",
    # Database migrations (often auto-generated)
    "migrations/*.py",
    # Package-manager lockfiles that don't end in .lock
    "package-lock.json",
    "npm-shrinkwrap.json",
    "*-lock.json",
    "*-lock.yaml",
    "pnpm-lock.yaml",
    "bun.lockb",
    "go.sum",
]


def matches_glob(path: str, pattern: str) -> bool:
    """Match a repo-relative path against a skip glob, anywhere in the tree.

    Plain ``fnmatch`` only matches from the repo root, so ``package-lock.json``
    would miss ``web/package-lock.json`` and ``vendor/*`` would miss
    ``services/api/vendor/x.go``. This matcher also tries the file name and
    every trailing sub-path, case-insensitively.
    """
    norm = path.replace("\\", "/").lstrip("/").lower()
    pat = pattern.lower()
    segments = norm.split("/")
    for i in range(len(segments)):
        if fnmatch.fnmatch("/".join(segments[i:]), pat):
            return True
    return False


def first_matching_pattern(path: str, patterns: list[str]) -> str | None:
    """Return the first pattern that matches ``path``, or None."""
    for pattern in patterns:
        if matches_glob(path, pattern):
            return pattern
    return None
