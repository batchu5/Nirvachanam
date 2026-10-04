"""Rule engine — groups files by which review rules apply to them.

Phase 2 of the OCR architecture migration. This is the core logic that:
  1. Maps file extensions → language families → built-in rule sets
  2. Loads custom project rules from .reviewrules.yaml (if present)
  3. Groups files by identical (source, pattern, rules_text) tuples
  4. Produces RuleGroup objects consumed by the agentic review loop

The key insight from OCR: files that share the same rules text can be
reviewed together in a single LLM call with those rules injected into
the system prompt. This avoids sending Python-specific NPE rules to
JavaScript files, and vice versa.

References: new_architecture.md Phase 2, Alibaba OCR rulegroup.go
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import structlog

from src.agents.file_selection import FileDecision
from src.rules.builtin.python_rules import PYTHON_RULES
from src.rules.builtin.javascript_rules import JAVASCRIPT_RULES
from src.rules.builtin.go_rules import GO_RULES
from src.rules.builtin.java_rules import JAVA_RULES
from src.rules.builtin.sql_rules import SQL_RULES
from src.rules.builtin.config_rules import CONFIG_RULES
from src.rules.builtin.rust_rules import RUST_RULES

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Extension → Language family mapping
# ---------------------------------------------------------------------------

EXTENSION_TO_LANGUAGE: dict[str, str] = {
    # Python
    ".py": "python",
    ".pyw": "python",
    ".pyi": "python",
    # JavaScript / TypeScript
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    # Go
    ".go": "go",
    # Java / Kotlin
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    # Rust
    ".rs": "rust",
    # SQL
    ".sql": "sql",
    # Config / IaC
    ".yaml": "config",
    ".yml": "config",
    ".json": "config",
    ".toml": "config",
    ".xml": "config",
    ".tf": "config",
    ".dockerfile": "config",
    ".env": "config",
    # Ruby
    ".rb": "ruby",
    # PHP
    ".php": "php",
    # C / C++
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    # C#
    ".cs": "csharp",
    # Swift
    ".swift": "swift",
    # Scala
    ".scala": "scala",
    # Shell
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
}

# Language family → built-in rules text
LANGUAGE_TO_RULES: dict[str, str] = {
    "python": PYTHON_RULES,
    "javascript": JAVASCRIPT_RULES,
    "typescript": JAVASCRIPT_RULES,  # TS shares JS rules with TS-specific additions
    "go": GO_RULES,
    "java": JAVA_RULES,
    "kotlin": JAVA_RULES,  # Kotlin shares JVM rules
    "rust": RUST_RULES,
    "sql": SQL_RULES,
    "config": CONFIG_RULES,
}

# Review categories applicable to each language
LANGUAGE_CATEGORIES: dict[str, list[str]] = {
    "python": ["bug", "security", "style", "test"],
    "javascript": ["bug", "security", "style", "test"],
    "typescript": ["bug", "security", "style", "test"],
    "go": ["bug", "security", "style", "test"],
    "java": ["bug", "security", "style", "test"],
    "kotlin": ["bug", "security", "style", "test"],
    "rust": ["bug", "security", "style"],
    "sql": ["security", "bug"],
    "config": ["security"],
    "ruby": ["bug", "security", "style"],
    "php": ["bug", "security", "style"],
    "c": ["bug", "security"],
    "cpp": ["bug", "security"],
    "csharp": ["bug", "security", "style"],
    "swift": ["bug", "security", "style"],
    "scala": ["bug", "security", "style"],
    "shell": ["security", "bug"],
}


# ---------------------------------------------------------------------------
# RuleGroup — the output of group_by_rules()
# ---------------------------------------------------------------------------

@dataclass
class RuleGroup:
    """A group of files that share the same review rules.

    Files with identical (source, pattern, rules_text) are batched
    into a single group for a shared LLM review call.

    Attributes:
        id: Unique group identifier (auto-assigned).
        source: Where the rules come from — "builtin" or "project".
        pattern: The glob/extension pattern that matched.
        rules_text: The actual review instructions injected into the prompt.
        categories: Which review categories apply (bug, security, etc.).
        files: Paths of files in this group.
        file_decisions: The actual FileDecision objects for each file.
    """
    id: int
    source: str          # "builtin" | "project"
    pattern: str         # glob or extension that matched
    rules_text: str      # the actual review instructions
    categories: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    file_decisions: list[FileDecision] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Project rules loader — reads .reviewrules.yaml
# ---------------------------------------------------------------------------

@dataclass
class ProjectRule:
    """A custom review rule from .reviewrules.yaml."""
    pattern: str         # glob pattern (e.g., "*.py", "src/auth/**")
    rules_text: str      # custom review instructions
    categories: list[str] = field(default_factory=list)


def load_project_rules(rules_config: dict | None) -> list[ProjectRule]:
    """Parse project rules from a .reviewrules.yaml config dict.

    Expected format:
    ```yaml
    rules:
      - pattern: "*.py"
        instructions: |
          - Always check for proper type hints
          - Ensure all public functions have docstrings
        categories: [style, bug]
      - pattern: "src/auth/**"
        instructions: |
          - Verify authentication checks on all endpoints
          - Ensure rate limiting is applied
        categories: [security]
    ```

    Args:
        rules_config: Parsed YAML dict, or None if no config file.

    Returns:
        List of ProjectRule objects.
    """
    if not rules_config:
        return []

    rules_list = rules_config.get("rules", [])
    project_rules: list[ProjectRule] = []

    for rule_dict in rules_list:
        if not isinstance(rule_dict, dict):
            continue
        pattern = rule_dict.get("pattern", "")
        instructions = rule_dict.get("instructions", "")
        categories = rule_dict.get("categories", ["bug", "security", "style"])

        if pattern and instructions:
            project_rules.append(ProjectRule(
                pattern=pattern,
                rules_text=instructions.strip(),
                categories=categories if isinstance(categories, list) else [categories],
            ))

    return project_rules


# ---------------------------------------------------------------------------
# Core grouping logic — PURE FUNCTION
# ---------------------------------------------------------------------------

def _get_extension(path: str) -> str:
    """Extract lowercased file extension."""
    lower = path.lower()
    dot_pos = lower.rfind(".")
    if dot_pos == -1:
        return ""
    return lower[dot_pos:]


def _detect_language(path: str) -> str:
    """Detect language family from file path."""
    ext = _get_extension(path)
    language = EXTENSION_TO_LANGUAGE.get(ext, "")

    # Special cases: Dockerfiles
    basename = Path(path).name.lower()
    if basename in ("dockerfile", "containerfile"):
        return "config"
    if basename.endswith(".env") or basename.startswith(".env"):
        return "config"

    return language


def _get_builtin_rules(language: str) -> str:
    """Get built-in rules for a language family."""
    return LANGUAGE_TO_RULES.get(language, "")


def _matches_project_rule(path: str, rule: ProjectRule) -> bool:
    """Check if a file path matches a project rule's glob pattern."""
    return (
        fnmatch.fnmatch(path, rule.pattern)
        or fnmatch.fnmatch(path.lower(), rule.pattern)
    )


def group_by_rules(
    selected_files: list[FileDecision],
    project_rules: dict | None = None,
) -> list[RuleGroup]:
    """Group files by which review rules apply to them.

    Files with identical (source, pattern, rules_text) share a group.
    This is the primary grouping strategy for the agentic review loop.

    The grouping order:
    1. Apply project rules first (user customization takes priority)
    2. Apply built-in language rules for files not covered by project rules
    3. Files with no matching rules get a generic "default" group

    Args:
        selected_files: List of FileDecision objects (only selected ones).
        project_rules: Optional dict from .reviewrules.yaml config.

    Returns:
        List of RuleGroup objects, each containing files that share
        the same review rules. Files may appear in multiple groups
        if both project and builtin rules apply.
    """
    parsed_project_rules = load_project_rules(project_rules)

    # Key: (source, pattern, rules_text) → list of (path, decision)
    group_map: dict[tuple[str, str, str], list[tuple[str, FileDecision]]] = {}

    for decision in selected_files:
        if not decision.selected:
            continue

        path = decision.file.path
        assigned = False

        # 1. Check project rules
        for rule in parsed_project_rules:
            if _matches_project_rule(path, rule):
                key = ("project", rule.pattern, rule.rules_text)
                group_map.setdefault(key, []).append((path, decision))
                assigned = True
                break  # First matching project rule wins

        # 2. Apply built-in language rules (always, even if project rule matched)
        language = _detect_language(path)
        builtin_text = _get_builtin_rules(language)
        if builtin_text:
            key = ("builtin", language, builtin_text)
            group_map.setdefault(key, []).append((path, decision))
            assigned = True

        # 3. Default group for files with no matching rules
        if not assigned:
            key = ("builtin", "default", "")
            group_map.setdefault(key, []).append((path, decision))

    # Build RuleGroup objects
    groups: list[RuleGroup] = []
    for group_id, ((source, pattern, rules_text), file_entries) in enumerate(
        group_map.items()
    ):
        # Deduplicate files within a group
        seen_paths: set[str] = set()
        unique_files: list[str] = []
        unique_decisions: list[FileDecision] = []
        for path, decision in file_entries:
            if path not in seen_paths:
                seen_paths.add(path)
                unique_files.append(path)
                unique_decisions.append(decision)

        # Determine categories
        if source == "project":
            # Find the project rule to get its categories
            categories = ["bug", "security", "style"]
            for rule in parsed_project_rules:
                if rule.pattern == pattern:
                    categories = rule.categories
                    break
        else:
            categories = LANGUAGE_CATEGORIES.get(pattern, ["bug", "security", "style"])

        groups.append(RuleGroup(
            id=group_id,
            source=source,
            pattern=pattern,
            rules_text=rules_text,
            categories=categories,
            files=unique_files,
            file_decisions=unique_decisions,
        ))

    logger.info(
        "rules.grouped",
        total_groups=len(groups),
        total_files=sum(len(g.files) for g in groups),
        group_summary={
            f"group_{g.id}": {
                "source": g.source,
                "pattern": g.pattern,
                "files": len(g.files),
            }
            for g in groups
        },
    )

    return groups


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_rules_for_file(path: str, project_rules: dict | None = None) -> str:
    """Get the combined rules text for a single file.

    Useful for debugging and testing — shows what rules would apply
    to a specific file path.

    Args:
        path: File path to check.
        project_rules: Optional project rules config.

    Returns:
        Combined rules text (project + builtin).
    """
    parts: list[str] = []

    # Check project rules
    if project_rules:
        for rule in load_project_rules(project_rules):
            if _matches_project_rule(path, rule):
                parts.append(f"## Project Rules ({rule.pattern})\n{rule.rules_text}")
                break

    # Add built-in rules
    language = _detect_language(path)
    builtin = _get_builtin_rules(language)
    if builtin:
        parts.append(builtin)

    return "\n\n".join(parts) if parts else ""


def format_rules_summary(groups: list[RuleGroup]) -> str:
    """Format a human-readable summary of rule groups.

    Useful for logging and debugging.
    """
    lines: list[str] = [f"Rule Groups: {len(groups)} groups"]
    for g in groups:
        lines.append(
            f"  Group {g.id}: [{g.source}] {g.pattern} — "
            f"{len(g.files)} files, categories={g.categories}"
        )
        for path in g.files[:5]:
            lines.append(f"    - {path}")
        if len(g.files) > 5:
            lines.append(f"    ... and {len(g.files) - 5} more")
    return "\n".join(lines)
