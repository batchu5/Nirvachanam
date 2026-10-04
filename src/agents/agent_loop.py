"""Agentic tool-use review loop — the core of Phase 3.

This replaces the one-shot reviewer with an LLM agent that has access
to tools for deeper code analysis. The agent runs in a loop:

  1. Send diff + rules as system/user messages
  2. LLM responds with tool calls (code_comment, file_read, etc.)
  3. Execute tools, feed results back to the LLM
  4. Repeat until task_done is called or max rounds reached

Key differences from the old one-shot reviewer:
  - The LLM can READ files beyond the diff (imports, class definitions)
  - The LLM can SEARCH the codebase for symbol usage
  - The LLM can CHECK cross-file relationships
  - Comments use verbatim code snippets instead of line numbers
    (line numbers are resolved deterministically later)

The agent loop is per-group: each RuleGroup gets its own loop with
the appropriate language-specific rules injected into the prompt.

References: new_architecture.md Phase 3, Alibaba OCR llmloop.go
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

from src.agents.file_selection import FileDecision
from src.agents.tools.definitions import (
    RawComment,
    ReviewTool,
    TaskDoneTool,
    ToolContext,
    build_review_tools,
    find_tool,
)
from src.agents.tools.repair import repair_tool_call_args
from src.llm.fallback import QuotaAwareFallbackLLM
from src.models.enums import Category, Severity
from src.models.schemas import DiffContext, Finding
from src.rules.engine import RuleGroup

logger = structlog.get_logger()

PROMPT_DIR = Path(__file__).parent.parent / "prompts" / "v1"


# ---------------------------------------------------------------------------
# System prompt builder
# ---------------------------------------------------------------------------

def _build_system_prompt(rule_text: str, base_prompt: str | None = None) -> str:
    """Build the system prompt for the agent loop.

    Combines the base reviewer prompt with language-specific rules.
    The agent is instructed to use tools for its review.

    Args:
        rule_text: Language-specific rules from the rule engine.
        base_prompt: Optional custom base prompt (uses default if None).

    Returns:
        Complete system prompt string.
    """
    if base_prompt is None:
        prompt_path = PROMPT_DIR / "reviewer.md"
        if prompt_path.exists():
            base_prompt = prompt_path.read_text(encoding="utf-8")
        else:
            base_prompt = (
                "You are an expert code reviewer. Analyze the diff for bugs, "
                "security issues, style problems, and test coverage gaps."
            )

    # Build the agentic system prompt
    agent_instructions = """
## Agent Mode — Tool-Use Review

You are operating in **agent mode** with access to review tools. Instead of returning a JSON response, use the provided tools to conduct your review:

### Available Tools

1. **code_comment** — Post a review finding. Provide the file path, a verbatim code snippet from the diff, and your analysis. DO NOT guess line numbers — the system will resolve them automatically from your code snippet.

2. **file_read** — Read the full content of any file in the repository. Use this when you need to see imports, class definitions, or other context beyond the diff hunks.

3. **file_read_diff** — Read the diff of another file in the same PR. Use this to check cross-file relationships (e.g., if a function signature changed, check its callers).

4. **code_search** — Search the codebase for a symbol, function name, or pattern. Use this to find all usages of a modified function, check if a deleted symbol is still referenced, etc.

5. **task_done** — Signal that you have completed your review. Call this ONLY after you have examined all the code and posted all relevant comments.

### Review Workflow

1. First, read and understand the diff carefully
2. For each file, identify potential issues
3. When you spot an issue, use `file_read` or `code_search` to gather context if needed
4. Post your finding using `code_comment` with a verbatim code snippet
5. After reviewing all files, call `task_done` with a brief summary

### Important Rules

- **ALWAYS use `code_comment` to report findings** — do not include findings in plain text
- **Copy code snippets VERBATIM** from the diff in `existing_code` — do not paraphrase
- **Be conservative** — prefer fewer high-confidence findings over many uncertain ones
- **Focus on added/modified code** — lines starting with `+` in the diff
- **Use `file_read` sparingly** — only when you genuinely need context beyond the diff
- **Call `task_done` when finished** — this is mandatory to complete the review
"""

    parts = [base_prompt.strip()]

    if rule_text:
        parts.append(f"\n{rule_text}")

    parts.append(agent_instructions)

    return "\n\n".join(parts)


def _build_diff_message(file_decisions: list[FileDecision]) -> str:
    """Build the user message containing the diff content.

    Includes all file patches from the group.
    """
    parts: list[str] = ["<user_diff>"]

    for decision in file_decisions:
        file_ctx = decision.file
        if file_ctx.is_deleted:
            continue
        if file_ctx.patch:
            parts.append(f"### File: {file_ctx.path}")
            if file_ctx.language:
                parts.append(f"Language: {file_ctx.language}")
            parts.append(file_ctx.patch)
            parts.append("")

    parts.append("</user_diff>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tool call response processing
# ---------------------------------------------------------------------------

def _process_gemini_tool_calls(response_content: str) -> list[dict[str, Any]]:
    """Extract tool calls from Gemini's response.

    Gemini returns tool calls in different formats depending on the API version.
    This handles the common patterns.
    """
    tool_calls: list[dict[str, Any]] = []

    # Try to parse as JSON containing function_call objects
    try:
        data = json.loads(response_content)
        if isinstance(data, dict):
            if "function_call" in data:
                call = data["function_call"]
                tool_calls.append({
                    "name": call.get("name", ""),
                    "arguments": call.get("args", call.get("arguments", {})),
                })
            elif "tool_calls" in data:
                for call in data["tool_calls"]:
                    func = call.get("function", call)
                    tool_calls.append({
                        "name": func.get("name", ""),
                        "arguments": func.get("arguments", {}),
                    })
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "function_call" in item:
                    call = item["function_call"]
                    tool_calls.append({
                        "name": call.get("name", ""),
                        "arguments": call.get("args", call.get("arguments", {})),
                    })
    except (json.JSONDecodeError, TypeError):
        pass

    return tool_calls


# ---------------------------------------------------------------------------
# Core agent loop
# ---------------------------------------------------------------------------

async def run_agent_loop(
    rule_group: RuleGroup,
    diff_context: DiffContext,
    llm: QuotaAwareFallbackLLM,
    *,
    repo: str = "",
    head_sha: str = "",
    github_client: Any = None,
    installation_id: int | None = None,
    max_rounds: int = 10,
    budget_tokens: int = 50_000,
    timeout: int = 45,
) -> list[RawComment]:
    """Run the LLM agent in a tool-use loop until task_done.

    This is the core of Phase 3. The agent has access to tools for
    deeper code analysis and posts findings via the code_comment tool.

    The loop terminates when:
    - The agent calls task_done
    - max_rounds is reached
    - Token budget is exhausted
    - The LLM returns no tool calls (text-only response)

    Args:
        rule_group: RuleGroup with files and rules for this review.
        diff_context: Full diff context (for cross-file access).
        llm: QuotaAwareFallbackLLM instance.
        repo: Repository full name for GitHub API access.
        head_sha: HEAD SHA for file reads.
        github_client: Optional GitHubClient for file reads.
        installation_id: GitHub installation ID.
        max_rounds: Maximum agent loop iterations.
        budget_tokens: Token budget for the entire loop.
        timeout: Timeout per LLM call in seconds.

    Returns:
        List of RawComment objects collected from code_comment calls.
    """
    # Build tool context
    context = ToolContext(
        diff_context=diff_context,
        repo=repo,
        head_sha=head_sha,
        github_client=github_client,
        installation_id=installation_id,
    )
    tools = build_review_tools(context)
    tool_schemas = [t.schema for t in tools]

    # Build system prompt with language-specific rules
    system_prompt = _build_system_prompt(rule_group.rules_text)
    diff_message = _build_diff_message(rule_group.file_decisions)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": diff_message},
    ]

    total_tokens_used = 0

    logger.info(
        "agent_loop.starting",
        group_id=rule_group.id,
        files=len(rule_group.files),
        pattern=rule_group.pattern,
        max_rounds=max_rounds,
    )

    for round_num in range(max_rounds):
        # Check token budget
        if total_tokens_used >= budget_tokens:
            logger.warning(
                "agent_loop.budget_exhausted",
                round=round_num,
                tokens_used=total_tokens_used,
                budget=budget_tokens,
            )
            break

        # Call LLM with tools
        # We pass tool schemas as part of the message context since
        # the QuotaAwareFallbackLLM uses structured output, not native tools.
        # We'll format tools into the system prompt for now.
        tool_instruction = _format_tools_for_prompt(tool_schemas)
        augmented_messages = list(messages)

        # On first round, add tool definitions
        if round_num == 0:
            augmented_messages[0] = {
                "role": "system",
                "content": messages[0]["content"] + "\n\n" + tool_instruction,
            }

        response = await llm.invoke(
            messages=augmented_messages,
            temperature=0.2,
            max_tokens=4000,
            timeout=timeout,
        )

        if response is None:
            logger.warning("agent_loop.llm_failed", round=round_num)
            break

        total_tokens_used += response.tokens_used
        response_text = response.content

        # Try to extract tool calls from the response
        tool_calls = _extract_tool_calls_from_response(response_text)

        if not tool_calls:
            # No tool calls — the LLM responded with plain text
            # This likely means it's done or confused
            logger.info(
                "agent_loop.no_tool_calls",
                round=round_num,
                response_preview=response_text[:200],
            )

            # Try to extract findings from plain text as fallback
            _extract_inline_findings(response_text, context)
            break

        # Process each tool call
        task_done = False
        for call in tool_calls:
            tool_name = call.get("name", "")
            raw_args = call.get("arguments", {})

            # Find the matching tool
            tool = find_tool(tool_name, tools)
            if tool is None:
                logger.warning(
                    "agent_loop.unknown_tool",
                    tool_name=tool_name,
                    round=round_num,
                )
                messages.append({
                    "role": "assistant",
                    "content": response_text,
                })
                messages.append({
                    "role": "user",
                    "content": f"Error: Unknown tool '{tool_name}'. Available tools: {[t.name for t in tools]}",
                })
                continue

            # Repair arguments if they're a string
            if isinstance(raw_args, str):
                repaired = repair_tool_call_args(raw_args, tool.parameters)
                if repaired is None:
                    messages.append({
                        "role": "assistant",
                        "content": response_text,
                    })
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Error: Could not parse arguments for tool '{tool_name}'. "
                            f"Please provide valid JSON matching the schema."
                        ),
                    })
                    continue
                raw_args = repaired

            # Execute the tool
            try:
                result = await tool.execute(raw_args)
            except Exception as e:
                logger.warning(
                    "agent_loop.tool_error",
                    tool=tool_name,
                    error=str(e),
                    round=round_num,
                )
                result = f"Error executing {tool_name}: {str(e)}"

            # Check if task_done was called
            if isinstance(tool, TaskDoneTool):
                task_done = True
                break

            # Feed result back to the LLM
            messages.append({
                "role": "assistant",
                "content": response_text,
            })
            messages.append({
                "role": "user",
                "content": f"Tool result ({tool_name}):\n{result}",
            })

        if task_done:
            logger.info(
                "agent_loop.completed",
                rounds=round_num + 1,
                comments=len(context.comments),
                tokens_used=total_tokens_used,
            )
            break

    logger.info(
        "agent_loop.finished",
        group_id=rule_group.id,
        total_comments=len(context.comments),
        total_tokens=total_tokens_used,
    )

    return context.comments


# ---------------------------------------------------------------------------
# Tool call extraction from LLM response text
# ---------------------------------------------------------------------------

def _extract_tool_calls_from_response(
    response_text: str,
) -> list[dict[str, Any]]:
    """Extract tool calls from the LLM response text.

    The LLM may embed tool calls in various formats:
    1. JSON with function_call objects
    2. Markdown code blocks with JSON
    3. Inline function call syntax

    This function handles all common patterns.
    """
    tool_calls: list[dict[str, Any]] = []

    # Pattern 1: JSON objects with "name" and "arguments" fields
    json_pattern = r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
    for match in re.finditer(json_pattern, response_text, re.DOTALL):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
            tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            tool_calls.append({"name": name, "arguments": match.group(2)})

    if tool_calls:
        return tool_calls

    # Pattern 2: Function-call-like syntax: tool_name({...})
    func_pattern = r'(code_comment|file_read|file_read_diff|code_search|task_done)\s*\((\{.*?\})\)'
    for match in re.finditer(func_pattern, response_text, re.DOTALL):
        name = match.group(1)
        try:
            args = json.loads(match.group(2))
            tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            tool_calls.append({"name": name, "arguments": match.group(2)})

    if tool_calls:
        return tool_calls

    # Pattern 3: Try to parse entire response as JSON with tool calls
    try:
        data = json.loads(response_text)
        if isinstance(data, dict):
            if "tool_calls" in data:
                for call in data["tool_calls"]:
                    tool_calls.append({
                        "name": call.get("name", ""),
                        "arguments": call.get("arguments", {}),
                    })
            elif "name" in data and "arguments" in data:
                tool_calls.append({
                    "name": data["name"],
                    "arguments": data["arguments"],
                })
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    tool_calls.append({
                        "name": item.get("name", ""),
                        "arguments": item.get("arguments", {}),
                    })
    except (json.JSONDecodeError, TypeError):
        pass

    # Pattern 4: Look for JSON blocks in markdown
    code_blocks = re.findall(r'```(?:json)?\s*\n(.*?)```', response_text, re.DOTALL)
    for block in code_blocks:
        try:
            data = json.loads(block.strip())
            if isinstance(data, dict) and "name" in data:
                tool_calls.append({
                    "name": data.get("name", ""),
                    "arguments": data.get("arguments", {}),
                })
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "name" in item:
                        tool_calls.append({
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", {}),
                        })
        except (json.JSONDecodeError, TypeError):
            pass

    return tool_calls


def _format_tools_for_prompt(tool_schemas: list[dict[str, Any]]) -> str:
    """Format tool schemas into a prompt-friendly description.

    Since we're using text-based tool calling (not native function calling),
    we need to describe the tools and their parameters in the prompt.
    """
    parts = ["## Tool Definitions\n"]
    parts.append("To call a tool, output a JSON object with 'name' and 'arguments' fields.")
    parts.append("You may call multiple tools in a single response.\n")
    parts.append("Format each tool call as:")
    parts.append('```json')
    parts.append('{"name": "tool_name", "arguments": {...}}')
    parts.append('```\n')

    for schema in tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "")
        desc = func.get("description", "")
        params = func.get("parameters", {})

        parts.append(f"### `{name}`")
        parts.append(f"{desc}\n")

        properties = params.get("properties", {})
        required = params.get("required", [])

        if properties:
            parts.append("Parameters:")
            for prop_name, prop_schema in properties.items():
                req_marker = " (required)" if prop_name in required else " (optional)"
                prop_type = prop_schema.get("type", "string")
                prop_desc = prop_schema.get("description", "")
                enum_values = prop_schema.get("enum")
                if enum_values:
                    prop_desc += f" — one of: {', '.join(enum_values)}"
                parts.append(f"  - `{prop_name}` ({prop_type}{req_marker}): {prop_desc}")
            parts.append("")

    return "\n".join(parts)


def _extract_inline_findings(
    response_text: str,
    context: ToolContext,
) -> None:
    """Fallback: extract findings from plain-text LLM response.

    When the LLM doesn't use tool calls, try to extract structured
    findings from its text response. This handles the case where
    the LLM ignores the tool-use instructions.
    """
    # Try to find a JSON array of findings
    try:
        # Look for {"findings": [...]} pattern
        match = re.search(r'\{[^{}]*"findings"\s*:\s*\[(.*?)\]\s*\}', response_text, re.DOTALL)
        if match:
            findings_json = '{"findings": [' + match.group(1) + ']}'
            data = json.loads(findings_json)
            for finding in data.get("findings", []):
                if isinstance(finding, dict) and "file" in finding and "message" in finding:
                    context.comments.append(RawComment(
                        file=finding["file"],
                        existing_code=finding.get("existing_code", ""),
                        message=finding["message"],
                        severity=finding.get("severity", "warning"),
                        category=finding.get("category", "bug"),
                        suggested_fix=finding.get("suggested_fix"),
                        confidence=finding.get("confidence", 0.5),
                        language=finding.get("language"),
                    ))
    except (json.JSONDecodeError, TypeError):
        pass


# ---------------------------------------------------------------------------
# Convert RawComments to Findings (with line resolution)
# ---------------------------------------------------------------------------

def raw_comments_to_findings(
    comments: list[RawComment],
    diff_context: DiffContext,
) -> list[Finding]:
    """Convert RawComment objects to Finding objects.

    Resolves line numbers from the verbatim code snippets in each comment.
    Uses the code-snippet-based resolution approach from OCR.

    Args:
        comments: List of RawComment from the agent loop.
        diff_context: Parsed diff for line resolution.

    Returns:
        List of Finding objects with resolved line numbers.
    """
    findings: list[Finding] = []

    # Category string → Category enum mapping
    category_map = {
        "bug": Category.BUG,
        "security": Category.SECURITY,
        "style": Category.STYLE,
        "test": Category.TEST,
    }

    for comment in comments:
        # Resolve line number from code snippet
        line = _resolve_line_from_snippet(
            comment.file, comment.existing_code, diff_context
        )

        if line is None:
            # Fallback: try to find any line in the file's diff
            line = _fallback_line_resolution(comment.file, diff_context)

        if line is None:
            line = 1  # Last resort default

        category = category_map.get(comment.category.lower(), Category.BUG)

        findings.append(Finding(
            file=comment.file,
            line=line,
            severity=Severity(comment.severity) if comment.severity in ("info", "warning", "critical") else Severity.WARNING,
            category=category,
            message=comment.message,
            suggested_fix=comment.suggested_fix,
            confidence=comment.confidence,
            agent="agent_loop",
            language=comment.language,
        ))

    return findings


def _resolve_line_from_snippet(
    file_path: str,
    code_snippet: str,
    diff_context: DiffContext,
) -> int | None:
    """Resolve a line number from a verbatim code snippet.

    Primary resolution: match against diff hunk added/context lines.
    Fallback: scan full patch text line-by-line.

    Args:
        file_path: File path to search in.
        code_snippet: Verbatim code snippet to match.
        diff_context: Parsed diff context.

    Returns:
        1-indexed line number, or None if no match found.
    """
    if not code_snippet or not code_snippet.strip():
        return None

    snippet_stripped = code_snippet.strip()
    snippet_lines = snippet_stripped.splitlines()

    for file_ctx in diff_context.files:
        if file_ctx.path != file_path:
            continue

        # Strategy 1: Match against hunk added lines
        for hunk in file_ctx.hunks:
            for line_no, line_content in hunk.added_lines:
                if snippet_stripped in line_content.strip():
                    return line_no
                # Match first line of multi-line snippet
                if snippet_lines and snippet_lines[0].strip() in line_content.strip():
                    return line_no

        # Strategy 2: Scan patch text
        if file_ctx.patch:
            current_line = file_ctx.hunks[0].target_start if file_ctx.hunks else 1
            for patch_line in file_ctx.patch.splitlines():
                if patch_line.startswith("@@"):
                    # Parse hunk header for line number
                    match = re.search(r'\+(\d+)', patch_line)
                    if match:
                        current_line = int(match.group(1))
                    continue

                if patch_line.startswith("-"):
                    continue  # Removed lines don't count

                if snippet_stripped in patch_line.strip():
                    return current_line

                if not patch_line.startswith("-"):
                    current_line += 1

        # Strategy 3: Cross-file relocation — search all diffs
        break  # Only search the target file

    # Cross-file search (if not found in target file)
    for file_ctx in diff_context.files:
        if file_ctx.path == file_path:
            continue
        for hunk in file_ctx.hunks:
            for line_no, line_content in hunk.added_lines:
                if snippet_stripped in line_content.strip():
                    logger.info(
                        "line_resolution.cross_file",
                        original_file=file_path,
                        actual_file=file_ctx.path,
                        line=line_no,
                    )
                    return line_no

    return None


def _fallback_line_resolution(
    file_path: str,
    diff_context: DiffContext,
) -> int | None:
    """Fallback: return the first added line in the file's diff."""
    for file_ctx in diff_context.files:
        if file_ctx.path == file_path:
            for hunk in file_ctx.hunks:
                if hunk.added_lines:
                    return hunk.added_lines[0][0]
                return hunk.target_start
    return None


# ---------------------------------------------------------------------------
# High-level entry point for per-group review
# ---------------------------------------------------------------------------

async def review_group(
    rule_group: RuleGroup,
    diff_context: DiffContext,
    llm: QuotaAwareFallbackLLM,
    *,
    repo: str = "",
    head_sha: str = "",
    github_client: Any = None,
    installation_id: int | None = None,
    max_rounds: int = 10,
    budget_tokens: int = 50_000,
    timeout: int = 45,
) -> list[Finding]:
    """Review a single RuleGroup using the agentic tool-use loop.

    This is the high-level entry point that:
    1. Runs the agent loop to collect RawComments
    2. Resolves line numbers from code snippets
    3. Returns validated Finding objects

    Args:
        rule_group: The file group with rules to review.
        diff_context: Full diff context.
        llm: LLM instance.
        repo: Repository name.
        head_sha: HEAD SHA for file reads.
        github_client: Optional GitHub client.
        installation_id: GitHub installation ID.
        max_rounds: Max agent loop rounds.
        budget_tokens: Token budget.
        timeout: Per-call timeout.

    Returns:
        List of Finding objects from this review group.
    """
    if not rule_group.files:
        return []

    # Run the agent loop
    raw_comments = await run_agent_loop(
        rule_group=rule_group,
        diff_context=diff_context,
        llm=llm,
        repo=repo,
        head_sha=head_sha,
        github_client=github_client,
        installation_id=installation_id,
        max_rounds=max_rounds,
        budget_tokens=budget_tokens,
        timeout=timeout,
    )

    if not raw_comments:
        return []

    # Convert to Findings with line resolution
    findings = raw_comments_to_findings(raw_comments, diff_context)

    logger.info(
        "review_group.completed",
        group_id=rule_group.id,
        pattern=rule_group.pattern,
        raw_comments=len(raw_comments),
        findings=len(findings),
    )

    return findings


# needed for regex import
import re
