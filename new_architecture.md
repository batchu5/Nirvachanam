# Architecture Analysis: Alibaba OCR → Your PR Review Agent

## What Makes Alibaba's Open-Code-Review Engineering Insane

After deep-reading the OCR repo, here's what's genuinely brilliant about their architecture. It's not just "use an LLM" — it's a **hybrid deterministic pipeline + LLM agent** system where the LLM does only what a machine *can't*.

---

## 🧬 The 8 Core Architectural Innovations

### 1. Deterministic Pre-Dispatch File Selection (`selectFiles`)

> [!IMPORTANT]
> This is the crown jewel. Before any LLM call, a **pure function** decides which files enter the review and which are excluded. It's stateless, no IO, no side effects.

**OCR's approach** ([selection.go](https://github.com/alibaba/open-code-review/blob/main/internal/agent/selection.go)):
```
For each changed file → apply in order:
  1. Extension allowlist gate  (is .go/.py/.ts etc?)
  2. Path exclusion patterns   (.lock, vendor/, node_modules/)
  3. Deletion check            (deleted files retained but not dispatched)
  4. Per-file diff-size ceiling (single file too big → exclude)
  → Output: fileDecision { Diff, Reason, DiffTokens }
```

The key insight: `--preview` and the real run **consume the same answer**. No drift between what you *said* you'd review and what you *actually* reviewed.

**Your current state**: Your [triage.py](file:///c:/Users/varsh/Desktop/Pr_review_agent/src/agents/triage.py) does some of this (skip patterns, security detection) but it's mixed into the tier classification. The selection and triage are one function.

---

### 2. Semantic File Grouping via LLM (`groupDiffs`)

**OCR's approach** ([grouping.go](https://github.com/alibaba/open-code-review/blob/main/internal/agent/grouping.go)):
- For small change sets: group without LLM (deterministic)
- For larger changes: call LLM with **file metadata only** (no diff content!) to produce semantic groups
- Groups share a single LLM review call → related files are reviewed together
- Max 10 files per group
- Fallback: one-file-per-group if LLM fails

**Your current state**: You don't have file grouping. Every file goes into one big prompt in [reviewer.py](file:///c:/Users/varsh/Desktop/Pr_review_agent/src/agents/reviewer.py). Large PRs get chunked in [large_pr.py](file:///c:/Users/varsh/Desktop/Pr_review_agent/src/graph/large_pr.py) but by directory, not semantics.

---

### 3. Agentic Tool-Use Loop (`llmloop`)

**OCR's approach**: The LLM is an **agent with tools**, not a one-shot prompt→response. The tools are:

| Tool | Purpose |
|------|---------|
| `code_comment` | Post a review comment with line-level precision |
| `file_read` | Read full file content beyond the diff |
| `file_find` | Find files by path pattern in the repo |
| `file_read_diff` | Read another file's diff for cross-file context |
| `code_search` | Grep the codebase for symbol usage |
| `task_done` | Signal the review is complete |

The agent runs in a **loop**: LLM emits tool calls → tool results fed back → LLM continues → until `task_done`. This is how it achieves "deep reviews, not just surface-level diff feedback."

**Your current state**: Your [reviewer.py](file:///c:/Users/varsh/Desktop/Pr_review_agent/src/agents/reviewer.py) is **one-shot**: system prompt + diff → JSON response. No tool use, no ability to read surrounding files, no codebase search.

---

### 4. Comment Line Resolution (`resolver.go`)

**OCR's approach** ([resolver.go](https://github.com/alibaba/open-code-review/blob/main/internal/diff/resolver.go)):
- LLM returns comments with `ExistingCode` (a verbatim code snippet) + file path
- **Deterministic resolver** matches the snippet against diff hunks to find exact line numbers
- Fallback: scan full new-file content line-by-line
- Cross-file relocation: if the LLM filed a comment against the wrong file, string-match the `ExistingCode` across all diffs to find its true home

**Your current state**: Your [critic.py](file:///c:/Users/varsh/Desktop/Pr_review_agent/src/agents/critic.py#L99-L113) has `_line_exists_in_diff` but it's a loose check (±5 lines). No code-snippet-based resolution.

---

### 5. Comment Args Repair (`comment_args_repair.go`)

When the LLM produces malformed tool-call arguments (broken JSON, wrong field names), OCR has a **deterministic repair layer** that tries to fix them before falling back to the LLM. This is separate from your `ValidationError` retry in reviewer.py — it operates at the tool-call argument level, not the whole response.

---

### 6. Delegate / Rule-Group System (`rulegroup.go`)

**OCR's approach**: Files are grouped by which **review rules** apply to them. Custom rules (from `.opencodereview/rules.yaml`) are resolved per-file via glob patterns, and files sharing the same rule text are batched together. This means:
- NPE detection rules only apply to Java/Go files
- XSS rules only apply to JS/TS files
- Custom project rules override defaults

**Your current state**: You have a single universal prompt in [reviewer.md](file:///c:/Users/varsh/Desktop/Pr_review_agent/prompts/v1/reviewer.md). No per-language or per-rule-type customization.

---

### 7. Identity-Based Dedup & Resume (`identity.go`)

**OCR's approach**:
- Before any LLM call, compute a **SealedInput** — a hash of the exact diffs that will be reviewed
- If a previous run with the same identity exists and completed partially, **resume from checkpoint**
- This is why `selectFiles` must be pure: the identity depends on it

**Your current state**: You have content_signature in [schemas.py](file:///c:/Users/varsh/Desktop/Pr_review_agent/src/models/schemas.py#L47-L63) for finding-level dedup, and a checkpointer in [pipeline.py](file:///c:/Users/varsh/Desktop/Pr_review_agent/src/graph/pipeline.py), but no run-level identity sealing.

---

### 8. Token Budget Enforcement

**OCR's approach** ([estimate.go](https://github.com/alibaba/open-code-review/blob/main/internal/agent/estimate.go)):
- Pre-run token estimation per file (prompt overhead + diff tokens × rounds × output/round)
- Groups are split if they'd exceed the token budget
- User-facing cost estimate before the run starts
- Budget tracks agent tool-use inflation (tools can multiply the prompt)

**Your current state**: You have `_get_max_tokens` in [reviewer.py](file:///c:/Users/varsh/Desktop/Pr_review_agent/src/agents/reviewer.py#L121-L128) but no pre-run estimation or budget enforcement.

---

## 🏗️ Implementation Plan: Adopting OCR's Architecture

### Your Current Pipeline
```
Webhook → Triage (rules) → Reviewer (1 LLM call) → Critic (rules) → Summarizer (template)
```

### Target Pipeline (OCR-style)
```
Webhook → File Selection (deterministic) → Triage (rules) → File Grouping (LLM or deterministic)
  → Per-Group Agent Loop (LLM + tools) → Comment Resolution (deterministic)
  → Rule-Based Critic (deterministic) → Summarizer (template)
```

---

### Phase 1: Deterministic File Selection Layer ⚡
**Files to create/modify:** `src/agents/file_selection.py` (new), modify `src/agents/triage.py`

Separate file selection from triage into a pure function:

```python
# src/agents/file_selection.py

@dataclass(frozen=True)
class ExcludeReason(StrEnum):
    NONE = "none"                    # Selected for review
    EXTENSION = "extension"          # Not in allowlist
    SKIP_PATTERN = "skip_pattern"    # Matches skip glob
    DELETED = "deleted"              # File was deleted
    TOO_LARGE = "too_large"          # Diff exceeds per-file ceiling
    BINARY = "binary"               # Binary file detected
    GENERATED = "generated"         # Auto-generated code

@dataclass(frozen=True)
class FileDecision:
    file: FileContext
    reason: ExcludeReason
    diff_tokens: int  # estimated token cost

    @property
    def selected(self) -> bool:
        return self.reason == ExcludeReason.NONE

    @property
    def retained(self) -> bool:
        """Deleted files stay in the context but aren't dispatched."""
        return self.reason in (ExcludeReason.NONE, ExcludeReason.DELETED)

# PURE FUNCTION — no IO, no side effects
def select_files(
    diff_context: DiffContext,
    *,
    max_diff_tokens_per_file: int = 4000,
    allowed_extensions: frozenset[str] | None = None,
    skip_patterns: list[str] | None = None,
) -> list[FileDecision]:
    ...
```

> [!TIP]
> This is the most impactful change. It makes your pipeline deterministic and testable at the boundary layer. Every downstream component gets a guaranteed-filtered input.

---

### Phase 2: Language-Aware Rule System 📜
**Files to create:** `src/rules/engine.py`, `src/rules/builtin/` directory

```python
# src/rules/engine.py

@dataclass
class RuleGroup:
    id: int
    source: str          # "builtin" | "project" | "custom"
    pattern: str         # glob that matched
    rules_text: str      # the actual review instructions
    categories: list[str] # which review categories apply
    files: list[str]     # files in this group

def group_by_rules(
    selected_files: list[FileDecision],
    project_rules: dict | None = None,  # from .reviewrules.yaml
) -> list[RuleGroup]:
    """Group files by which review rules apply to them.
    
    Files with identical (source, pattern, rules_text) share a group.
    """
```

Built-in rules for:
- **Python**: NPE (None checks), resource leaks, type errors
- **JavaScript/TypeScript**: XSS, prototype pollution, async/await pitfalls
- **SQL**: Injection patterns
- **Config files**: Secret detection, insecure defaults

---

### Phase 3: Agentic Tool-Use Review Loop 🔧
**Files to create:** `src/agents/tools/`, `src/agents/agent_loop.py`

This is the biggest upgrade — moving from one-shot to a tool-use agent loop:

```python
# src/agents/tools/definitions.py
REVIEW_TOOLS = [
    CodeCommentTool(),    # Post a finding with code context
    FileReadTool(),       # Read full file content
    FileReadDiffTool(),   # Read another file's diff
    CodeSearchTool(),     # Grep for symbol usage
    TaskDoneTool(),       # Signal completion
]

# src/agents/agent_loop.py
async def run_agent_loop(
    diff_group: FileGroup,
    llm: QuotaAwareFallbackLLM,
    tools: list[Tool],
    rule_text: str,
    *,
    max_rounds: int = 10,
    budget_tokens: int = 50_000,
) -> list[RawComment]:
    """Run the LLM agent in a tool-use loop until task_done."""
    messages = [build_system_prompt(rule_text)]
    messages.append(build_diff_message(diff_group))
    
    for round_num in range(max_rounds):
        response = await llm.invoke(
            messages=messages,
            tools=[t.schema for t in tools],
        )
        
        if response.has_tool_calls:
            for call in response.tool_calls:
                tool = find_tool(call.name, tools)
                result = await tool.execute(call.arguments)
                
                if isinstance(tool, TaskDoneTool):
                    return collect_comments(messages)
                    
                messages.append(tool_result_message(call.id, result))
        else:
            break
    
    return collect_comments(messages)
```

---

### Phase 4: Deterministic Comment Resolution 🎯
**Files to create:** `src/diff/resolver.py` (new)

```python
# src/diff/resolver.py

def resolve_line_numbers(
    comments: list[RawComment],
    diff_context: DiffContext,
) -> list[RawComment]:
    """Match comment code snippets against diff hunks to find exact lines.
    
    Primary: match against diff hunk added/context lines
    Fallback: scan full new-file content line-by-line
    """

def relocate_across_files(
    comment: RawComment,
    all_diffs: list[FileContext],
) -> str | None:
    """If ExistingCode doesn't match the filed-against file,
    search all diffs for a unique match and re-file."""
```

---

### Phase 5: Semantic File Grouping 📦
**Files to create:** `src/agents/grouping.py`

```python
# src/agents/grouping.py

SMALL_CHANGE_THRESHOLD = 5  # files

async def group_files(
    decisions: list[FileDecision],
    llm: QuotaAwareFallbackLLM | None,
    *,
    max_files_per_group: int = 10,
    token_limit: int = 50_000,
) -> list[FileGroup]:
    """Group related files for review.
    
    - ≤1 file: no grouping needed
    - ≤SMALL_CHANGE_THRESHOLD: group all together (no LLM)
    - >threshold: call LLM with file metadata (paths only, no diffs)
      to produce semantic groups, then enforce token budget per group
    """
```

---

### Phase 6: Token Budget & Cost Estimation 💰
**Files to create:** `src/agents/budget.py`

```python
# src/agents/budget.py

PROMPT_OVERHEAD_TOKENS = 2000
AVG_ROUNDS_PER_FILE = 7
AVG_OUTPUT_TOKENS_PER_ROUND = 700

@dataclass
class ReviewEstimate:
    files: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    estimated_cost_description: str  # "~3 Gemini API calls"

def estimate_review_cost(
    decisions: list[FileDecision],
) -> ReviewEstimate:
    """Pre-run, order-of-magnitude cost projection."""
```

---

### Phase 7: Run Identity & Checkpoint Resume 🔄
**Files to modify:** `src/graph/pipeline.py`

```python
# In pipeline.py

def compute_run_identity(
    selected_files: list[FileDecision],
    head_sha: str,
    base_sha: str,
) -> str:
    """SHA-256 of the selected file set + commit range.
    
    If a previous run with this identity completed, skip the review.
    If it completed partially, resume from checkpoint.
    """
```

---

### Phase 8: Comment Repair Layer 🔧
**Files to create:** `src/agents/tools/repair.py`

```python
# src/agents/tools/repair.py

def repair_tool_call_args(
    raw_args: str,
    tool_schema: dict,
) -> dict | None:
    """Attempt to fix malformed LLM tool-call arguments.
    
    Handles:
    - Missing required fields (fill defaults)
    - Wrong field names (fuzzy match)
    - Broken JSON (attempt repair)
    - Type coercion (string "42" → int 42)
    """
```

---

## 📊 Priority Order & Effort Estimates

| Phase | Impact | Effort | Dependencies |
|-------|--------|--------|-------------|
| **1. File Selection** | 🔴 Critical | ~1 day | None |
| **3. Agent Tool Loop** | 🔴 Critical | ~3 days | Phase 1 |
| **4. Comment Resolution** | 🟡 High | ~1 day | Phase 3 |
| **2. Rule System** | 🟡 High | ~2 days | Phase 1 |
| **5. Semantic Grouping** | 🟢 Medium | ~1 day | Phase 1 |
| **6. Token Budget** | 🟢 Medium | ~0.5 day | Phase 1 |
| **7. Run Identity** | 🟢 Medium | ~1 day | Phase 1 |
| **8. Repair Layer** | 🔵 Nice-to-have | ~0.5 day | Phase 3 |

---

## 🔑 Key Principle to Internalize

> **OCR's genius is that >80% of the work is deterministic code, not LLM calls.** The LLM is a precision instrument used only where human judgment is required — analyzing code semantics. Everything else (selection, grouping heuristics, line resolution, dedup, formatting) is fast, testable, reproducible code.

Your current codebase already embodies this principle with your rule-based triage and critic. The OCR upgrade path deepens it with:
1. **Sharper pre-dispatch filtering** (file selection as a pure function)
2. **Richer LLM interaction** (tool-use loop instead of one-shot)
3. **Better post-LLM verification** (code-snippet-based line resolution)

---

## ⚠️ What NOT to Copy

- OCR is written in Go — don't try to port it line-by-line. Adapt the *patterns* to your Python/LangGraph stack
- OCR's `--preview` mode and `--scan` (full-file review) are CLI features you don't need in a GitHub App
- Their MCP (Model Context Protocol) integration is an extension point you can skip for now
- Their npm packaging / GitHub Action wrapper is deployment infrastructure, not architecture
