"""Language-aware rule system for per-file review customization.

Phase 2 of the OCR architecture migration. Implements a rule-group system
that batches files by which review rules apply to them. This ensures:
  - NPE detection rules only apply to Java/Go/Python files
  - XSS rules only apply to JS/TS files
  - SQL injection rules only apply to SQL/Python/Java files
  - Custom project rules (from .reviewrules.yaml) override defaults

References: new_architecture.md Phase 2 (Language-Aware Rule System),
            Alibaba OCR rulegroup.go
"""
