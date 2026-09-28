"""Tests for secret redaction in diff text."""

from src.security.redaction import redact_secrets


class TestRedactSecrets:
    """Test the pre-LLM secret redaction pipeline."""

    def test_github_token_redacted(self):
        """GitHub personal access tokens should be redacted."""
        text = 'TOKEN = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"'
        redacted, records = redact_secrets(text)

        assert "ghp_ABCDEFG" not in redacted
        assert "[REDACTED:GitHub Token]" in redacted
        assert len(records) >= 1
        assert any(r.type == "GitHub Token" for r in records)

    def test_aws_access_key_redacted(self):
        """AWS access key IDs (AKIA...) should be redacted."""
        text = "aws_key = AKIAIOSFODNN7EXAMPLE"
        redacted, records = redact_secrets(text)

        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED:AWS Access Key]" in redacted

    def test_private_key_pem_redacted(self):
        """PEM private key headers should be redacted."""
        text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQ..."
        redacted, records = redact_secrets(text)

        assert "-----BEGIN RSA PRIVATE KEY-----" not in redacted
        assert "[REDACTED:Private Key PEM]" in redacted

    def test_slack_token_redacted(self):
        """Slack tokens (xoxb-...) should be redacted."""
        text = 'slack_token = "xoxb-1234567890-abcdefghij"'
        redacted, records = redact_secrets(text)

        assert "xoxb-1234567890" not in redacted
        assert "[REDACTED:Slack Token]" in redacted

    def test_generic_api_key_redacted(self):
        """Generic API keys in assignments should be redacted."""
        text = 'api_key = ""'
        redacted, records = redact_secrets(text)

        assert "sk_live_ABCDEFG" not in redacted
        assert "[REDACTED:" in redacted

    def test_no_secrets_unchanged(self):
        """Normal code without secrets should pass through unchanged."""
        text = """
def hello():
    name = "world"
    return f"Hello, {name}!"
"""
        redacted, records = redact_secrets(text)

        assert redacted == text
        assert len(records) == 0

    def test_empty_input(self):
        """Empty string should return empty."""
        redacted, records = redact_secrets("")
        assert redacted == ""
        assert records == []

    def test_multiple_secrets_all_redacted(self):
        """Multiple secrets in the same text should all be redacted."""
        text = """
+GITHUB_TOKEN = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn"
+AWS_KEY = AKIAIOSFODNN7EXAMPLE
+SLACK = "xoxb-slack-token-value"
"""
        redacted, records = redact_secrets(text)

        assert "ghp_ABCDEFG" not in redacted
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "xoxb-slack" not in redacted
        assert len(records) >= 3

    def test_redaction_records_have_line_numbers(self):
        """Each redaction record should include approximate line number."""
        text = "line1\nline2\nghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmn\nline4"
        _, records = redact_secrets(text)

        assert len(records) >= 1
        assert records[0].line == 3  # Secret is on line 3

    def test_safe_strings_not_redacted(self):
        """Common variable names and short strings should NOT be redacted."""
        text = """
username = "admin"
password_hash = bcrypt.hash(raw)
api_url = "https://api.example.com"
token_count = 42
"""
        redacted, records = redact_secrets(text)

        # These should NOT trigger false positives
        assert redacted == text
        assert len(records) == 0
