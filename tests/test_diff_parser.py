"""Tests for unified diff parsing and file filtering."""

from src.diff.parser import (
    build_diff_context,
    detect_language,
    parse_diff,
    should_skip_file,
)
from src.models.schemas import PRMetadata
from tests.conftest import (
    SAMPLE_DIFF,
    SAMPLE_DIFF_WITH_SECRETS,
    SAMPLE_EMPTY_DIFF,
    SAMPLE_LOCKFILE_DIFF,
)


class TestDetectLanguage:
    """Test language detection from file extensions."""

    def test_python_file(self):
        assert detect_language("src/utils/helpers.py") == "python"

    def test_typescript_file(self):
        assert detect_language("components/Button.tsx") == "typescript"

    def test_javascript_file(self):
        assert detect_language("app.js") == "javascript"

    def test_go_file(self):
        assert detect_language("cmd/server/main.go") == "go"

    def test_unknown_extension(self):
        assert detect_language("README") is None

    def test_sql_file(self):
        assert detect_language("migrations/001.sql") == "sql"

    def test_yaml_file(self):
        assert detect_language("config.yaml") == "yaml"
        assert detect_language("config.yml") == "yaml"


class TestShouldSkipFile:
    """Test file filtering with skip patterns."""

    def test_skip_lockfile(self):
        assert should_skip_file("package-lock.json") is True

    def test_skip_minified_js(self):
        assert should_skip_file("bundle.min.js") is True

    def test_skip_node_modules(self):
        assert should_skip_file("node_modules/lodash/index.js") is True

    def test_skip_vendor(self):
        assert should_skip_file("vendor/github.com/lib/pq/conn.go") is True

    def test_skip_image(self):
        assert should_skip_file("assets/logo.png") is True

    def test_allow_source_code(self):
        assert should_skip_file("src/auth/login.py") is False

    def test_allow_test_file(self):
        assert should_skip_file("tests/test_auth.py") is False

    def test_custom_skip_pattern(self):
        assert should_skip_file("docs/api.md", extra_skip_patterns=["docs/*"]) is True

    def test_skip_sourcemaps(self):
        assert should_skip_file("dist/app.js.map") is True


class TestParseDiff:
    """Test parsing unified diffs into structured FileContext objects."""

    def test_parse_normal_diff(self):
        """Parse a standard diff with additions and context."""
        files = parse_diff(SAMPLE_DIFF)

        assert len(files) == 1
        f = files[0]
        assert f.path == "src/auth/login.py"
        assert f.language == "python"
        assert f.additions > 0
        assert len(f.hunks) == 2  # Two @@ hunks in the sample

    def test_parse_empty_diff(self):
        """Empty diff should return empty list."""
        files = parse_diff(SAMPLE_EMPTY_DIFF)
        assert files == []

    def test_parse_filters_lockfiles(self):
        """Lockfiles should be filtered out by default skip patterns."""
        files = parse_diff(SAMPLE_LOCKFILE_DIFF)
        assert len(files) == 0

    def test_hunks_have_added_lines(self):
        """Hunks should contain added lines with line numbers."""
        files = parse_diff(SAMPLE_DIFF)
        assert len(files) > 0
        hunk = files[0].hunks[0]
        assert len(hunk.added_lines) > 0
        # Each added line is (line_number, content)
        line_no, content = hunk.added_lines[0]
        assert isinstance(line_no, int)
        assert isinstance(content, str)

    def test_hunks_have_removed_lines(self):
        """Parse diff with secrets — the diff itself should parse fine
        (redaction happens in build_diff_context, not parse_diff)."""
        files = parse_diff(SAMPLE_DIFF_WITH_SECRETS)
        assert len(files) == 1
        assert files[0].path == "config.py"


class TestBuildDiffContext:
    """Test the full DiffContext construction pipeline."""

    def test_build_context_with_metadata(self):
        """Build DiffContext from diff + PR metadata."""
        metadata = PRMetadata(
            repo="testorg/testrepo",
            pr_number=42,
            head_sha="abc123",
            base_sha="def456",
        )
        ctx = build_diff_context(SAMPLE_DIFF, metadata)

        assert ctx.head_sha == "abc123"
        assert ctx.base_sha == "def456"
        assert len(ctx.files) == 1
        assert ctx.total_additions > 0

    def test_build_context_redacts_secrets(self):
        """Secrets should be redacted in the raw_patch and records created."""
        metadata = PRMetadata(repo="test/repo", pr_number=1)
        ctx = build_diff_context(SAMPLE_DIFF_WITH_SECRETS, metadata)

        # Secrets should be replaced with [REDACTED:...] markers
        assert "ghp_ABCDEFG" not in ctx.raw_patch
        assert "[REDACTED:" in ctx.raw_patch
        assert len(ctx.redaction_records) > 0

    def test_build_context_empty_diff(self):
        """Empty diff should produce empty context."""
        metadata = PRMetadata(repo="test/repo", pr_number=1)
        ctx = build_diff_context("", metadata)

        assert len(ctx.files) == 0
        assert ctx.total_additions == 0
