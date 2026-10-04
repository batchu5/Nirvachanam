"""Built-in review rules for configuration files.

Covers YAML, JSON, TOML, Dockerfile, Terraform, and similar
infrastructure-as-code and config files.
"""

CONFIG_RULES = """## Configuration File Review Rules

### Secret Detection
- Flag hardcoded passwords, API keys, tokens, or connection strings
- Detect base64-encoded secrets (look for `base64` or long alphanumeric strings)
- Watch for private keys embedded in config files
- Flag AWS access keys, GitHub tokens, Slack webhooks, or similar patterns

### Insecure Defaults
- Detect `debug: true` or `DEBUG=1` in production-facing configs
- Flag permissive CORS settings (`Access-Control-Allow-Origin: *`)
- Watch for wildcard TLS certificates or disabled certificate verification
- Detect `allowAll`, `permitAll`, or disabled authentication settings
- Flag HTTP (not HTTPS) endpoints in production configuration

### Dockerfile Security
- Flag `FROM` with `:latest` tag (use pinned versions)
- Detect `RUN` commands with `curl | bash` or `wget | sh` patterns
- Watch for running as `root` user (should use non-root `USER`)
- Flag `ADD` when `COPY` would suffice (ADD has extra features that can be exploited)
- Detect secrets passed via `ARG` or `ENV` (use multi-stage builds or secrets mount)

### Terraform / IaC
- Flag security groups with `0.0.0.0/0` ingress rules
- Detect unencrypted S3 buckets or EBS volumes
- Watch for overly permissive IAM policies (`Action: "*"`)
- Flag public access enabled on databases or storage
"""
