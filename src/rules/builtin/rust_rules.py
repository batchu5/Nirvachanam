"""Built-in review rules for Rust files.

Covers Rust-specific patterns and common pitfalls.
"""

RUST_RULES = """## Rust-Specific Review Rules

### Safety
- Flag `unsafe` blocks without safety documentation comments
- Detect raw pointer dereferencing without bounds checks
- Watch for `unwrap()` / `expect()` on `Result` or `Option` in library code
- Flag `transmute` usage (almost always wrong; use safe alternatives)
- Detect `as` casts between numeric types that could truncate or overflow

### Error Handling
- Flag `.unwrap()` in non-test code (use `?` operator or proper error handling)
- Detect `panic!()` in library code (should return `Result` instead)
- Watch for `Box<dyn Error>` when more specific error types exist
- Flag error handling that discards context (use `anyhow` or `thiserror` for chaining)

### Concurrency
- Detect `Arc<Mutex<T>>` patterns where `Arc<RwLock<T>>` would be more appropriate
- Flag shared mutable state across threads without `Send + Sync` bounds
- Watch for deadlock-prone lock ordering
- Detect `clone()` used excessively where borrowing would suffice

### Memory & Performance
- Flag unnecessary heap allocations (`Box` where stack would work)
- Detect `String` used where `&str` would suffice (excessive allocation)
- Watch for `collect()` into `Vec` followed by immediate iteration
- Flag missing `#[inline]` on small public functions in library crates
"""
