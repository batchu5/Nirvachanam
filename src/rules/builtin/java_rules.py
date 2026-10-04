"""Built-in review rules for Java and Kotlin files.

Covers JVM-specific patterns and common pitfalls.
"""

JAVA_RULES = """## Java/Kotlin-Specific Review Rules

### Null Pointer Safety (NPE Prevention)
- Flag method calls on potentially null references without null checks
- Detect Optional.get() without isPresent()/isEmpty() check
- Watch for autoboxing NPE (`Integer i = null; int x = i;`)
- Flag nullable return types passed directly to non-null parameters
- Detect missing @Nullable/@NonNull annotations on public API parameters

### Resource Management
- Flag `try` blocks opening resources without try-with-resources
- Detect `InputStream`, `OutputStream`, `Connection` without close/try-with-resources
- Watch for `PreparedStatement` created but not closed on error paths
- Flag manual resource management in code targeting Java 9+ (should use try-with-resources)

### Thread Safety
- Detect non-synchronized access to shared mutable state
- Flag `SimpleDateFormat` used across threads (not thread-safe)
- Watch for `HashMap` used in concurrent context (use `ConcurrentHashMap`)
- Detect double-checked locking without `volatile` keyword
- Flag lazy initialization of singletons without synchronization

### SQL Injection
- Flag string concatenation in SQL queries (use PreparedStatement)
- Detect `Statement.execute()` with interpolated strings
- Watch for JPA/Hibernate native queries with string concatenation
- Flag JPQL queries built with string concatenation

### Exception Handling
- Detect `catch (Exception e) {}` — swallowed exceptions
- Flag `catch (Throwable t)` — too broad, catches OutOfMemoryError
- Watch for `finally` blocks that throw exceptions (masks original)
- Detect exceptions used for flow control
"""
