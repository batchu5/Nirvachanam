"""Built-in review rules for SQL files and embedded SQL.

Covers SQL-specific security and correctness patterns.
"""

SQL_RULES = """## SQL-Specific Review Rules

### Injection Prevention
- Flag dynamic SQL construction with string concatenation or interpolation
- Detect `EXECUTE IMMEDIATE` with variable input (PL/SQL)
- Watch for `sp_executesql` with unsanitized parameters
- Flag `LIKE` patterns constructed from user input without escaping

### Data Integrity
- Detect missing `WHERE` clause in `UPDATE` / `DELETE` statements
- Flag `SELECT *` in production queries (should enumerate columns)
- Watch for implicit type conversions in `JOIN` / `WHERE` conditions
- Detect `TRUNCATE` without confirmation or audit logging
- Flag missing `NOT NULL` constraints on columns that should never be null

### Performance
- Detect `SELECT` inside loops (N+1 query pattern)
- Flag missing indexes implied by `WHERE` / `JOIN` / `ORDER BY` columns
- Watch for `DISTINCT` used as a band-aid for duplicate rows (fix the join)
- Detect correlated subqueries that could be rewritten as JOINs

### Access Control
- Flag `GRANT ALL` or overly permissive privilege assignments
- Detect DDL operations (CREATE, ALTER, DROP) in application code
- Watch for hardcoded credentials in connection strings
"""
