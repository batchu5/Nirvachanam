"""Built-in review rules for Python files.

Covers common Python-specific pitfalls that LLMs should look for
beyond generic code review. These are injected into the system prompt
when reviewing Python files.
"""

PYTHON_RULES = """## Python-Specific Review Rules

### Null-Safety / None Checks
- Flag attribute access on variables that could be `None` without a preceding guard
- Watch for `Optional[T]` return types used without `is None` checks
- Detect `dict.get()` results used without None handling
- Flag `len()` or iteration on possibly-None collections

### Resource Management
- Flag file handles, DB connections, sockets opened without `with` statement
- Detect `open()` without corresponding `close()` or context manager
- Watch for `asyncio.Lock` / `threading.Lock` acquired without `async with` / `with`
- Flag unclosed `httpx.AsyncClient` or `aiohttp.ClientSession` instances

### Type Safety
- Detect mixed-type operations (e.g., `str + int` without explicit conversion)
- Flag `isinstance()` checks against wrong types
- Watch for `Any` return types leaking into typed code paths
- Detect mutable default arguments (`def f(x=[])` anti-pattern)

### Async Pitfalls
- Flag `await` inside synchronous functions or missing `await` on coroutines
- Detect blocking I/O calls (`time.sleep`, `requests.get`) inside async functions
- Watch for `asyncio.run()` called inside an already-running event loop
- Flag fire-and-forget tasks without error handling (`asyncio.create_task` without storing ref)

### Exception Handling
- Flag bare `except:` or overly broad `except Exception:`
- Detect swallowed exceptions (empty `except` blocks)
- Watch for `raise` without `from` in exception chaining
- Flag `finally` blocks that could mask exceptions

### Import Safety
- Detect circular import patterns (imports inside functions as a workaround)
- Flag wildcard imports (`from module import *`)
- Watch for missing `__future__` annotations when using PEP 604 union types
"""
