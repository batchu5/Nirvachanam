"""Built-in review rules for Go files.

Covers Go-specific patterns and common pitfalls that
the LLM reviewer should flag.
"""

GO_RULES = """## Go-Specific Review Rules

### Nil Pointer Safety
- Flag pointer dereference without nil check
- Detect interface assertion without comma-ok pattern (`v := i.(T)` vs `v, ok := i.(T)`)
- Watch for nil map access (maps must be initialized with `make()` before write)
- Flag nil channel operations (send/receive on nil channel blocks forever)

### Error Handling
- Flag unchecked error return values (`_, err := f(); // err unused`)
- Detect errors silently discarded with `_ = f()` on error-returning functions
- Watch for `log.Fatal` / `os.Exit` in library code (should return error instead)
- Flag error wrapping without `%w` verb (breaks `errors.Is` / `errors.As`)

### Goroutine & Concurrency
- Detect goroutine leaks (goroutines without cancellation via context)
- Flag shared variable access without mutex or channel synchronization
- Watch for `sync.WaitGroup.Add()` called inside the goroutine instead of before
- Detect race conditions from closure variable capture in goroutine loops

### Resource Management
- Flag `defer` inside loops (defers pile up until function returns)
- Detect unclosed `http.Response.Body` (must close even on error)
- Watch for missing `defer rows.Close()` after database query
- Flag `os.File` opened without corresponding close/defer

### Thread Safety
- Detect `sync.Mutex` copied by value (should use pointer)
- Flag concurrent map read/write without `sync.RWMutex` or `sync.Map`
- Watch for `atomic` operations mixed with non-atomic access on same variable
"""
