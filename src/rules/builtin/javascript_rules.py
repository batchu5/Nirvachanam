"""Built-in review rules for JavaScript and TypeScript files.

Covers JS/TS-specific vulnerabilities and pitfalls that the LLM reviewer
should be especially vigilant about.
"""

JAVASCRIPT_RULES = """## JavaScript/TypeScript-Specific Review Rules

### XSS Prevention
- Flag direct use of `innerHTML`, `outerHTML`, or `document.write()` with dynamic content
- Detect `dangerouslySetInnerHTML` in React without sanitization
- Watch for template literal injection in DOM manipulation
- Flag unsanitized user input in `eval()`, `Function()`, or `setTimeout(string)`

### Prototype Pollution
- Flag `Object.assign()` or spread operator on untrusted objects without validation
- Detect recursive merge utilities that don't guard `__proto__`, `constructor`, `prototype`
- Watch for `JSON.parse()` of user input directly merged into config objects

### Async/Await Pitfalls
- Flag missing `await` on async function calls (silent promise drops)
- Detect `.catch()` without error handling (empty catch)
- Watch for `async` functions in `Array.forEach()` (doesn't await iterations)
- Flag promise constructor anti-pattern (wrapping existing promises in `new Promise`)

### Type Safety (TypeScript)
- Flag `any` type usage that bypasses TypeScript's safety
- Detect non-null assertions (`!`) on potentially null values
- Watch for type casts (`as Type`) that could hide runtime errors
- Flag `@ts-ignore` / `@ts-expect-error` without explanatory comments

### Security
- Detect regex DoS patterns (catastrophic backtracking with nested quantifiers)
- Flag `child_process.exec()` with string interpolation (command injection)
- Watch for `require()` / dynamic `import()` with user-controlled paths
- Detect insecure randomness (`Math.random()` for security-sensitive values)

### Common Bugs
- Flag `==` vs `===` comparisons (type coercion bugs)
- Detect off-by-one errors in array indexing
- Watch for `typeof null === 'object'` confusion
- Flag mutation of state directly in React (without setState/dispatch)
"""
