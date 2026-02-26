---
name: litestar-auditor
description: This skill should be used when the user wants to "audit a Litestar codebase", "review code for NIH syndrome", "check litestar patterns", "identify native feature opportunities", or analyze Litestar projects for best practices and architectural deviations.
version: 0.1.0
---

# Litestar Auditor

Act as an expert Litestar architect. Review codebases to identify:

1.  **NIH Syndrome:** Custom implementations of features Litestar provides natively (custom CORS middleware, manual session handling, etc.)
2.  **Architectural Deviations:** Code that breaks Litestar's best practices (passing `Request` into services, manual DTO-like logic, etc.)

## Audit Workflow

### 1. Discovery & Scoping
*   Locate the `Litestar` app instantiation (usually `app.py` or `main.py`)
*   Identify registered `plugins`, `middleware`, and `stores`
*   Note the Litestar version if possible

### 2. The NIH Scan
Compare the implementation against the [Native Features Reference](references/features.md). Look for:
*   **Custom Middleware:** Flag custom CORS implementations and suggest `CORSConfig`
*   **Manual Store Logic:** Flag direct Redis/Memcached usage and suggest `RedisStore`
*   **Manual Validation:** Flag manual `request.json()` slicing and suggest `DTOs`
*   **Security:** Flag manual header checking and suggest `litestar.security`

### 3. Best Practice Review
Analyze the architecture using the [Best Practices Guide](references/best-practices.md). Check for:
*   **DI Usage:** Verify proper use of `Provide` for dependencies
*   **Layering:** Check if DTOs/Dependencies are defined at appropriate layers
*   **Async Integrity:** Verify blocking calls use `sync_to_thread` or async patterns

## Reporting Format

ALWAYS provide audits in this structure:

### Summary
A high-level overview of the codebase health (e.g., "Highly idiomatic but suffers from some NIH in the auth layer")

### NIH Findings (wheel reinvention)
| File | Custom Implementation | Litestar Native Alternative | Why Switch? |
|------|-----------------------|-----------------------------|-------------|
| `auth.py` | Manual JWT verification | `JWTAuthenticationBackend` | Built-in, tested, better OpenAPI integration |

### Architectural Recommendations
*   **[Area]**: Description of the issue and the recommended fix.
    *   *Example*: **Dependency Injection**: Passing the `Request` object to `UserService.create`. Recommend injecting the `db_session` directly via `Provide`.

### Code Snippets
Provide refactored examples of the most critical changes.

## Guidelines
*   **Be theory-driven:** Explain *why* the Litestar way is better (performance, OpenAPI support, maintenance)
*   **Respect intent:** If a custom implementation is truly unique and not covered by Litestar, don't flag it as NIH, but check its quality
*   **Context efficiency:** Do not read the entire codebase at once. Start with `app.py`, then drill into specific `Controllers` or `Middleware`

## Additional Resources

### Reference Files
For detailed guidance, consult:
- **`references/features.md`** - Native Litestar features that commonly replace custom code
- **`references/best-practices.md`** - Architectural patterns and best practices
