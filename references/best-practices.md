# Litestar Best Practices Guide

## 1. Controller Organization
*   **Keep Handlers Focused:** Route handlers should ideally handle a single HTTP method and path.
*   **Use Controllers for Grouping:** Group related handlers into a `Controller` class to share common paths, tags, dependencies, and DTOs.
*   **Avoid Logic in Handlers:** Business logic belongs in services/repositories. Handlers should focus on request/response mapping and calling services via Dependency Injection.

## 2. Layered Configuration
Litestar uses a layered system (App -> Router -> Controller -> Handler).
*   **DRY Configuration:** Define `dto`, `return_dto`, `dependencies`, `guards`, and `opt` at the highest relevant layer to avoid repetition.
*   **Selective Overrides:** Only override at the Handler level when strictly necessary.

## 3. Dependency Injection (DI)
*   **Inject Services, not Requests:** Use DI to inject service classes or database sessions. Avoid passing the `Request` object into deep service layers.
*   **Use `yield` for Cleanup:** Database sessions or file handles should be injected as `Generator` or `AsyncGenerator` to ensure `finally` blocks (cleanup) execute after the request.
*   **Decouple with Type Hints:** Inject interfaces or abstract classes where possible to make unit testing easier.

## 4. DTO Usage
*   **Prefer `DataclassDTO` or `MsgspecDTO`:** They offer the best performance (especially with the codegen backend).
*   **Split Read/Write DTOs:** Create separate DTOs for `POST`/`PUT` (Write) and `GET` (Read) to handle field exclusions like `id`, `created_at`, or password fields properly.
*   **Avoid Manual `dict` Conversion:** Let the DTO handle the conversion from request bytes to model objects.

## 5. Error Handling
*   **Custom Exceptions:** Subclass `litestar.exceptions.HTTPException` for domain-specific errors.
*   **Exception Handlers:** Define exception handlers at the App layer to map custom exceptions to consistent JSON responses.

## 6. Asynchronous Patterns
*   **Non-blocking code:** Ensure handlers and dependencies are `async` if they perform I/O.
*   **`sync_to_thread`:** Use this for legacy blocking code that cannot be made async, but avoid it for logic that *should* be async.
