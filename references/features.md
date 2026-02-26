# Litestar Native Features Reference

This document serves as a "ground truth" for auditing Litestar applications to detect NIH (Not Invented Here) syndrome and ensure best practices.

## 1. Middleware & Security
**Rule:** Avoid custom-built middleware for standard HTTP/Security concerns. Use Litestar's native configurations.

| Feature | Native Litestar Component | Recommended Configuration |
|---------|---------------------------|---------------------------|
| CORS | `litestar.config.cors.CORSConfig` | Pass to `Litestar(cors_config=...)` |
| CSRF | `litestar.config.csrf.CSRFConfig` | Pass to `Litestar(csrf_config=...)` |
| Allowed Hosts | `litestar.config.allowed_hosts.AllowedHostsConfig` | Pass to `Litestar(allowed_hosts=...)` |
| Compression | `litestar.config.compression.CompressionConfig` | Supports Gzip, Brotli, Zstd. |
| Rate Limiting | `litestar.middleware.rate_limit.RateLimitConfig` | Uses `Store` for persistence. |
| Logging Middleware | `litestar.middleware.logging.LoggingMiddlewareConfig` | Integrated with Litestar's `LoggingConfig`. |
| Sessions | `litestar.middleware.session.client_side.CookieBackendConfig` or `litestar.middleware.session.server_side.ServerSideSessionConfig` | Server-side needs a `Store`. |

## 2. Key-Value Storage (Stores)
**Rule:** Use the `Store` abstraction instead of direct Redis/Memcached/File clients for transient data (caching, sessions).

*   **Registry:** `app.stores.get("name")`
*   **Backends:**
    *   `MemoryStore`: Best for single-worker, low-overhead.
    *   `FileStore`: Persistent on-disk storage.
    *   `RedisStore`: Scalable, multi-worker support.
    *   `ValkeyStore`: Open-source Redis alternative.

## 3. Data Transfer Objects (DTOs)
**Rule:** Use DTOs for filtering data between the client and handlers. Avoid manual `dict` slicing or manual Pydantic filtering in business logic.

*   **Classes:** `DataclassDTO`, `PydanticDTO`, `MsgspecDTO`.
*   **Usage:** Define `dto` (inbound/outbound) and `return_dto` (outbound only) on Handlers, Controllers, or Routers.
*   **Configuration:** Use `DTOConfig` for `exclude`, `rename`, `partial` (for PATCH).

## 4. Dependency Injection
**Rule:** Use `Provide` and Litestar's DI system. Avoid global singleton dependencies or passing `request` objects through layers just to access services.

*   **Syntax:** `dependencies={"key": Provide(callable)}`
*   **Yield:** Use `Generator` or `AsyncGenerator` for cleanup (e.g., database sessions).
*   **Markers:** Use `Annotated[T, Dependency()]` to exclude from OpenAPI or fail early.

## 5. Built-in Plugins & Contrib
**Rule:** Check `litestar.contrib` or `litestar.plugins` before implementing integrations.

*   **SQLAlchemy:** Use `litestar.plugins.sqlalchemy.SQLAlchemyPlugin`.
*   **Templates:** Use `litestar.template.TemplateConfig` (Jinja2, Mako, MiniJinja).
*   **OpenAPI:** Use `litestar.openapi.OpenAPIConfig` and `litestar.openapi.plugins`.
