# Litestar Native Features Reference

Comprehensive reference of Litestar's built-in capabilities for detecting NIH (Not Invented Here) syndrome.

## 1. Application Configuration

### Core App Features
| Feature | Native Component | NIH Detection |
|---------|-----------------|---------------|
| **Application Factory** | `Litestar()` class with `create_app()` pattern | Manual app instantiation with custom wiring |
| **Configuration Management** | `BaseSettings` from Pydantic Settings | Custom config dicts, env var parsing |
| **Debug Mode** | `debug=True/False` parameter | Manual DEBUG flags, custom logging setups |
| **Lifecycle Hooks** | `on_startup`, `on_shutdown`, `before_send` | Manual signal handlers, custom decorators |
| **Event Emitters** | `litestar.events.emitter` | Custom event buses, pub/sub implementations |

### Detecting NIH:
```python
# NIH - Custom config handling
class Config:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"

# NATIVE - Use Pydantic Settings
class Settings(BaseSettings):
    database_url: str
    debug: bool = False
    model_config = SettingsConfigDict(env_file=".env")
```

## 2. Middleware (High NIH Risk Area)

**Rule:** Custom ASGI middleware for standard HTTP/Security concerns is NIH.

### Built-in Middleware
| Feature | Native Component | NIH Detection |
|---------|-----------------|---------------|
| **CORS** | `litestar.config.cors.CORSConfig` | Custom CORSMiddleware class |
| **CSRF Protection** | `litestar.config.csrf.CSRFConfig` | Manual token validation |
| **Allowed Hosts** | `litestar.config.allowed_hosts.AllowedHostsConfig` | Manual host checking |
| **Compression** | `litestar.config.compression.CompressionConfig` | Custom compression middleware |
| **Rate Limiting** | `litestar.middleware.rate_limit.RateLimitConfig` | Custom rate limiting logic |
| **Logging** | `litestar.middleware.logging.LoggingMiddlewareConfig` | Manual request logging |
| **Sessions (Client-side)** | `litestar.middleware.session.client_side.CookieBackendConfig` | Manual cookie handling |
| **Sessions (Server-side)** | `litestar.middleware.session.server_side.ServerSideSessionConfig` | Custom session storage |
| **Request Body Cache** | `litestar.middleware.compression.RequestEncodingMiddleware` | Manual body buffering |

### NIH Examples to Flag:
```python
# NIH - Custom CORS middleware
class CORSMiddleware:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = scope.get("headers", [])
            # Manual origin checking...

# NATIVE - Use CORSConfig
from litestar.config.cors import CORSConfig

app = Litestar(
    cors_config=CORSConfig(
        allow_origins=["https://example.com"],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        allow_credentials=True,
    )
)
```

```python
# NIH - Custom rate limiting
from redis import Redis
redis_client = Redis()

async def rate_limit_middleware(scope, receive, send):
    client_ip = scope.get("client", [None])[0]
    key = f"rate_limit:{client_ip}"
    current = redis_client.incr(key)
    if current > 100:
        raise HTTPException(status_code=429)

# NATIVE - Use RateLimitConfig
from litestar.middleware.rate_limit import RateLimitConfig

app = Litestar(
    middleware=[
        RateLimitConfig(
            rate_limit=("minute", 100),
            exclude=["/health"],
        ).middleware
    ]
)
```

```python
# NIH - Manual session handling
@app.post("/login")
async def login(request: Request) -> dict:
    session_id = str(uuid.uuid4())
    redis_client.setex(f"session:{session_id}", 3600, json.dumps(user_data))
    response = Response({"message": "Logged in"})
    response.set_cookie("session_id", session_id, httponly=True)
    return response

# NATIVE - Use Server-side Sessions
from litestar.middleware.session.server_side import ServerSideSessionConfig
from litestar.stores.redis import RedisStore

store = RedisStore.with_client(url="redis://localhost")
app = Litestar(
    middleware=[
        ServerSideSessionConfig(store=store).middleware
    ]
)

@app.post("/login")
async def login(request: Request) -> dict:
    request.session["user_id"] = user.id  # Automatic session management
    return {"message": "Logged in"}
```

## 3. Storage (Stores)

**Rule:** Use `Store` abstraction for transient data, not direct Redis/Memcached/File clients.

### Built-in Store Backends
| Backend | Use Case | NIH Detection |
|---------|----------|---------------|
| **MemoryStore** | Single-worker caching, testing | Custom dict-based cache |
| **FileStore** | Persistent local storage | Manual file I/O for caching |
| **RedisStore** | Distributed caching, sessions | Direct redis-py usage |
| **ValkeyStore** | Redis alternative | Direct valkey client usage |

### Store Registry
Access stores via: `app.stores.get("name")` or inject via DI

### NIH Examples to Flag:
```python
# NIH - Direct Redis usage for caching
from redis import Redis
redis_client = Redis()

@app.get("/users")
async def get_users() -> list[User]:
    cached = redis_client.get("users")
    if cached:
        return json.loads(cached)
    users = await fetch_users()
    redis_client.setex("users", 300, json.dumps(users))
    return users

# NATIVE - Use RedisStore
from litestar.stores.redis import RedisStore

store = RedisStore.with_client(url="redis://localhost")
app = Litestar(stores={"cache": store})

@app.get("/users", cache=300)
async def get_users() -> list[User]:
    return await fetch_users()  # Automatic caching
```

```python
# NIH - Manual file-based caching
import pickle
from pathlib import Path

def get_cached_data(key: str):
    cache_file = Path(f"/tmp/cache/{key}.pkl")
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    return None

# NATIVE - Use FileStore
from litestar.stores.file import FileStore

store = FileStore(path="/tmp/litestar-cache")
app = Litestar(stores={"cache": store})

# Use with caching decorator or manual store access
```

## 4. Data Transfer Objects (DTOs)

**Rule:** Use DTOs for all request/response transformation. Manual dict slicing is NIH.

### DTO Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **DTO Classes** | `DataclassDTO`, `PydanticDTO`, `MsgspecDTO` | Manual dataclass/pydantic conversion |
| **Partial Updates** | `DTOConfig(partial=True)` | Manual field validation for PATCH |
| **Field Exclusion** | `DTOConfig(exclude={"field"})` | Manual dict key deletion |
| **Renaming Strategy** | `DTOConfig(rename_strategy="camel")` | Manual key transformation |
| **Max Depth** | `DTOConfig(max_nested_depth=2)` | Manual nested validation |

### NIH Examples to Flag:
```python
# NIH - Manual request parsing
@app.post("/users")
async def create_user(request: Request) -> dict:
    data = await request.json()
    # Manual validation
    if "email" not in data or "@" not in data["email"]:
        raise HTTPException(400, "Invalid email")
    # Manual transformation
    user = User(
        email=data["email"],
        name=data.get("name", ""),
        created_at=datetime.now(),
    )
    await db.save(user)
    # Manual response filtering
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
    }

# NATIVE - Use DTOs
@dataclass
class UserCreate:
    email: str
    name: str = ""

@dataclass
class UserRead:
    id: UUID
    email: str
    name: str
    created_at: datetime

class UserCreateDTO(DataclassDTO[UserCreate]):
    config = DTOConfig()

class UserReadDTO(DataclassDTO[UserRead]):
    config = DTOConfig(exclude={"password_hash"})

class UserController(Controller):
    dto = UserCreateDTO
    return_dto = UserReadDTO
    
    @post()
    async def create_user(self, data: DTOData[UserCreate]) -> User:
        return await user_service.create(data.create_instance())
```

```python
# NIH - Manual PATCH handling
@app.patch("/users/{user_id}")
async def update_user(user_id: UUID, request: Request) -> dict:
    data = await request.json()
    user = await get_user(user_id)
    # Manual partial update logic
    if "name" in data:
        user.name = data["name"]
    if "email" in data:
        user.email = data["email"]
    await db.save(user)
    return user.to_dict()

# NATIVE - Use partial DTOs
@dataclass
class UserUpdate:
    name: str | None = None
    email: str | None = None

class UserUpdateDTO(DataclassDTO[UserUpdate]):
    config = DTOConfig(partial=True)

class UserController(Controller):
    @patch("/{user_id:uuid}", dto=UserUpdateDTO)
    async def update_user(
        self, 
        user_id: UUID, 
        data: DTOData[UserUpdate]
    ) -> UserReadDTO:
        update_data = data.create_instance()
        return await user_service.update(user_id, update_data)
```

## 5. Dependency Injection

**Rule:** Use `Provide` and Litestar's DI system. Manual singletons or request-passing is NIH.

### DI Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Dependency Declaration** | `Provide(callable)` | Global singletons, manual instantiation |
| **Generator Dependencies** | `yield` for cleanup | Manual try/finally in handlers |
| **Dependency Override** | Layer-based override | Custom override logic |
| **Dependency Markers** | `Annotated[T, Dependency()]` | Manual OpenAPI exclusion |

### NIH Examples to Flag:
```python
# NIH - Global singletons
db = Database("postgresql://localhost/db")  # Global state

@app.get("/users")
async def get_users() -> list[User]:
    return await db.fetch_all("SELECT * FROM users")

# NATIVE - Use DI with generators
async def provide_db() -> AsyncGenerator[Database, None]:
    async with Database() as db:
        yield db

app = Litestar(dependencies={"db": Provide(provide_db)})

@app.get("/users")
async def get_users(db: Database) -> list[User]:
    return await db.fetch_all("SELECT * FROM users")
```

```python
# NIH - Passing Request through layers
class UserService:
    async def create(self, request: Request, data: dict) -> User:
        # Business logic accessing request directly
        user_agent = request.headers.get("user-agent")
        # ...

# NATIVE - Inject specific dependencies
async def provide_user_service(db: Database) -> UserService:
    return UserService(db)

class UserController(Controller):
    dependencies = {"user_service": Provide(provide_user_service)}
    
    @post()
    async def create_user(
        self, 
        data: DTOData[UserCreate],
        user_service: UserService,
    ) -> User:
        return await user_service.create(data.create_instance())
```

## 6. Security & Authentication

**Rule:** Use built-in security backends and guards. Manual auth checks are NIH.

### Security Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **JWT Auth** | `JWTCookieAuth`, `JWTAuth` | Manual JWT parsing/validation |
| **Session Auth** | `SessionAuth` | Manual session validation |
| **Basic Auth** | `BasicAuth` | Manual header parsing |
| **Bearer Token** | `BearerTokenAuth` | Manual bearer extraction |
| **Custom Backends** | `AbstractAuthenticationBackend` | Custom auth middleware |
| **Authorization** | `Guard` functions | Manual permission checks in handlers |

### NIH Examples to Flag:
```python
# NIH - Manual JWT handling
import jwt

@app.get("/protected")
async def protected_route(request: Request) -> dict:
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    
    token = auth_header[7:]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        user_id = payload["sub"]
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")
    
    return {"message": f"Hello {user_id}"}

# NATIVE - Use JWTCookieAuth
from litestar.security.jwt import JWTCookieAuth, Token

class UserToken(Token):
    sub: UUID
    email: str

async def retrieve_user(token: UserToken, connection: ASGIConnection) -> User | None:
    return await user_service.get(token.sub)

jwt_auth = JWTCookieAuth[User](
    retrieve_user_handler=retrieve_user,
    token_secret=SECRET_KEY,
    exclude=["/login", "/register"],
)

app = Litestar(on_app_init=[jwt_auth.on_app_init])

@app.get("/protected")
async def protected_route(request: Request[User, None, None]) -> dict:
    return {"message": f"Hello {request.user.email}"}  # User automatically injected
```

```python
# NIH - Manual permission checks
@app.delete("/users/{user_id}")
async def delete_user(request: Request, user_id: UUID) -> None:
    # Check authentication
    if not request.user:
        raise HTTPException(401)
    
    # Check authorization
    if request.user.role != "admin":
        raise HTTPException(403, "Admin required")
    
    await user_service.delete(user_id)

# NATIVE - Use Guards
from litestar.connection import ASGIConnection
from litestar.handlers.base import BaseRouteHandler
from litestar.exceptions import NotAuthorizedException, PermissionDeniedException

def admin_guard(connection: ASGIConnection, handler: BaseRouteHandler) -> None:
    if not connection.user:
        raise NotAuthorizedException()
    if connection.user.role != "admin":
        raise PermissionDeniedException("Admin access required")

class UserController(Controller):
    @delete("/{user_id:uuid}", guards=[admin_guard])
    async def delete_user(self, user_id: UUID) -> None:
        await user_service.delete(user_id)
```

## 7. Database Integration

**Rule:** Use Advanced Alchemy and Litestar plugins. Manual SQLAlchemy wiring is NIH.

### Database Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **SQLAlchemy Plugin** | `SQLAlchemyPlugin` | Manual engine/session management |
| **Repository Pattern** | `SQLAlchemyAsyncRepository` | Manual CRUD operations |
| **DTO Integration** | SQLAlchemy DTOs | Manual model conversion |
| **Async Support** | `AsyncSession`, `create_async_engine` | Synchronous database calls |

### NIH Examples to Flag:
```python
# NIH - Manual SQLAlchemy setup
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

engine = create_engine("postgresql://localhost/db")
SessionLocal = sessionmaker(bind=engine)

@app.get("/users")
async def get_users() -> list[dict]:
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return [{"id": u.id, "email": u.email} for u in users]
    finally:
        db.close()

# NATIVE - Use SQLAlchemyPlugin
from advanced_alchemy.extensions.litestar import SQLAlchemyAsyncConfig, SQLAlchemyPlugin

sqlalchemy_config = SQLAlchemyAsyncConfig(
    connection_string="postgresql+asyncpg://localhost/db",
    create_all=True,
)

app = Litestar(plugins=[SQLAlchemyPlugin(config=sqlalchemy_config)])

@app.get("/users")
async def get_users(db_session: AsyncSession) -> list[User]:
    return await db_session.execute(select(User)).scalars().all()
```

```python
# NIH - Manual CRUD operations
class UserService:
    async def create(self, db: AsyncSession, data: dict) -> User:
        user = User(**data)
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user
    
    async def get(self, db: AsyncSession, user_id: UUID) -> User | None:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

# NATIVE - Use Repository Pattern
from advanced_alchemy.repository import SQLAlchemyAsyncRepository

class UserRepository(SQLAlchemyAsyncRepository[User]):
    model_type = User

class UserService:
    def __init__(self, db_session: AsyncSession) -> None:
        self.repository = UserRepository(session=db_session)
    
    async def create(self, data: UserCreate) -> User:
        return await self.repository.add(data)
    
    async def get(self, user_id: UUID) -> User | None:
        return await self.repository.get_one_or_none(id=user_id)
```

## 8. Caching

**Rule:** Use Litestar's response caching and Store abstraction.

### Caching Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Response Caching** | `@get(cache=300)` | Manual cache decorators |
| **Cache Key Builder** | `cache_key_builder` | Manual cache key generation |
| **Store Backend** | Any `Store` implementation | Direct Redis/file caching |
| **Cache Invalidation** | Manual or TTL-based | Complex invalidation logic |

### NIH Examples to Flag:
```python
# NIH - Manual caching decorator
def cache_result(ttl: int = 300):
    def decorator(func):
        cache = {}
        async def wrapper(*args, **kwargs):
            key = f"{func.__name__}:{args}:{kwargs}"
            if key in cache and time.time() - cache[key]["time"] < ttl:
                return cache[key]["value"]
            result = await func(*args, **kwargs)
            cache[key] = {"value": result, "time": time.time()}
            return result
        return wrapper
    return decorator

# NATIVE - Use Litestar caching
@app.get("/expensive-query", cache=300)
async def expensive_query() -> dict:
    return await perform_expensive_operation()
```

## 9. WebSockets

**Rule:** Use Litestar's WebSocket support. Manual ASGI websocket handling is NIH.

### WebSocket Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Class-based Handler** | `WebsocketListener` | Manual ASGI websocket scope handling |
| **Function-based Handler** | `@websocket` decorator | Manual accept/receive/send loops |
| **Connection Management** | Built-in connection handling | Manual connection tracking |

### NIH Examples to Flag:
```python
# NIH - Manual websocket handling
async def websocket_endpoint(scope, receive, send):
    if scope["type"] == "websocket":
        await send({"type": "websocket.accept"})
        while True:
            message = await receive()
            if message["type"] == "websocket.receive":
                await send({"type": "websocket.send", "text": f"Echo: {message['text']}"})
            elif message["type"] == "websocket.disconnect":
                break

# NATIVE - Use WebsocketListener
from litestar.handlers.websocket_handlers import WebsocketListener
from litestar import WebSocket

class ChatHandler(WebsocketListener):
    path = "/ws/chat"
    
    async def on_accept(self, socket: WebSocket) -> None:
        await socket.accept()
    
    async def on_receive(self, socket: WebSocket, data: str) -> str:
        return f"Echo: {data}"
    
    async def on_disconnect(self, socket: WebSocket) -> None:
        pass
```

## 10. Background Tasks

**Rule:** Use Litestar's BackgroundTask system. Manual threading/asyncio task creation is NIH.

### Background Task Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Single Task** | `BackgroundTask` | Manual asyncio.create_task |
| **Multiple Tasks** | `BackgroundTasks` | Manual thread pool executors |
| **Task Dependencies** | Task receives injected dependencies | Manual dependency passing |

### NIH Examples to Flag:
```python
# NIH - Manual background task
import asyncio

@app.post("/users")
async def create_user(data: UserCreate) -> User:
    user = await user_service.create(data)
    # Fire and forget (bad pattern)
    asyncio.create_task(send_welcome_email(user.email))
    return user

# NATIVE - Use BackgroundTask
from litestar.background_tasks import BackgroundTask

async def send_welcome_email(email: str) -> None:
    # Send email logic
    pass

@app.post("/users")
async def create_user(data: UserCreate) -> Response[User]:
    user = await user_service.create(data)
    task = BackgroundTask(send_welcome_email, email=user.email)
    return Response(content=user, background=task)
```

## 11. Exception Handling

**Rule:** Use custom exceptions with handlers. Manual error responses are NIH.

### Exception Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Custom Exceptions** | Subclass `HTTPException` | Manual error dict creation |
| **Exception Handlers** | `exception_handlers` mapping | Manual try/except in handlers |
| **Problem Details** | RFC 9457 compliance | Custom error formats |

### NIH Examples to Flag:
```python
# NIH - Manual error handling
@app.get("/users/{user_id}")
async def get_user(user_id: UUID) -> dict:
    try:
        user = await user_service.get(user_id)
        if not user:
            return {"error": "Not found", "status": 404}
        return {"id": str(user.id), "email": user.email}
    except Exception as e:
        return {"error": str(e), "status": 500}

# NATIVE - Use custom exceptions and handlers
from litestar.exceptions import HTTPException

class UserNotFoundException(HTTPException):
    status_code = HTTP_404_NOT_FOUND
    detail = "User not found"

class UserController(Controller):
    @get("/{user_id:uuid}")
    async def get_user(self, user_id: UUID) -> User:
        user = await user_service.get(user_id)
        if not user:
            raise UserNotFoundException()
        return user

# Exception handler at app level
exception_handlers = {
    UserNotFoundException: lambda req, exc: Response(
        content={"error": exc.detail, "status": exc.status_code},
        status_code=exc.status_code,
    )
}

app = Litestar(exception_handlers=exception_handlers)
```

## 12. Template Engines

**Rule:** Use Litestar's template configuration. Manual Jinja2/Mako setup is NIH.

### Template Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Jinja2** | `litestar.template.TemplateConfig` | Manual Jinja2 environment setup |
| **Mako** | `litestar.template.TemplateConfig` | Manual Mako template lookup |
| **Template Response** | `Template` return type | Manual template rendering |

### NIH Examples to Flag:
```python
# NIH - Manual Jinja2 setup
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("templates"))

@app.get("/page")
async def render_page() -> Response:
    template = env.get_template("page.html")
    html = template.render(title="My Page")
    return Response(content=html, media_type=MediaType.HTML)

# NATIVE - Use TemplateConfig
from litestar.template import TemplateConfig
from litestar.contrib.jinja import JinjaTemplateEngine

app = Litestar(
    template_config=TemplateConfig(
        directory="templates",
        engine=JinjaTemplateEngine,
    )
)

@app.get("/page")
async def render_page() -> Template:
    return Template(template_name="page.html", context={"title": "My Page"})
```

## 13. Static Files

**Rule:** Use Litestar's static files configuration. Manual file serving is NIH.

### Static File Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Static Files** | `StaticFilesConfig` | Manual file reading and responses |
| **File Serving** | Built-in MIME type detection | Manual content-type headers |
| **SPA Support** | `html_mode=True` for SPAs | Manual index.html serving |

### NIH Examples to Flag:
```python
# NIH - Manual static file serving
from pathlib import Path

@app.get("/static/{file_path:path}")
async def serve_static(file_path: str) -> Response:
    full_path = Path("static") / file_path
    if not full_path.exists():
        raise HTTPException(404)
    
    content = full_path.read_bytes()
    mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    return Response(content=content, media_type=mime_type)

# NATIVE - Use StaticFilesConfig
from litestar.static_files import StaticFilesConfig

app = Litestar(
    static_files_config=[
        StaticFilesConfig(
            directories=["static"],
            path="/static",
        )
    ]
)
```

## 14. OpenAPI / Documentation

**Rule:** Use Litestar's automatic OpenAPI generation. Manual schema definition is NIH.

### OpenAPI Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Auto Schema Gen** | `OpenAPIConfig` | Manual OpenAPI spec files |
| **Documentation UI** | Scalar, Swagger-UI, ReDoc, etc. | Manual API documentation |
| **Examples** | `polyfactory` integration | Manual example creation |
| **Schema Customization** | `openapi_config` parameters | Manual schema overrides |

### NIH Examples to Flag:
```python
# NIH - Manual OpenAPI spec
openapi_spec = {
    "openapi": "3.1.0",
    "paths": {
        "/users": {
            "get": {
                "responses": {
                    "200": {"description": "List of users"}
                }
            }
        }
    }
}

# NATIVE - Auto-generated from types
app = Litestar(
    openapi_config=OpenAPIConfig(
        title="My API",
        version="1.0.0",
        description="Auto-generated documentation",
    )
)
```

## 15. Testing Utilities

**Rule:** Use Litestar's testing utilities. Manual ASGI test client setup is NIH.

### Testing Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Test Client** | `AsyncTestClient`, `TestClient` | Manual httpx/requests setup |
| **Test App Factory** | `create_test_client` | Manual app instantiation |
| **Dependency Override** | `dependencies` parameter in test client | Manual dependency mocking |

### NIH Examples to Flag:
```python
# NIH - Manual testing setup
import httpx

async def test_create_user():
    async with httpx.AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/users", json={"email": "test@test.com"})
        assert response.status_code == 201

# NATIVE - Use AsyncTestClient
from litestar.testing import AsyncTestClient

async def test_create_user(client: AsyncTestClient) -> None:
    response = await client.post("/users", json={"email": "test@test.com"})
    assert response.status_code == 201
```

## Quick Reference: NIH Detection Checklist

When auditing, look for these red flags:

### High Priority (Always NIH)
- [ ] Custom CORS/CSRF/session middleware
- [ ] Manual JWT parsing/validation
- [ ] Direct Redis/Memcached usage for caching/sessions
- [ ] Global singleton database connections
- [ ] Manual request.json() parsing without DTOs
- [ ] Manual dict slicing for response filtering
- [ ] Custom rate limiting implementations
- [ ] Manual Redis pub/sub for broadcasting
- [ ] Custom error response formats (vs ProblemDetailsPlugin)
- [ ] Manual flash message session handling
- [ ] Direct S3/boto3 client usage (vs fsspec)

### Medium Priority (Often NIH)
- [ ] Manual exception handling with error dicts
- [ ] Manual background task creation
- [ ] Manual Jinja2/static file setup
- [ ] Manual websocket scope handling
- [ ] Custom compression middleware
- [ ] Manual OpenAPI spec files
- [ ] Custom event streaming systems
- [ ] Manual pagination metadata calculation
- [ ] Manual store key namespacing

### Low Priority (Context-dependent)
- [ ] Custom logging middleware (vs LoggingMiddlewareConfig)
- [ ] Manual testing clients
- [ ] Custom template filters/context processors
- [ ] Manual file uploads handling
- [ ] Manual file system symlink resolution
- [ ] Manual store TTL management

## 16. Channels (Pub/Sub System)

**Rule:** Use Litestar's Channels for pub/sub messaging. Manual event systems are NIH.

### Channels Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Pub/Sub Backend** | `ChannelsPlugin` | Custom Redis pub/sub wiring |
| **WebSocket Broadcasting** | Built-in channel support | Manual WebSocket client tracking |
| **Backpressure** | Built-in backlog management | Manual queue management |
| **History** | Per-channel message history | Manual message storage |
| **Multiple Backends** | Memory, Redis Pub/Sub, Redis Streams, Postgres | Custom backend implementations |

### NIH Examples to Flag:
```python
# NIH - Manual Redis pub/sub for WebSocket broadcasting
import redis
from typing import Dict, List

connected_clients: Dict[str, List[WebSocket]] = {}
redis_client = redis.Redis()

async def broadcast_message(channel: str, message: str) -> None:
    # Manual client tracking
    for ws in connected_clients.get(channel, []):
        await ws.send_text(message)
    
    # Manual Redis publishing
    redis_client.publish(channel, message)

@app.websocket("/ws/{channel}")
async def websocket_handler(socket: WebSocket, channel: str) -> None:
    await socket.accept()
    if channel not in connected_clients:
        connected_clients[channel] = []
    connected_clients[channel].append(socket)
    
    try:
        while True:
            message = await socket.receive_text()
            await broadcast_message(channel, message)
    except Exception:
        connected_clients[channel].remove(socket)

# NATIVE - Use ChannelsPlugin
from litestar.channels import ChannelsPlugin
from litestar.channels.backends.memory import MemoryChannelsBackend

channels_plugin = ChannelsPlugin(
    backend=MemoryChannelsBackend(history=20),
    channels=["chat", "notifications"],
)

@app.websocket("/ws")
async def websocket_handler(socket: WebSocket, channels: ChannelsPlugin) -> None:
    await socket.accept()
    
    async with channels.start_subscription(["chat"]) as subscriber:
        async for message in subscriber.iter_events():
            await socket.send_text(message)

# Publishing
@app.post("/message")
async def post_message(channels: ChannelsPlugin) -> None:
    channels.publish({"text": "Hello"}, "chat")
```

## 17. Problem Details (RFC 9457)

**Rule:** Use ProblemDetailsPlugin for standardized error responses. Custom error formats are NIH.

### Problem Details Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **RFC 9457 Compliance** | `ProblemDetailsPlugin` | Custom error response formats |
| **Type URLs** | Standardized error types | String error codes |
| **Extra Fields** | `extra` dict for context | Manual error detail construction |
| **Exception Mapping** | `exception_to_problem_detail_map` | Manual exception conversion |

### NIH Examples to Flag:
```python
# NIH - Custom error response format
class ValidationError(HTTPException):
    status_code = 422

@app.exception_handler(ValidationError)
async def validation_handler(request: Request, exc: ValidationError) -> Response:
    return Response(
        content={
            "error_code": "VALIDATION_ERROR",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
        },
        status_code=422,
    )

# NATIVE - Use ProblemDetailsPlugin
from litestar.plugins.problem_details import ProblemDetailsPlugin, ProblemDetailsConfig, ProblemDetailsException

problem_details_plugin = ProblemDetailsPlugin(ProblemDetailsConfig())
app = Litestar(plugins=[problem_details_plugin])

@app.post("/purchase")
async def purchase(data: PurchaseData) -> None:
    if data.amount > user.balance:
        raise ProblemDetailsException(
            type_="https://example.com/probs/out-of-credit",
            title="You do not have enough credit",
            detail=f"Your balance is {user.balance}, but that costs {data.amount}",
            instance=f"/account/{user.id}/msgs/{msg_id}",
            extra={"balance": user.balance, "required": data.amount},
        )
```

## 18. Flash Messages

**Rule:** Use FlashPlugin for one-time session messages. Manual session message handling is NIH.

### Flash Messages Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **One-time Messages** | `FlashPlugin` | Manual session message management |
| **Template Integration** | Built-in template helpers | Manual session parsing in templates |
| **Categories** | Message categories (success, error, etc.) | Manual message type handling |
| **Auto-clear** | Automatic message consumption | Manual message deletion |

### NIH Examples to Flag:
```python
# NIH - Manual flash messages
@app.post("/login")
async def login(request: Request) -> Redirect:
    # Manual flash message storage
    if not request.session.get("flash_messages"):
        request.session["flash_messages"] = []
    request.session["flash_messages"].append({
        "message": "Login successful",
        "category": "success",
    })
    return Redirect("/dashboard")

@app.get("/dashboard")
async def dashboard(request: Request) -> Template:
    # Manual flash message retrieval and cleanup
    messages = request.session.pop("flash_messages", [])
    return Template("dashboard.html", context={"messages": messages})

# NATIVE - Use FlashPlugin
from litestar.plugins.flash import FlashPlugin, FlashConfig, flash
from litestar.middleware.session.server_side import ServerSideSessionConfig

template_config = TemplateConfig(engine=JinjaTemplateEngine, directory="templates")
flash_plugin = FlashPlugin(config=FlashConfig(template_config=template_config))

app = Litestar(
    plugins=[flash_plugin],
    middleware=[ServerSideSessionConfig().middleware],
)

@app.post("/login")
async def login(request: Request) -> Redirect:
    flash(request, "Login successful!", category="success")
    return Redirect("/dashboard")

# In template: {% for message in get_flashes() %}
```

## 19. File Systems (fsspec Integration)

**Rule:** Use Litestar's file system abstraction for remote storage. Manual file handling is NIH.

### File System Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **fsspec Integration** | Built-in fsspec support | Manual boto3 (S3) / gcs client setup |
| **Registry Pattern** | `FileSystemRegistry` | Manual file system instantiation |
| **Async Support** | Async file operations | Synchronous file I/O in async handlers |
| **Symlink Support** | `LinkableFileSystem` | Manual symlink resolution |

### NIH Examples to Flag:
```python
# NIH - Manual S3 file handling
import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client("s3")

@app.get("/files/{filename}")
async def get_file(filename: str) -> Response:
    try:
        response = s3_client.get_object(Bucket="my-bucket", Key=filename)
        content = response["Body"].read()
        return Response(content=content, media_type="application/octet-stream")
    except ClientError as e:
        raise HTTPException(404, "File not found")

# NATIVE - Use fsspec with FileSystemRegistry
import fsspec
from litestar.file_system import FileSystemRegistry
from litestar.response import File

s3_fs = fsspec.filesystem("s3", asynchronous=True)

@app.get("/files/{filename:str}")
async def get_file(filename: str) -> File:
    return File(filename, file_system="assets")

app = Litestar(
    route_handlers=[get_file],
    plugins=[FileSystemRegistry({"assets": s3_fs})],
)
```

```python
# NIH - Manual local file serving with path traversal checks
from pathlib import Path

@app.get("/download/{filepath:path}")
async def download_file(filepath: str) -> Response:
    base_path = Path("/var/www/files")
    full_path = (base_path / filepath).resolve()
    
    # Manual path traversal check
    if not str(full_path).startswith(str(base_path)):
        raise HTTPException(403, "Access denied")
    
    if not full_path.exists():
        raise HTTPException(404)
    
    content = full_path.read_bytes()
    return Response(content=content, media_type="application/octet-stream")

# NATIVE - Use FileSystemRegistry with safety built-in
@app.get("/download/{filepath:path}")
async def download_file(filepath: str) -> File:
    return File(filepath, file_system="downloads")

app = Litestar(
    plugins=[FileSystemRegistry({"downloads": FileStore(path="/var/www/files")})],
)
```

## 20. Store Namespacing & Advanced Features

**Rule:** Use Litestar's Store features properly. Manual key management is NIH.

### Store Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Namespacing** | `namespace` parameter | Manual key prefixing |
| **Expiration** | `expires_in` parameter | Manual TTL tracking |
| **Renewal** | `renew_for` parameter | Manual expiration refreshing |
| **Bulk Operations** | `get_many`, `set_many`, `delete_many` | Multiple individual operations |

### NIH Examples to Flag:
```python
# NIH - Manual key namespacing
async def get_user_cache(user_id: str) -> dict | None:
    key = f"user_cache:{user_id}"  # Manual namespace
    return await redis_client.get(key)

async def set_user_cache(user_id: str, data: dict, ttl: int = 3600) -> None:
    key = f"user_cache:{user_id}"
    await redis_client.setex(key, ttl, json.dumps(data))

# NATIVE - Use Store namespacing
from litestar.stores.redis import RedisStore

store = RedisStore.with_client(url="redis://localhost").with_namespace("user_cache")

async def get_user_cache(user_id: str) -> bytes | None:
    return await store.get(user_id)

async def set_user_cache(user_id: str, data: bytes) -> None:
    await store.set(user_id, data, expires_in=3600)

# NIH - Manual TTL refresh
@app.get("/session-data")
async def get_session_data(request: Request) -> dict:
    session_key = f"session:{request.session_id}"
    data = await redis_client.get(session_key)
    
    # Manual TTL refresh
    await redis_client.expire(session_key, 3600)
    
    return json.loads(data)

# NATIVE - Use renew_for
@app.get("/session-data")
async def get_session_data(request: Request, store: Store) -> bytes | None:
    # Automatically extends expiration on access
    return await store.get(request.session_id, renew_for=3600)
```

## 21. Pagination

**Rule:** Use Litestar's built-in pagination. Manual offset/limit handling is NIH.

### Pagination Features
| Feature | Component | NIH Detection |
|---------|-----------|---------------|
| **Offset Pagination** | `OffsetPagination` | Manual OFFSET/LIMIT in queries |
| **Cursor Pagination** | `CursorPagination` | Manual cursor handling |
| **Automatic Metadata** | Total count, next/prev links | Manual pagination metadata |

### NIH Examples to Flag:
```python
# NIH - Manual pagination
@app.get("/users")
async def list_users(page: int = 1, per_page: int = 20) -> dict:
    offset = (page - 1) * per_page
    
    users = await db.execute(
        select(User).offset(offset).limit(per_page)
    ).scalars().all()
    
    total = await db.execute(select(func.count(User.id))).scalar()
    
    return {
        "items": users,
        "page": page,
        "per_page": per_page,
        "total": total,
        "pages": (total + per_page - 1) // per_page,
    }

# NATIVE - Use OffsetPagination
from litestar.pagination import OffsetPagination

@app.get("/users")
async def list_users(
    offset: int = 0,
    limit: int = 20,
) -> OffsetPagination[User]:
    users = await user_service.list(offset=offset, limit=limit)
    total = await user_service.count()
    
    return OffsetPagination(
        items=users,
        total=total,
        offset=offset,
        limit=limit,
    )
```
