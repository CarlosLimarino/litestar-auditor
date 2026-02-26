# Litestar Best Practices Guide

Comprehensive guide for auditing Litestar codebases, including anti-patterns, architectural guidelines, and security considerations.

## 1. Layered Architecture

### Understanding the Layer System

Litestar uses a 4-layer hierarchy where parameters defined closer to the handler take precedence:

```
Application (Litestar)
  └── Router
        └── Controller
              └── Handler (@get, @post, etc.)
```

**Parameters supporting layering:** `dependencies`, `dto`, `return_dto`, `guards`, `middleware`, `exception_handlers`, `before_request`, `after_request`, `cache_control`, `response_headers`, `response_cookies`

### Best Practice: DRY Configuration

**DO:** Define common configurations at the highest relevant layer.

```python
# App level - applies to all routes
app = Litestar(
    dependencies={"db": Provide(provide_db)},
    exception_handlers={BusinessException: business_error_handler},
)

# Router level - applies to all routes in this router
api_router = Router(
    path="/api/v1",
    dependencies={"current_user": Provide(get_current_user)},
    guards=[authenticated_guard],
    route_handlers=[UserController, OrderController],
)

# Controller level - applies to all handlers in this controller
class AdminController(Controller):
    path = "/admin"
    guards=[admin_guard]  # Override router's guard
    dependencies={"audit_service": Provide(provide_audit_service)}

# Handler level - only this specific handler
@post("/users", guards=[superadmin_guard])  # Override controller's guard
async def create_user(self, data: UserCreate) -> User:
    pass
```

### Anti-Pattern: Repetitive Configuration

**DON'T:** Repeat the same configuration at every handler.

```python
# ANTI-PATTERN - Repetitive guards and DTOs
class UserController(Controller):
    @post("/users", dto=UserCreateDTO, guards=[auth_guard])
    async def create_user(self, data: UserCreate) -> User:
        pass
    
    @get("/users", dto=UserReadDTO, guards=[auth_guard])
    async def list_users(self) -> list[User]:
        pass
    
    @get("/users/{id}", dto=UserReadDTO, guards=[auth_guard])
    async def get_user(self, id: UUID) -> User:
        pass
    
    @put("/users/{id}", dto=UserUpdateDTO, guards=[auth_guard])
    async def update_user(self, id: UUID, data: UserUpdate) -> User:
        pass
```

**Audit Check:** Look for controllers with >3 handlers repeating the same `guards`, `dto`, or `dependencies`.

## 2. Dependency Injection Patterns

### Best Practice: Inject Services, Not Requests

**DO:** Pass specific dependencies to services, not the entire Request object.

```python
# GOOD - Service receives specific dependencies
class UserService:
    def __init__(self, db_session: AsyncSession) -> None:
        self.db = db_session
        self.repository = UserRepository(session=db_session)
    
    async def create(self, data: UserCreate) -> User:
        # Business logic here - no request handling
        return await self.repository.add(data)

# GOOD - Controller orchestrates request/service interaction
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

### Anti-Pattern: Request Bleed

**DON'T:** Pass Request objects through multiple layers.

```python
# ANTI-PATTERN - Request bleeds into service layer
class UserService:
    async def create(self, request: Request, data: dict) -> User:
        # Bad: Service knows about HTTP request
        user_agent = request.headers.get("user-agent")
        ip_address = request.client[0]
        
        # Business logic mixed with HTTP concerns
        user = await self.repository.create(data)
        await self.audit_log.log(
            user_id=user.id,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        return user

# Controller is forced to pass request
@post("/users")
async def create_user(request: Request, data: UserCreate) -> User:
    return await user_service.create(request, data)
```

**Audit Check:** Flag any service methods accepting `Request` parameters.

### Best Practice: Generator Dependencies for Cleanup

**DO:** Use `yield` in dependencies for proper resource cleanup.

```python
# GOOD - Generator ensures cleanup
async def provide_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

app = Litestar(dependencies={"db_session": Provide(provide_db_session)})
```

### Anti-Pattern: Manual Resource Management

**DON'T:** Manually manage resources without guaranteed cleanup.

```python
# ANTI-PATTERN - No guaranteed cleanup
async def provide_db_session() -> AsyncSession:
    session = async_session()
    return session  # Who closes this?

@app.post("/users")
async def create_user(db_session: AsyncSession) -> User:
    try:
        return await user_service.create(db_session, data)
    finally:
        await db_session.close()  # Easy to forget!
```

**Audit Check:** Look for dependencies returning resources without `yield`.

## 3. Data Transfer Objects (DTOs)

### Best Practice: Separate Read/Write DTOs

**DO:** Use different DTOs for different operations.

```python
# GOOD - Separate DTOs for each operation type
@dataclass
class UserCreate:
    email: str
    name: str
    password: str  # Only in create

@dataclass
class UserUpdate:
    name: str | None = None
    email: str | None = None
    # Password changes should be separate endpoint

@dataclass
class UserRead:
    id: UUID
    email: str
    name: str
    created_at: datetime
    is_active: bool
    # No password_hash!

class UserCreateDTO(DataclassDTO[UserCreate]):
    config = DTOConfig()

class UserUpdateDTO(DataclassDTO[UserUpdate]):
    config = DTOConfig(partial=True)

class UserReadDTO(DataclassDTO[UserRead]):
    config = DTOConfig(
        exclude={"password_hash"},  # Never expose
        rename_strategy="camel",    # API convention
    )

class UserController(Controller):
    dto = UserCreateDTO  # Default for POST/PUT
    return_dto = UserReadDTO  # Default for responses
    
    @post()
    async def create(self, data: DTOData[UserCreate]) -> User:
        pass
    
    @patch("/{id}", dto=UserUpdateDTO)
    async def update(self, id: UUID, data: DTOData[UserUpdate]) -> User:
        pass
    
    @get()
    async def list(self) -> list[User]:
        pass
```

### Anti-Pattern: Single DTO for Everything

**DON'T:** Use one DTO for all operations.

```python
# ANTI-PATTERN - Single DTO used everywhere
@dataclass
class UserDTO:
    id: UUID | None = None  # Optional for create?
    email: str = ""
    name: str = ""
    password: str = ""  # Included in responses!
    created_at: datetime | None = None
    is_active: bool = True

# Problems:
# - Password exposed in GET responses
# - ID shouldn't be in POST body
# - created_at shouldn't be user-provided
# - PATCH requires all fields
```

**Audit Check:** Look for DTOs used across multiple operation types (CREATE, READ, UPDATE).

### Best Practice: DTOConfig for Field Control

**DO:** Use DTOConfig features instead of manual manipulation.

```python
# GOOD - Use DTOConfig features
class UserReadDTO(DataclassDTO[UserRead]):
    config = DTOConfig(
        exclude={"password_hash", "internal_notes"},  # Security
        rename_strategy="camel",                       # Convention
        max_nested_depth=2,                            # Prevent over-fetching
    )

# For PATCH operations
class UserUpdateDTO(DataclassDTO[UserUpdate]):
    config = DTOConfig(partial=True)  # All fields optional
```

### Anti-Pattern: Manual Field Filtering

**DON'T:** Manually filter or transform fields.

```python
# ANTI-PATTERN - Manual response building
@app.get("/users/{id}")
async def get_user(id: UUID) -> dict:
    user = await user_service.get(id)
    # Manual filtering - easy to miss sensitive fields
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        # Oops, forgot to exclude password_hash!
    }
```

**Audit Check:** Look for handlers returning `dict` instead of typed objects with DTOs.

## 4. Controller Design

### Best Practice: Resource-Based Controllers

**DO:** Group by resource, follow REST conventions.

```python
# GOOD - Resource-based controller
class UserController(Controller):
    path = "/users"
    tags = ["Users"]
    
    @post()
    async def create(self, data: UserCreate) -> User:
        """Create a new user."""
        pass
    
    @get()
    async def list(self) -> list[User]:
        """List all users."""
        pass
    
    @get("/{id:uuid}")
    async def get(self, id: UUID) -> User:
        """Get a user by ID."""
        pass
    
    @patch("/{id:uuid}")
    async def update(self, id: UUID, data: UserUpdate) -> User:
        """Partially update a user."""
        pass
    
    @delete("/{id:uuid}")
    async def delete(self, id: UUID) -> None:
        """Delete a user."""
        pass
```

### Anti-Pattern: Action-Based Routes

**DON'T:** Create routes based on actions instead of resources.

```python
# ANTI-PATTERN - RPC-style endpoints
@app.post("/createUser")
async def create_user_endpoint(data: dict) -> dict:
    pass

@app.post("/getUserById")
async def get_user_endpoint(data: dict) -> dict:
    pass

@app.post("/updateUserInfo")
async def update_user_endpoint(data: dict) -> dict:
    pass
```

**Audit Check:** Look for POST endpoints with action names instead of resource paths.

### Best Practice: Thin Controllers, Rich Services

**DO:** Keep controllers focused on HTTP concerns.

```python
# GOOD - Controller handles HTTP, service handles business
class OrderController(Controller):
    @post()
    async def create_order(
        self,
        data: DTOData[OrderCreate],
        order_service: OrderService,
    ) -> Order:
        # Controller: HTTP input validation via DTO
        order_data = data.create_instance()
        
        # Service: Business logic
        order = await order_service.create(
            user_id=self.request.user.id,
            items=order_data.items,
            shipping_address=order_data.shipping_address,
        )
        
        return order
```

### Anti-Pattern: Fat Controllers

**DON'T:** Put business logic in controllers.

```python
# ANTI-PATTERN - Business logic in controller
@post("/orders")
async def create_order(data: OrderCreate) -> Order:
    # Validation (should be in DTO)
    if len(data.items) == 0:
        raise HTTPException(400, "Order must have items")
    
    # Price calculation (should be in service)
    total = 0
    for item in data.items:
        product = await db.get(Product, item.product_id)
        if not product:
            raise HTTPException(404, f"Product {item.product_id} not found")
        if product.stock < item.quantity:
            raise HTTPException(400, f"Insufficient stock for {product.name}")
        total += product.price * item.quantity
    
    # Discount logic (should be in service)
    if total > 100:
        total *= 0.9
    
    # Persistence (should be in repository)
    order = Order(
        user_id=request.user.id,
        items=data.items,
        total=total,
        status="pending",
    )
    db.add(order)
    await db.commit()
    
    return order
```

**Audit Check:** Controllers with >20 lines of business logic are too fat.

## 5. Asynchronous Patterns

### Best Practice: Async Throughout

**DO:** Use async for all I/O operations.

```python
# GOOD - Fully async
class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
    
    async def get(self, id: UUID) -> User | None:
        result = await self.session.execute(
            select(User).where(User.id == id)
        )
        return result.scalar_one_or_none()
    
    async def list(self) -> list[User]:
        result = await self.session.execute(select(User))
        return result.scalars().all()
```

### Anti-Pattern: Sync in Async

**DON'T:** Use blocking calls in async handlers.

```python
# ANTI-PATTERN - Blocking I/O in async handler
@app.post("/upload")
async def upload_file(data: UploadFile) -> dict:
    content = await data.read()
    
    # BLOCKING! This stops the event loop
    with open(f"/uploads/{data.filename}", "wb") as f:
        f.write(content)
    
    return {"filename": data.filename}
```

**Fix:** Use `sync_to_thread` or async alternatives.

```python
# GOOD - Run blocking code in thread
from litestar.utils import sync_to_thread

@app.post("/upload")
async def upload_file(data: UploadFile) -> dict:
    content = await data.read()
    
    def write_file():
        with open(f"/uploads/{data.filename}", "wb") as f:
            f.write(content)
    
    await sync_to_thread(write_file)
    return {"filename": data.filename}
```

**Audit Check:** Look for file I/O, synchronous HTTP requests, or CPU-intensive operations in async handlers.

### Best Practice: Proper Database Drivers

**DO:** Use async database drivers.

```python
# GOOD - Async database driver
# postgresql+asyncpg://
# mysql+aiomysql://
# sqlite+aiosqlite://

sqlalchemy_config = SQLAlchemyAsyncConfig(
    connection_string="postgresql+asyncpg://user:pass@localhost/db",
)
```

### Anti-Pattern: Sync Database in Async

**DON'T:** Use sync database drivers in async handlers.

```python
# ANTI-PATTERN - Sync SQLAlchemy in async app
from sqlalchemy import create_engine

engine = create_engine("postgresql://...")  # sync driver!
Session = sessionmaker(bind=engine)

@app.get("/users")
async def get_users() -> list[User]:
    db = Session()
    users = db.query(User).all()  # BLOCKING!
    return users
```

**Audit Check:** Check connection strings for sync drivers (missing `+asyncpg`, `+aiomysql`, etc.)

## 6. Error Handling

### Best Practice: Custom Exceptions + Handlers

**DO:** Use domain-specific exceptions with centralized handlers.

```python
# GOOD - Domain exceptions
class ResourceNotFoundException(HTTPException):
    status_code = HTTP_404_NOT_FOUND

class ValidationException(HTTPException):
    status_code = HTTP_422_UNPROCESSABLE_ENTITY

class BusinessLogicException(HTTPException):
    status_code = HTTP_400_BAD_REQUEST

# Centralized handlers
def resource_not_found_handler(request: Request, exc: ResourceNotFoundException) -> Response:
    return Response(
        content={
            "error": "NotFound",
            "message": str(exc.detail),
            "status_code": exc.status_code,
        },
        status_code=exc.status_code,
    )

app = Litestar(
    exception_handlers={
        ResourceNotFoundException: resource_not_found_handler,
        ValidationException: validation_handler,
        BusinessLogicException: business_logic_handler,
    }
)

# Usage in service layer
class UserService:
    async def get(self, id: UUID) -> User:
        user = await self.repository.get(id)
        if not user:
            raise ResourceNotFoundException(f"User {id} not found")
        return user
```

### Anti-Pattern: Generic Error Responses

**DON'T:** Return generic errors or inconsistent formats.

```python
# ANTI-PATTERN - Inconsistent error handling
@app.get("/users/{id}")
async def get_user(id: UUID) -> dict:
    try:
        user = await user_service.get(id)
        if not user:
            return {"error": "User not found", "code": 404}
        return {"data": user}
    except Exception as e:
        return {"error": str(e), "code": 500}

@app.post("/users")
async def create_user(data: dict) -> dict:
    try:
        user = await user_service.create(data)
        return {"user": user}
    except ValidationError as e:
        return {"errors": e.errors(), "status": "error"}
    except Exception as e:
        return {"message": "Something went wrong"}
```

**Audit Check:** Look for inconsistent error response formats across endpoints.

## 7. Security Patterns

### Best Practice: Defense in Depth

**DO:** Layer security at multiple levels.

```python
# GOOD - Multiple security layers
app = Litestar(
    # Layer 1: CORS
    cors_config=CORSConfig(
        allow_origins=["https://app.example.com"],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_credentials=True,
    ),
    # Layer 2: CSRF for state-changing operations
    csrf_config=CSRFConfig(secret=SECRET_KEY),
    # Layer 3: Allowed hosts
    allowed_hosts=AllowedHostsConfig(
        allowed_hosts=["api.example.com"],
    ),
)

# Layer 4: Authentication via JWT
jwt_auth = JWTCookieAuth[User](...)

# Layer 5: Authorization via Guards
class AdminController(Controller):
    path = "/admin"
    guards=[admin_guard]  # Applies to all handlers
    
    @delete("/users/{id}", guards=[superadmin_guard])
    async def delete_user(self, id: UUID) -> None:
        # Layer 6: Additional checks in service
        await audit_service.log_action("delete_user", id)
        await user_service.delete(id)
```

### Anti-Pattern: Security Only in Handlers

**DON'T:** Rely solely on handler-level checks.

```python
# ANTI-PATTERN - Security only in handler
@app.delete("/users/{id}")
async def delete_user(request: Request, id: UUID) -> None:
    # Single point of failure - easy to forget
    if request.user.role != "admin":
        raise HTTPException(403)
    
    await user_service.delete(id)

# ANTI-PATTERN - No CORS/CSRF protection
app = Litestar(
    cors_config=CORSConfig(allow_origins=["*"]),  # Too permissive
    # Missing CSRF config
    # Missing allowed_hosts
)
```

**Audit Check:** Verify security is configured at app level (CORS, CSRF, rate limiting).

### Best Practice: Proper Secret Management

**DO:** Use environment variables and settings.

```python
# GOOD - Secrets in environment
class Settings(BaseSettings):
    secret_key: str
    database_url: SecretStr  # Pydantic SecretStr hides value in logs
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    model_config = SettingsConfigDict(env_file=".env")
```

### Anti-Pattern: Hardcoded Secrets

**DON'T:** Hardcode secrets or expose them in code.

```python
# ANTI-PATTERN - Hardcoded secrets
SECRET_KEY = "my-super-secret-key-12345"  # In git!
DB_PASSWORD = "password123"  # In git!

app = Litestar(
    cors_config=CORSConfig(allow_origins=["*"]),  # Too permissive
)
```

**Audit Check:** Search for hardcoded passwords, API keys, or secrets.

## 8. Performance Considerations

### Best Practice: N+1 Prevention

**DO:** Use eager loading for relationships.

```python
# GOOD - Eager loading
class UserRepository:
    async def list_with_orders(self) -> list[User]:
        result = await self.session.execute(
            select(User)
            .options(selectinload(User.orders))  # Eager load
            .where(User.is_active == True)
        )
        return result.scalars().all()
```

### Anti-Pattern: N+1 Queries

**DON'T:** Access relationships in loops.

```python
# ANTI-PATTERN - N+1 query problem
@app.get("/users")
async def list_users() -> list[dict]:
    users = await db.execute(select(User)).scalars().all()
    
    result = []
    for user in users:  # N users
        user_data = {
            "id": user.id,
            "name": user.name,
            "orders": [
                {"id": o.id, "total": o.total} 
                for o in user.orders  # +1 query per user!
            ]
        }
        result.append(user_data)
    
    return result
```

**Audit Check:** Look for relationship access in loops without eager loading.

### Best Practice: Response Caching

**DO:** Use Litestar's caching for expensive operations.

```python
# GOOD - Response caching
@app.get("/products", cache=300)  # Cache for 5 minutes
async def list_products() -> list[Product]:
    return await product_service.list()

@app.get("/products/{id}", cache=60)
async def get_product(id: UUID) -> Product:
    return await product_service.get(id)
```

### Anti-Pattern: Repeated Expensive Queries

**DON'T:** Repeat expensive operations without caching.

```python
# ANTI-PATTERN - No caching
@app.get("/products")
async def list_products() -> list[Product]:
    # Expensive query runs every request
    return await db.execute(
        select(Product)
        .options(selectinload(Product.category))
        .where(Product.active == True)
    ).scalars().all()
```

## 9. Sync vs Async Patterns

### Best Practice: Use sync_to_thread for Blocking Operations

**DO:** Run blocking operations in a thread pool to prevent blocking the event loop.

```python
# GOOD - sync_to_thread for blocking I/O
from litestar.utils import sync_to_thread
import pandas as pd

@app.post("/analyze")
async def analyze_csv(file: UploadFile) -> dict:
    content = await file.read()
    
    def process_data():
        # This is CPU-intensive and blocking
        df = pd.read_csv(io.BytesIO(content))
        return df.describe().to_dict()
    
    # Run in thread pool to avoid blocking event loop
    result = await sync_to_thread(process_data)
    return result
```

### Best Practice: Declare sync_to_thread in Dependencies

**DO:** Explicitly declare if synchronous dependencies should run in threads.

```python
# GOOD - Explicit sync_to_thread declaration
from litestar.di import Provide

def get_cache_client() -> Redis:
    # This is synchronous but non-blocking
    return redis.Redis()

# sync_to_thread=False since it's non-blocking
app = Litestar(
    dependencies={"cache": Provide(get_cache_client, sync_to_thread=False)}
)

# For blocking operations
import requests

def fetch_external_data(url: str) -> dict:
    # This blocks - should use sync_to_thread
    return requests.get(url).json()

# sync_to_thread=True (or use httpx for async)
app = Litestar(
    dependencies={"external_api": Provide(fetch_external_data, sync_to_thread=True)}
)
```

### Anti-Pattern: Blocking the Event Loop

**DON'T:** Run blocking operations directly in async handlers.

```python
# ANTI-PATTERN - Blocking I/O blocks event loop
@app.post("/upload")
async def upload_file(data: UploadFile) -> dict:
    content = await data.read()
    
    # BLOCKING! Stops the entire application
    with open(f"/uploads/{data.filename}", "wb") as f:
        f.write(content)
    
    # BLOCKING! HTTP request without await
    response = requests.post("https://api.example.com/webhook", json={"file": data.filename})
    
    return {"filename": data.filename}
```

**Audit Check:** Look for:
- File operations (open, read, write)
- Synchronous HTTP libraries (requests, urllib)
- Database drivers without async support (psycopg2, pymongo without motor)
- CPU-intensive operations (pandas, numpy, image processing)

### Best Practice: Use Async Libraries

**DO:** Prefer async-native libraries.

```python
# GOOD - Async libraries
import httpx  # Instead of requests
import aiofiles  # Instead of built-in open
from sqlalchemy.ext.asyncio import create_async_engine  # Instead of create_engine
from aiobotocore.session import get_session  # Instead of boto3

@app.post("/upload")
async def upload_file(data: UploadFile) -> dict:
    # Async file operations
    async with aiofiles.open(f"/uploads/{data.filename}", "wb") as f:
        await f.write(await data.read())
    
    # Async HTTP
    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.example.com/webhook", json={"file": data.filename})
    
    return {"filename": data.filename}
```

## 10. Testing Best Practices

### Best Practice: Use Test Client with Overrides

**DO:** Use Litestar's test utilities with dependency overrides.

```python
# GOOD - Proper testing setup
import pytest
from litestar.testing import AsyncTestClient

@pytest.fixture
async def test_client():
    app = create_app()
    async with AsyncTestClient(app=app) as client:
        yield client

@pytest.mark.asyncio
async def test_create_user(test_client: AsyncTestClient) -> None:
    response = await test_client.post(
        "/users",
        json={"email": "test@example.com", "name": "Test"},
    )
    assert response.status_code == 201
    assert response.json()["email"] == "test@example.com"
```

### Anti-Pattern: Manual App Testing

**DON'T:** Create manual HTTP requests to test.

```python
# ANTI-PATTERN - Manual testing
import httpx

async def test_create_user():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/users",
            json={"email": "test@example.com"},
        )
        assert response.status_code == 201
```

**Audit Check:** Look for tests making actual HTTP requests instead of using TestClient.

## 11. Channels Best Practices

### Best Practice: Use start_subscription Context Manager

**DO:** Use the context manager for automatic cleanup.

```python
# GOOD - Automatic cleanup with context manager
@app.websocket("/ws")
async def websocket_handler(socket: WebSocket, channels: ChannelsPlugin) -> None:
    await socket.accept()
    
    async with channels.start_subscription(["updates"]) as subscriber:
        async for message in subscriber.iter_events():
            await socket.send_text(message)
```

### Anti-Pattern: Manual Subscription Management

**DON'T:** Manually subscribe/unsubscribe without guaranteed cleanup.

```python
# ANTI-PATTERN - No guaranteed cleanup
@app.websocket("/ws")
async def websocket_handler(socket: WebSocket, channels: ChannelsPlugin) -> None:
    await socket.accept()
    subscriber = await channels.subscribe(["updates"])
    
    # If exception occurs here, unsubscribe never called
    async for message in subscriber.iter_events():
        await socket.send_text(message)
    
    await channels.unsubscribe(subscriber)  # May never reach here
```

### Best Practice: Use run_in_background for Concurrent Operations

**DO:** Use run_in_background when you need to receive WebSocket messages while processing channel events.

```python
# GOOD - Concurrent channel and websocket handling
@app.websocket("/ws")
async def websocket_handler(socket: WebSocket, channels: ChannelsPlugin) -> None:
    await socket.accept()
    
    async with (
        channels.start_subscription(["broadcast"]) as subscriber,
        subscriber.run_in_background(socket.send_text),
    ):
        # Can receive client messages while channel events are processed
        while True:
            message = await socket.receive_text()
            await channels.publish(message, "echo")
```

**Audit Check:** Look for custom Redis pub/sub implementations when ChannelsPlugin should be used.

## 10. Common Code Smells (Audit Checklist)

### High Priority Issues

- [ ] **Request objects passed to services** - Breaks separation of concerns
- [ ] **Global state** - Singletons, module-level variables
- [ ] **Manual JSON parsing** - Not using DTOs
- [ ] **Manual authentication** - Not using built-in security backends
- [ ] **Sync database drivers** - Using psycopg2 instead of asyncpg
- [ ] **Blocking I/O in async** - File operations, HTTP requests without sync_to_thread
- [ ] **Missing sync_to_thread** - Dependencies performing blocking I/O
- [ ] **No input validation** - Accepting raw dicts without DTOs
- [ ] **Hardcoded secrets** - In code, not environment
- [ ] **Synchronous HTTP libraries** - Using requests/urllib instead of httpx

### Medium Priority Issues

- [ ] **Repetitive controller config** - Same guards/DTOs on every handler
- [ ] **Fat controllers** - Business logic in route handlers
- [ ] **No custom exceptions** - Using generic HTTPException everywhere
- [ ] **Manual caching** - Not using response caching or stores
- [ ] **N+1 queries** - Accessing relationships in loops
- [ ] **Missing type hints** - Incomplete or missing annotations
- [ ] **RPC-style routes** - POST /doSomething instead of REST
- [ ] **Manual middleware** - Custom implementations of built-in features
- [ ] **Manual pub/sub** - Custom Redis pub/sub instead of ChannelsPlugin
- [ ] **Missing sync_to_thread declaration** - Dependencies without explicit sync_to_thread

### Low Priority Issues

- [ ] **Missing docstrings** - No handler documentation
- [ ] **Inconsistent naming** - Mixed naming conventions
- [ ] **Too many dependencies** - Handlers with >5 injected dependencies
- [ ] **Deep nesting** - Services calling services calling services
- [ ] **Missing tests** - No test coverage for handlers
- [ ] **Over-eager loading** - Loading all relationships when not needed

## Audit Report Template

When auditing, organize findings by severity:

### Critical (Must Fix)
Security vulnerabilities, data integrity issues, or severe performance problems.

### High (Should Fix)
Clear violations of best practices that impact maintainability or scalability.

### Medium (Consider Fixing)
Deviations from idiomatic patterns that may cause issues later.

### Low (Nice to Have)
Style improvements or minor optimizations.

### Positive Findings
Also document what the codebase does well!

---

**Remember:** The goal is not to achieve perfect purity, but to ensure the codebase follows Litestar's design philosophy: leveraging native features for maintainability, type safety, and performance.
