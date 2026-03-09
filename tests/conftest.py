"""
Pytest fixtures for integration tests.

All tests run against real Docker services (PostgreSQL + AGE, Redis 8.4).
No mocks — we want to catch real bugs in query syntax, index schema, etc.

Start services before running tests:
    docker compose up -d
    pytest
"""

import os
import time

import psycopg2
import pytest
import redis as redis_lib

# ---------------------------------------------------------------------------
# Test environment — uses separate DB/index prefix to avoid polluting dev data
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://openclaw:openclaw@localhost:5433/openclaw_memory",
)
TEST_REDIS_URL = os.getenv(
    "TEST_REDIS_URL",
    "redis://localhost:6380/0",
)
TEST_REDIS_PREFIX = "test:"
TEST_GRAPH_NAME = "test_memory_graph"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def wait_for_postgres(dsn: str, retries: int = 20, delay: float = 1.0) -> None:
    """Wait until PostgreSQL is ready to accept connections."""
    import urllib.parse

    parsed = urllib.parse.urlparse(dsn)
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                database=parsed.path.lstrip("/"),
                connect_timeout=3,
            )
            conn.close()
            return
        except psycopg2.OperationalError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def wait_for_redis(url: str, retries: int = 20, delay: float = 1.0) -> None:
    """Wait until Redis is ready."""
    client = redis_lib.from_url(url)
    for attempt in range(retries):
        try:
            client.ping()
            client.close()
            return
        except (redis_lib.ConnectionError, redis_lib.ResponseError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)


# ---------------------------------------------------------------------------
# Session-scoped fixtures (one per test run)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres_conn():
    """
    Raw psycopg2 connection for AGE operations.
    Session-scoped — shared across all tests for speed.
    """
    wait_for_postgres(TEST_DATABASE_URL)

    import urllib.parse

    parsed = urllib.parse.urlparse(TEST_DATABASE_URL)
    conn = psycopg2.connect(
        host=parsed.hostname,
        port=parsed.port or 5432,
        user=parsed.username,
        password=parsed.password,
        database=parsed.path.lstrip("/"),
    )
    conn.autocommit = True

    # Load AGE and set search path
    with conn.cursor() as cur:
        cur.execute("LOAD 'age';")
        cur.execute("SET search_path = ag_catalog, \"$user\", public;")

    yield conn
    conn.close()


@pytest.fixture(scope="session")
def redis_client():
    """
    Redis client for the test suite.
    Session-scoped — shared across all tests.
    """
    wait_for_redis(TEST_REDIS_URL)
    client = redis_lib.from_url(TEST_REDIS_URL, decode_responses=False)
    yield client
    client.close()


# ---------------------------------------------------------------------------
# Function-scoped fixtures (fresh state per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_redis(redis_client):
    """
    Flush all test: prefixed keys before each test.
    Keeps test isolation without flushing the whole DB.
    """
    # Delete all test keys
    keys = redis_client.keys(f"{TEST_REDIS_PREFIX}*")
    if keys:
        redis_client.delete(*keys)

    # Also drop any test search indices
    try:
        redis_client.execute_command("FT.DROPINDEX", "test_memory_idx", "DD")
    except redis_lib.ResponseError:
        pass  # Index didn't exist

    yield redis_client

    # Cleanup after test
    keys = redis_client.keys(f"{TEST_REDIS_PREFIX}*")
    if keys:
        redis_client.delete(*keys)


@pytest.fixture
def clean_postgres(postgres_conn):
    """
    Drop and recreate test graph + tables before each test.
    """
    with postgres_conn.cursor() as cur:
        # Drop test graph if exists
        try:
            cur.execute(f"SELECT drop_graph('{TEST_GRAPH_NAME}', true);")
        except Exception:
            postgres_conn.rollback()

        # Clean test entities/relations
        try:
            cur.execute("DELETE FROM relations WHERE id LIKE 'test-%';")
            cur.execute("DELETE FROM entities WHERE id LIKE 'test-%';")
        except Exception:
            postgres_conn.rollback()

    yield postgres_conn


# ---------------------------------------------------------------------------
# Config override for tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def override_settings():
    """Override settings to point at test services."""
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL
    os.environ["REDIS_URL"] = TEST_REDIS_URL
    os.environ["GRAPH_NAME"] = TEST_GRAPH_NAME
    yield
