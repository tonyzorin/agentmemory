FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Build deps for psycopg2 (no binary wheel for Python 3.14)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only FIRST to avoid pulling in CUDA (~10GB).
# sentence-transformers will reuse this installation instead of pulling GPU wheels.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies (layer cache)
COPY pyproject.toml ./
RUN pip install -e ".[dev]"

# Pre-download the embedding model into the image so startup is instant.
# Must happen BEFORE COPY . . so this layer is cached across code-only changes.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-base-en-v1.5')"

# Copy source
COPY . .

# Default: run the MCP server in Streamable HTTP mode (supports both modern and legacy clients)
CMD ["python", "-m", "agentmemory.mcp.server", "--transport", "streamable-http"]
