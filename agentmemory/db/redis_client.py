"""
Redis search client for agentmemory.md.

Handles:
- Storing memory documents as Redis JSON
- Vector similarity search (FT.SEARCH with KNN)
- Hybrid search (FT.HYBRID with BM25 + vector, RRF fusion)
- Embedding cache with TTL
- Tag-based filtering

Adapted from feedback1/apps/ai/retrieval/redis_search_client.py,
simplified for single-user memory (no tenant_id) with memory-specific schema.
"""

import hashlib
import logging
import struct
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import redis
import redis.exceptions

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 768


def _floats_to_bytes(floats: list[float]) -> bytes:
    """Pack float list to bytes for Redis vector storage (FLOAT32)."""
    return struct.pack(f"{len(floats)}f", *floats)


def _bytes_to_floats(data: bytes, dim: int) -> list[float]:
    """Unpack bytes back to float list."""
    return list(struct.unpack(f"{dim}f", data))


class MemoryRedisClient:
    """
    Redis client for the agentmemory.md memory system.

    All keys are prefixed with `key_prefix` (default: "memory:") to allow
    test isolation and future multi-workspace support.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6380/0",
        key_prefix: str = "",
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        embedding_cache_ttl: int = 86400,
    ):
        self.redis_url = redis_url
        self.prefix = key_prefix
        self.embedding_dim = embedding_dim
        self.embedding_cache_ttl = embedding_cache_ttl
        self.index_name = f"{self.prefix}memory_idx"
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.redis = redis.from_url(redis_url, decode_responses=False)
        self._ensure_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """
        Create the memory search index if it doesn't exist.

        Drops and recreates the index when:
        - The embedding dimension changed (e.g. 384 -> 768 after model upgrade)
        - The schema is missing required fields (e.g. node_type was added later)
        """
        try:
            info_raw = self.redis.execute_command("FT.INFO", self.index_name)
            # Parse the flat list into a dict
            info: dict = {}
            for i in range(0, len(info_raw) - 1, 2):
                k = info_raw[i]
                if isinstance(k, bytes):
                    k = k.decode()
                info[k] = info_raw[i + 1]

            needs_rebuild = False

            # Check vector dimension matches
            stored_dim = self._extract_index_dim(info)
            if stored_dim is not None and stored_dim != self.embedding_dim:
                logger.warning(
                    "Index %s has dim=%d but embedding_dim=%d — dropping and recreating",
                    self.index_name, stored_dim, self.embedding_dim,
                )
                needs_rebuild = True

            # Check that node_type field is indexed (added after initial schema)
            if not needs_rebuild and not self._index_has_field(info, "node_type"):
                logger.warning(
                    "Index %s is missing node_type field — dropping and recreating",
                    self.index_name,
                )
                needs_rebuild = True

            if needs_rebuild:
                try:
                    self.redis.execute_command("FT.DROPINDEX", self.index_name)
                except Exception:
                    pass
                self._create_index()
            else:
                logger.debug("Memory index %s already exists (dim=%s)", self.index_name, stored_dim)
        except redis.ResponseError:
            self._create_index()

    def _extract_index_dim(self, info: dict) -> int | None:
        """Extract the vector dimension from FT.INFO output."""
        try:
            attrs = info.get("attributes", [])
            if not attrs:
                return None
            # attributes is a list of attribute descriptors (each is a list)
            for attr in attrs:
                if not isinstance(attr, (list, tuple)):
                    continue
                attr_dict: dict = {}
                for i in range(0, len(attr) - 1, 2):
                    k = attr[i]
                    v = attr[i + 1]
                    if isinstance(k, bytes):
                        k = k.decode()
                    if isinstance(v, bytes):
                        v = v.decode()
                    attr_dict[k] = v
                if attr_dict.get("type", "").upper() == "VECTOR":
                    dim = attr_dict.get("dim") or attr_dict.get("DIM")
                    if dim is not None:
                        return int(dim)
        except Exception:
            pass
        return None

    def _index_has_field(self, info: dict, field_name: str) -> bool:
        """Check whether a named field is present in the FT index schema."""
        try:
            attrs = info.get("attributes", [])
            for attr in attrs:
                if not isinstance(attr, (list, tuple)):
                    continue
                for item in attr:
                    if isinstance(item, bytes) and item.decode() == field_name:
                        return True
                    if isinstance(item, str) and item == field_name:
                        return True
        except Exception:
            pass
        return False

    def _create_index(self) -> None:
        """Create FT index for memory documents with vector + text fields."""
        prefix = f"{self.prefix}memory:"
        self.redis.execute_command(
            "FT.CREATE", self.index_name,
            "ON", "JSON",
            "PREFIX", "1", prefix,
            "SCHEMA",
            "$.id",           "AS", "id",           "TAG",
            "$.content",      "AS", "content",      "TEXT",
            "$.node_type",    "AS", "node_type",    "TAG",
            "$.source",       "AS", "source",        "TAG",
            "$.tags.*",       "AS", "tags",          "TAG",
            "$.importance",   "AS", "importance",    "NUMERIC", "SORTABLE",
            "$.ttl_days",     "AS", "ttl_days",      "NUMERIC",
            "$.access_count", "AS", "access_count",  "NUMERIC",
            "$.created_at",   "AS", "created_at",    "TEXT",    "SORTABLE",
            "$.embedding",  "AS", "embedding",   "VECTOR",  "FLAT", "6",
            "TYPE", "FLOAT32",
            "DIM", str(self.embedding_dim),
            "DISTANCE_METRIC", "COSINE",
        )
        logger.info("Created memory index %s", self.index_name)

    # ------------------------------------------------------------------
    # Store / delete
    # ------------------------------------------------------------------

    def store_memory(self, doc: dict[str, Any]) -> bool:
        """
        Store a memory document in Redis JSON.

        doc must have: id, content, embedding (list[float]), and optionally
        tags, source, importance, created_at.
        """
        memory_id = doc["id"]
        key = f"{self.prefix}memory:{memory_id}"
        try:
            self.redis.json().set(key, "$", doc)
            return True
        except Exception as e:
            logger.error("Failed to store memory %s: %s", memory_id, e)
            return False

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory document by ID. Returns False if not found."""
        key = f"{self.prefix}memory:{memory_id}"
        deleted = self.redis.delete(key)
        return deleted > 0

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        """Retrieve a single memory document by ID."""
        key = f"{self.prefix}memory:{memory_id}"
        try:
            return self.redis.json().get(key)
        except Exception as e:
            logger.error("Failed to get memory %s: %s", memory_id, e)
            return None

    # ------------------------------------------------------------------
    # Vector search (KNN)
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        min_score: float = 0.0,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        KNN vector similarity search.

        Returns list of memory dicts with added 'similarity' field (0–1).
        Optionally filters by tags.
        """
        query_vector = _floats_to_bytes(query_embedding)

        # Build filter
        if tags:
            tag_filters = " | ".join(f"@tags:{{{t}}}" for t in tags)
            filter_expr = f"({tag_filters})"
        else:
            filter_expr = "*"

        query = f"({filter_expr})=>[KNN {limit} @embedding $vec AS score]"

        try:
            result = self.redis.execute_command(
                "FT.SEARCH", self.index_name,
                query,
                "PARAMS", "2", "vec", query_vector,
                "SORTBY", "score",
                "RETURN", "11",
                "id", "content", "node_type", "source", "tags", "importance", "ttl_days", "access_count", "created_at", "score", "$.id",
                "LIMIT", "0", str(limit),
                "DIALECT", "2",
            )
            return self._parse_knn_results(result, min_score)
        except redis.ResponseError as e:
            logger.error("Vector search failed: %s", e)
            return []

    def _parse_knn_results(
        self, result: Any, min_score: float = 0.0
    ) -> list[dict[str, Any]]:
        """Parse FT.SEARCH KNN results into a list of dicts."""
        if not result or result[0] == 0:
            return []

        docs = []
        i = 1
        while i < len(result):
            doc_key = result[i]
            if isinstance(doc_key, bytes):
                doc_key = doc_key.decode()

            fields_raw = result[i + 1] if i + 1 < len(result) else []
            doc: dict[str, Any] = {"_key": doc_key}

            j = 0
            while j < len(fields_raw):
                fname = fields_raw[j]
                fval = fields_raw[j + 1] if j + 1 < len(fields_raw) else None
                if isinstance(fname, bytes):
                    fname = fname.decode()
                if isinstance(fval, bytes):
                    fval = fval.decode()

                if fname == "score":
                    try:
                        distance = float(fval)
                        doc["similarity"] = max(0.0, 1.0 - distance)
                    except (ValueError, TypeError):
                        doc["similarity"] = 0.0
                elif fname == "$.id":
                    doc["id"] = fval
                else:
                    doc[fname] = fval
                j += 2

            similarity = doc.get("similarity", 0.0)
            if similarity >= min_score:
                # Ensure id is set
                if "id" not in doc and "$.id" not in doc:
                    # Extract from key
                    doc["id"] = doc_key.split(":")[-1]
                docs.append(doc)
            i += 2

        return docs

    # ------------------------------------------------------------------
    # Hybrid search (FT.HYBRID — BM25 + vector, RRF fusion)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_text: str,
        query_embedding: list[float],
        limit: int = 10,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search combining BM25 keyword + vector similarity via FT.HYBRID.

        Falls back to vector-only search if FT.HYBRID is unavailable.
        """
        query_vector = _floats_to_bytes(query_embedding)

        # Build tag filter
        if tags:
            tag_filters = " | ".join(f"@tags:{{{t}}}" for t in tags)
            filter_expr = f"({tag_filters})"
        else:
            filter_expr = "*"

        try:
            # Redis 8.6 FT.HYBRID syntax.
            # Parameter count convention: the integer after each keyword counts ALL
            # tokens that follow it within that clause.
            #
            # COMBINE RRF 4 WINDOW 20 YIELD_SCORE_AS hybrid_score
            #   └─ count=4: WINDOW, 20, YIELD_SCORE_AS, hybrid_score
            #
            # YIELD_SCORE_AS exposes the fused RRF score as a loadable field.
            vsim_filter = None
            if tags:
                tag_filters = " | ".join(f"@tags:{{{t}}}" for t in tags)
                vsim_filter = f"({tag_filters})"

            cmd = [
                "FT.HYBRID", self.index_name,
                "SEARCH", query_text if query_text.strip() else "*",
                "VSIM", "@embedding", "$vec",
                "KNN", "2", "K", str(limit),
            ]
            if vsim_filter:
                cmd += ["FILTER", vsim_filter]
            cmd += [
                "COMBINE", "RRF", "4", "WINDOW", "20", "YIELD_SCORE_AS", "hybrid_score",
                "LOAD", "*",
                "LIMIT", "0", str(limit),
                "PARAMS", "2", "vec", query_vector,
            ]

            result = self.redis.execute_command(*cmd)
            return self._parse_hybrid_results(result)
        except redis.ResponseError as e:
            logger.warning(
                "FT.HYBRID unavailable (%s), falling back to vector search", e
            )
            return self.vector_search(query_embedding, limit=limit, tags=tags)

    def _parse_hybrid_results(self, result: Any) -> list[dict[str, Any]]:
        """
        Parse FT.HYBRID results.

        Redis 8.6 returns:
          [b'total_results', N, b'results',
           [[b'hybrid_score', '0.032...', b'$', json_str], ...],
           b'warnings', [...]]

        hybrid_score is the raw RRF value (≈ 1/61 ≈ 0.0164 per contributing rank-1 list).
        We normalise it to [0, 1] using the theoretical maximum for `limit` candidates:
          max_rrf = limit * (1 / (60 + 1))   (all lists rank this doc #1)
        """
        if not result:
            return []

        import json as _json

        # Convert flat top-level list to dict
        result_dict: dict = {}
        for i in range(0, len(result) - 1, 2):
            k = result[i]
            if isinstance(k, bytes):
                k = k.decode()
            result_dict[k] = result[i + 1]

        raw_docs = result_dict.get("results", [])
        # RRF max: both SEARCH and VSIM rank the doc #1 → 2 * 1/(60+1)
        RRF_MAX = 2.0 / 61.0

        docs = []
        for doc_fields in raw_docs:
            if not doc_fields:
                continue
            doc_dict: dict = {}
            for i in range(0, len(doc_fields) - 1, 2):
                fname = doc_fields[i]
                fval = doc_fields[i + 1]
                if isinstance(fname, bytes):
                    fname = fname.decode()
                if isinstance(fval, bytes):
                    fval = fval.decode()
                doc_dict[fname] = fval

            json_str = doc_dict.get("$")
            if not json_str:
                continue
            try:
                doc_data = _json.loads(json_str)
                raw_score = float(doc_dict.get("hybrid_score", 0.0))
                score_f = min(raw_score / RRF_MAX, 1.0)
                doc_data["score"] = score_f
                doc_data["similarity"] = score_f
                docs.append(doc_data)
            except Exception:
                pass

        return docs

    # ------------------------------------------------------------------
    # Embedding cache
    # ------------------------------------------------------------------

    def cache_embedding(
        self, text: str, embedding: list[float], ttl: int | None = None
    ) -> None:
        """Cache an embedding for a text string."""
        key = self._cache_key(text)
        packed = _floats_to_bytes(embedding)
        ttl = ttl if ttl is not None else self.embedding_cache_ttl
        self.redis.setex(key, ttl, packed)

    def get_cached_embedding(self, text: str) -> list[float] | None:
        """Retrieve a cached embedding. Returns None on cache miss or dimension mismatch."""
        key = self._cache_key(text)
        data = self.redis.get(key)
        if data is None:
            return None
        expected_bytes = self.embedding_dim * 4  # FLOAT32 = 4 bytes per float
        if len(data) != expected_bytes:
            # Stale cache entry from a different embedding model — treat as miss
            self.redis.delete(key)
            return None
        return _bytes_to_floats(data, self.embedding_dim)

    def _cache_key(self, text: str) -> str:
        digest = hashlib.sha256(text.encode()).hexdigest()
        return f"{self.prefix}emb_cache:{digest}"

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """Return basic stats about the memory index."""
        try:
            info = self.redis.execute_command("FT.INFO", self.index_name)
            info_dict = {}
            for i in range(0, len(info) - 1, 2):
                k = info[i]
                v = info[i + 1]
                if isinstance(k, bytes):
                    k = k.decode()
                info_dict[k] = v

            num_docs = info_dict.get("num_docs", 0)
            if isinstance(num_docs, bytes):
                num_docs = int(num_docs.decode())

            return {
                "index_name": self.index_name,
                "memory_count": int(num_docs),
                "embedding_dim": self.embedding_dim,
            }
        except Exception as e:
            logger.error("Failed to get stats: %s", e)
            return {"index_name": self.index_name, "memory_count": 0, "embedding_dim": self.embedding_dim}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the Redis connection and thread pool."""
        try:
            self.redis.close()
        except Exception:
            pass
        self.executor.shutdown(wait=False)
