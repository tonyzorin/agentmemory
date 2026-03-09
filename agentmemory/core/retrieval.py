"""
Hybrid retrieval — combines Redis FT.HYBRID (BM25 + vector) with
Apache AGE graph traversal and confidence scoring.

Score formula (similarity-adaptive):
    When similarity > HIGH_SIM_THRESHOLD (0.6):
        base_score = similarity * 0.80 + graph_boost * 0.15 + recency * 0.05
    Otherwise:
        base_score = similarity * 0.50 + graph_boost * 0.20 + recency * 0.20 + importance_component * 0.10
    final_score = base_score * importance_weight

Where:
    similarity        — min-max normalized RRF score (0–1)
    graph_boost       — graduated: 0.3 for 1-hop neighbors, 0.1 for 2-hop, 0 otherwise
    recency           — exponential decay: exp(-DECAY_RATE * age_hours)
    importance_weight — 1.0 + (effective_importance - 0.5) * IMPORTANCE_SCALE
    effective_importance — base_importance × access_boost × age_penalty (dynamic)
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from math import exp
from typing import Any

from agentmemory.core.memory import MemoryService

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query expansion — synonym map for BM25 side of hybrid search
# ---------------------------------------------------------------------------

# Maps a term to a list of synonyms/related terms to inject into the BM25 query.
# Keys are lowercase. Values are appended to the query text (not the vector query).
_SYNONYM_MAP: dict[str, list[str]] = {
    # Pricing / commercial
    "pricing": ["price", "cost", "fee", "rate", "tariff", "charge", "EUR", "USD"],
    "price": ["pricing", "cost", "fee", "rate", "tariff"],
    "cost": ["price", "pricing", "fee", "rate", "charge"],
    "revenue": ["income", "sales", "MRR", "ARR", "turnover"],
    "mrr": ["monthly recurring revenue", "revenue", "subscription"],
    "arr": ["annual recurring revenue", "revenue"],
    "subscription": ["plan", "tier", "pricing", "MRR"],
    "plan": ["tier", "pricing", "subscription"],
    # Business / strategy
    "grant": ["funding", "subsidy", "award", "EMEL", "EU", "finance"],
    "funding": ["grant", "investment", "capital", "finance"],
    "strategy": ["plan", "roadmap", "approach", "direction"],
    "roadmap": ["plan", "strategy", "timeline", "milestones"],
    "gtm": ["go to market", "launch", "sales strategy", "marketing"],
    "go-to-market": ["GTM", "launch", "sales", "marketing"],
    # Technical
    "deploy": ["deployment", "release", "ship", "publish", "production"],
    "deployment": ["deploy", "release", "production", "ship"],
    "api": ["endpoint", "REST", "HTTP", "interface", "integration"],
    "bug": ["error", "issue", "defect", "problem", "fix"],
    "performance": ["speed", "latency", "slow", "fast", "optimization"],
    # People / roles
    "ceo": ["founder", "director", "executive", "leadership"],
    "cto": ["technical", "engineering", "technology", "lead"],
    "pm": ["product manager", "product management"],
    # Transport / busonmap-specific
    "bus": ["transit", "transport", "operator", "route", "vehicle"],
    "operator": ["bus company", "transit authority", "fleet", "carrier"],
    "route": ["line", "trip", "journey", "path"],
    "passenger": ["rider", "commuter", "traveler", "user"],
    "gtfs": ["transit feed", "schedule", "timetable", "GTFS-RT"],
    # Misc abbreviations
    "msp360": ["MSP360", "CloudBerry", "backup software"],
    "emel": ["EMEL", "grant", "Lisbon", "municipality"],
    "busonmap": ["bus on map", "busonmap.com", "transit", "bus tracking"],
}


def _expand_query(query: str) -> str:
    """
    Expand a query with synonyms for the BM25 side of hybrid search.

    Splits camelCase/compound words, looks up each token in the synonym map,
    and appends unique expansion terms to the query string.

    The original query is always preserved. Expansion terms are appended
    so they only affect BM25 keyword matching, not vector similarity.

    Returns the expanded query string (original + synonyms).
    """
    # Split camelCase and PascalCase into separate tokens
    # e.g. "busonmap" stays as-is (no capitals), "GTMStrategy" -> "GTM Strategy"
    split_camel = re.sub(r"([a-z])([A-Z])", r"\1 \2", query)

    tokens = re.findall(r"[a-zA-Z0-9]+", split_camel.lower())

    expansion_terms: list[str] = []
    seen: set[str] = set(tokens)

    for token in tokens:
        synonyms = _SYNONYM_MAP.get(token, [])
        for syn in synonyms:
            syn_lower = syn.lower()
            if syn_lower not in seen:
                expansion_terms.append(syn)
                seen.add(syn_lower)

    if not expansion_terms:
        return query

    expanded = query + " " + " ".join(expansion_terms)
    logger.debug("Query expanded: %r -> %r", query, expanded)
    return expanded

# Graph boost values — graduated by hop distance
GRAPH_BOOST_1HOP = 0.3   # Direct neighbor of anchor
GRAPH_BOOST_2HOP = 0.1   # 2 hops away from anchor

# Similarity threshold above which similarity dominates the score
HIGH_SIM_THRESHOLD = 0.6

# Importance multiplier scale: 0 = no effect, 0.3 = ±15% at extremes
IMPORTANCE_SCALE = 0.3

# Recency decay: half-life ≈ 29 days (0.001 * 693 hours ≈ 29 days)
DECAY_RATE = 0.001

# Dynamic importance: access_count multiplier (capped at 1.5)
ACCESS_BOOST_PER_HIT = 0.05
ACCESS_BOOST_CAP = 1.5

# Age penalty thresholds per node type (days → penalty factor)
# Nodes older than the threshold get their importance multiplied by the factor.
_AGE_DECAY_RULES: dict[str, tuple[int, float]] = {
    # (threshold_days, penalty_factor)
    "Task": (30, 0.3),
    "Initiative": (90, 0.5),
    "Memory": (180, 0.7),       # General memories slowly fade
    "CustomerFeedback": (60, 0.5),
}


def _effective_importance(
    base_importance: float,
    node_type: str,
    access_count: int,
    created_at_str: str | None,
) -> float:
    """
    Compute dynamic effective importance.

    effective = base × access_boost × age_penalty

    access_boost: increases with how often a node has been recalled
                  (capped at 1.5 so frequently-accessed nodes don't dominate)
    age_penalty:  type-specific decay for ephemeral node types past their prime
    """
    # Access boost: more recalls = more relevant
    access_boost = min(ACCESS_BOOST_CAP, 1.0 + access_count * ACCESS_BOOST_PER_HIT)

    # Age penalty: only for types with defined decay rules
    age_penalty = 1.0
    if node_type in _AGE_DECAY_RULES and created_at_str:
        threshold_days, penalty = _AGE_DECAY_RULES[node_type]
        try:
            dt_str = str(created_at_str).replace("Z", "+00:00")
            try:
                created = datetime.fromisoformat(dt_str)
            except ValueError:
                created = datetime.fromisoformat(dt_str[:19])
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - created).total_seconds() / 86400
            if age_days > threshold_days:
                age_penalty = penalty
        except Exception:
            pass

    effective = base_importance * access_boost * age_penalty
    return min(1.0, max(0.0, effective))


def _recency_score(created_at_str: str | None, ttl_days: int | None = None) -> float:
    """
    Compute exponential recency score in [0, 1].

    Returns 1.0 for brand-new memories, decaying toward 0 over time.
    Half-life is ~29 days at the default DECAY_RATE.

    When ttl_days is set, applies accelerated decay so nodes past their TTL
    drop to near-zero recency (effectively archived without being deleted).
    At 2× TTL age, the score is ~0.05.
    """
    if not created_at_str:
        return 0.5  # Unknown age — neutral

    try:
        # Handle both naive and timezone-aware ISO strings
        dt_str = str(created_at_str).replace("Z", "+00:00")
        try:
            created = datetime.fromisoformat(dt_str)
        except ValueError:
            # Fallback: strip microseconds and try again
            created = datetime.fromisoformat(dt_str[:19])

        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        age_hours = max(0.0, (now - created).total_seconds() / 3600)

        if ttl_days is not None:
            # Accelerated decay: half-life = ttl_days / 2
            # At age == ttl_days: score ≈ 0.25; at 2× ttl_days: score ≈ 0.06
            ttl_hours = ttl_days * 24
            ttl_decay_rate = 0.693 / (ttl_hours / 2)  # ln(2) / half_life
            return exp(-ttl_decay_rate * age_hours)

        return exp(-DECAY_RATE * age_hours)
    except Exception:
        return 0.5


def _normalize_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Min-max normalize the 'similarity' field across a result set to [0, 1].

    RRF scores from FT.HYBRID are rank-based (not cosine similarity) and
    naturally land in a narrow range. This maps them to the full [0, 1] range
    so they combine meaningfully with other score components.

    Pure cosine similarity scores (already in [0, 1]) are left unchanged.
    """
    if not results:
        return results

    scores = [float(r.get("similarity", r.get("score", 0.0))) for r in results]
    min_s = min(scores)
    max_s = max(scores)

    if max_s <= 0:
        return results

    score_range = max_s - min_s

    normalized = []
    for r, raw in zip(results, scores):
        if raw > 1.0:
            # RRF score — apply min-max normalization
            if score_range > 0:
                norm = (raw - min_s) / score_range
            else:
                norm = 1.0
        else:
            # Already a proper cosine similarity — keep as-is
            norm = raw
        normalized.append({**r, "similarity": round(norm, 4)})

    return normalized


def _detect_anchor(query: str, memory_service: Any) -> str | None:
    """
    Auto-detect the most likely anchor entity from the query text.

    Extracts candidate terms (words 4+ chars, camelCase tokens, known project names)
    and looks each one up in the entities table. Returns the ID of the first match,
    or None if nothing is found.

    This runs only when anchor_entity_id is not explicitly provided.
    """
    # Extract candidate terms: words of 4+ chars, plus any CamelCase tokens
    tokens = re.findall(r"[A-Z][a-z]+[A-Za-z]*|[a-zA-Z]{4,}", query)
    # Deduplicate while preserving order
    seen: set[str] = set()
    candidates: list[str] = []
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            candidates.append(t)

    for candidate in candidates:
        try:
            entity = memory_service.find_entity_by_name(candidate)
            if entity:
                logger.debug("Auto-detected anchor: %r -> %s (%s)", candidate, entity["id"], entity.get("name"))
                return entity["id"]
        except Exception as e:
            logger.debug("_detect_anchor lookup failed for %r: %s", candidate, e)

    return None


class HybridRetrieval:
    """
    Orchestrates hybrid retrieval across Redis and AGE.

    Usage:
        retrieval = HybridRetrieval(...)
        results = retrieval.retrieve("what does Anton prefer for coding")
    """

    def __init__(
        self,
        database_url: str = "postgresql://openclaw:openclaw@localhost:5433/openclaw_memory",
        redis_url: str = "redis://localhost:6380/0",
        key_prefix: str = "",
        graph_name: str = "memory_graph",
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        embedding_dim: int = 768,
    ):
        self.memory = MemoryService(
            database_url=database_url,
            redis_url=redis_url,
            key_prefix=key_prefix,
            graph_name=graph_name,
            embedding_model=embedding_model,
            embedding_dim=embedding_dim,
        )

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        tags: list[str] | None = None,
        node_type: str | None = None,
        anchor_entity_id: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Hybrid retrieval with graph boost, recency decay, and importance weighting.

        Args:
            query: Natural language query.
            limit: Maximum number of results.
            tags: Optional tag filter.
            node_type: Optional node type filter.
            anchor_entity_id: If provided, boost results connected to this entity.
                              If None, the top result's ID is used as a fallback anchor.
            min_score: Minimum score threshold (0.0–1.0).

        Returns:
            List of result dicts with id, content, score, similarity,
            graph_boost, recency, and importance fields.
        """
        if not query.strip():
            return []

        # Step 1: Get query embedding (with BGE query prefix if applicable)
        try:
            query_embedding = self.memory.embeddings.encode(query, is_query=True)
        except Exception as e:
            logger.error("Failed to encode query: %s", e)
            return []

        # Step 2: Redis hybrid search (BM25 + vector)
        # Expand the BM25 text query with synonyms; vector query stays clean.
        expanded_query_text = _expand_query(query)
        try:
            raw_results = self.memory.redis.hybrid_search(
                query_text=expanded_query_text,
                query_embedding=query_embedding,
                limit=limit * 3,  # Over-fetch for re-ranking
                tags=tags,
            )
        except Exception as e:
            logger.error("Redis hybrid search failed: %s", e)
            raw_results = []

        if not raw_results:
            # Fall back to vector-only
            try:
                raw_results = self.memory.redis.vector_search(
                    query_embedding=query_embedding,
                    limit=limit * 2,
                    tags=tags,
                )
            except Exception as e:
                logger.error("Vector search fallback failed: %s", e)
                return []

        if not raw_results:
            return []

        # Step 3: Normalize scores across the result set
        raw_results = _normalize_scores(raw_results)

        # Step 4: Determine anchor for graph boost
        # Priority: explicit anchor > auto-detected from query > top result fallback
        effective_anchor = anchor_entity_id
        if not effective_anchor:
            effective_anchor = _detect_anchor(query, self.memory) or ""
        if not effective_anchor and raw_results:
            top = raw_results[0]
            effective_anchor = top.get("id") or top.get("$.id") or ""
            if not effective_anchor:
                key = top.get("_key", "")
                if ":" in key:
                    effective_anchor = key.split(":")[-1]

        # Step 5: Get graph neighborhood for anchor entity (graduated by hop distance)
        # Returns {node_id: hop_distance} — 1-hop gets GRAPH_BOOST_1HOP, 2-hop gets GRAPH_BOOST_2HOP
        graph_depth_map: dict[str, int] = {}
        if effective_anchor:
            try:
                graph_depth_map = self.memory.graph.get_neighborhood_with_depth(
                    effective_anchor, max_depth=2
                )
                logger.debug(
                    "Graph neighborhood for anchor %s: %d nodes",
                    effective_anchor, len(graph_depth_map)
                )
            except Exception as e:
                logger.warning("Graph traversal failed for anchor %s: %s", effective_anchor, e)

        # Step 6: Score and re-rank results
        scored = []
        for result in raw_results:
            result_id = result.get("id") or result.get("$.id", "")
            if not result_id:
                key = result.get("_key", "")
                if ":" in key:
                    result_id = key.split(":")[-1]

            # Normalized similarity from step 3
            similarity = float(result.get("similarity", 0.0))

            # Graduated graph boost: only actual neighbors get boosted
            hop = graph_depth_map.get(result_id)
            if hop == 1:
                graph_boost = GRAPH_BOOST_1HOP
            elif hop == 2:
                graph_boost = GRAPH_BOOST_2HOP
            else:
                graph_boost = 0.0

            # Recency decay (accelerated for nodes with a TTL)
            ttl_days_val = result.get("ttl_days")
            if ttl_days_val is not None:
                try:
                    ttl_days_val = int(ttl_days_val)
                except (ValueError, TypeError):
                    ttl_days_val = None
            recency = _recency_score(result.get("created_at"), ttl_days=ttl_days_val)

            # Dynamic importance: base × access_boost × age_penalty
            base_importance = float(result.get("importance", 0.5))
            access_count = int(result.get("access_count", 0) or 0)
            node_type_val = result.get("node_type", "Memory")
            eff_importance = _effective_importance(
                base_importance, node_type_val, access_count, result.get("created_at")
            )
            importance_weight = 1.0 + (eff_importance - 0.5) * IMPORTANCE_SCALE

            # Similarity-adaptive scoring formula:
            # High similarity → similarity dominates (prevents recency from hijacking)
            # Medium similarity → balanced formula with recency + importance as tiebreakers
            if similarity > HIGH_SIM_THRESHOLD:
                final_score = (
                    similarity * 0.80
                    + graph_boost * 0.15
                    + recency * 0.05
                ) * importance_weight
            else:
                importance_component = eff_importance  # direct 0–1 value
                final_score = (
                    similarity * 0.50
                    + graph_boost * 0.20
                    + recency * 0.20
                    + importance_component * 0.10
                ) * importance_weight

            final_score = min(1.0, max(0.0, final_score))

            # Filter by node_type
            if node_type and result.get("node_type") != node_type:
                continue

            if final_score < min_score:
                continue

            scored.append({
                **{k: v for k, v in result.items() if k != "embedding"},
                "id": result_id,
                "score": round(final_score, 4),
                "similarity": round(similarity, 4),
                "graph_boost": round(graph_boost, 4),
                "recency": round(recency, 4),
                "importance": round(base_importance, 4),
                "effective_importance": round(eff_importance, 4),
                "access_count": access_count,
            })

        # Sort by final score
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    def close(self) -> None:
        """Release resources."""
        self.memory.close()
