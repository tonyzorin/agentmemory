"""
Memory consolidation — merges near-duplicate nodes into canonical summaries.

This is a maintenance tool to be run periodically (weekly or monthly) as the
corpus grows. It does NOT run automatically. Use the CLI:

    memory consolidate --dry-run          # preview what would be merged
    memory consolidate --threshold 0.85   # merge near-duplicates
    memory consolidate --type Memory      # only consolidate Memory nodes

How it works
------------
1. Load all searchable nodes' embeddings from Redis (Memory, Learning,
   Decision, Preference, Workflow, CustomerFeedback).
2. Group nodes into clusters using greedy cosine similarity:
   - Sort nodes by importance descending.
   - For each node, if it's similar (>= threshold) to the centroid of an
     existing cluster, add it to that cluster. Otherwise start a new cluster.
3. For each cluster with 2+ nodes:
   - Keep the node with highest importance as the canonical node.
   - Append unique content from other nodes into the canonical node's content.
   - Union all tags.
   - Re-attach outgoing graph edges from removed nodes to the canonical node.
   - Delete the non-canonical nodes.

The greedy algorithm avoids scipy/scikit-learn dependencies. It's O(N²) but
fine for corpora up to ~10K nodes (consolidation runs infrequently).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Node types that have embeddings and can be consolidated
CONSOLIDATABLE_TYPES = {
    "Memory", "Learning", "Decision", "Preference", "Workflow", "CustomerFeedback"
}


@dataclass
class Cluster:
    """A group of near-duplicate nodes."""
    canonical: dict[str, Any]          # Node that will be kept (highest importance)
    duplicates: list[dict[str, Any]]    # Nodes that will be merged into canonical
    centroid: list[float]               # Average embedding of the cluster

    @property
    def all_nodes(self) -> list[dict[str, Any]]:
        return [self.canonical] + self.duplicates

    @property
    def all_ids(self) -> list[str]:
        return [n["id"] for n in self.all_nodes]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two unit-normalized vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    # BGE embeddings are already unit-normalized, so ||a|| = ||b|| = 1
    return max(-1.0, min(1.0, dot))


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    """Element-wise mean of a list of vectors."""
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    result = [0.0] * dim
    for v in vectors:
        for i, x in enumerate(v):
            result[i] += x
    return [x / n for x in result]


class ConsolidationService:
    """
    Finds and merges near-duplicate memory nodes.

    Usage:
        svc = ConsolidationService(memory_service)
        clusters = svc.find_clusters(similarity_threshold=0.85)
        for cluster in clusters:
            svc.merge_cluster(cluster, dry_run=False)
    """

    def __init__(self, memory_service: Any):
        self.memory = memory_service

    def find_clusters(
        self,
        similarity_threshold: float = 0.85,
        node_type: str | None = None,
    ) -> list[Cluster]:
        """
        Load all searchable nodes and group near-duplicates into clusters.

        Args:
            similarity_threshold: Cosine similarity threshold (0–1).
                                  0.85 is conservative (catches clear duplicates).
                                  0.75 is aggressive (also merges related content).
            node_type: Limit to one node type (e.g. "Memory"). Default: all searchable types.

        Returns:
            List of Cluster objects. Only clusters with 2+ nodes are returned
            (singletons are not duplicates and need no merging).
        """
        types_to_check = {node_type} if node_type else CONSOLIDATABLE_TYPES
        nodes = self._load_nodes_with_embeddings(types_to_check)

        if not nodes:
            logger.info("No searchable nodes found for consolidation")
            return []

        logger.info("Loaded %d nodes for consolidation analysis", len(nodes))

        # Sort by importance descending — higher importance nodes become cluster centers
        nodes.sort(key=lambda n: float(n.get("importance", 0.5)), reverse=True)

        clusters: list[Cluster] = []

        for node in nodes:
            embedding = node.get("_embedding")
            if not embedding:
                continue

            best_cluster: Cluster | None = None
            best_sim = similarity_threshold - 0.001  # must exceed threshold

            for cluster in clusters:
                sim = _cosine_similarity(embedding, cluster.centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_cluster = cluster

            if best_cluster is not None:
                best_cluster.duplicates.append(node)
                # Update centroid to include new member
                all_embeddings = [
                    n["_embedding"] for n in best_cluster.all_nodes
                    if n.get("_embedding")
                ]
                best_cluster.centroid = _mean_vector(all_embeddings)
            else:
                # Start a new cluster with this node as canonical
                clusters.append(Cluster(
                    canonical=node,
                    duplicates=[],
                    centroid=list(embedding),
                ))

        # Return only clusters with actual duplicates
        return [c for c in clusters if c.duplicates]

    def merge_cluster(self, cluster: Cluster, dry_run: bool = True) -> dict[str, Any]:
        """
        Merge a cluster of near-duplicate nodes into the canonical node.

        The canonical node (highest importance) is kept. Its content is
        extended with unique sentences from duplicate nodes. Tags are unioned.
        All outgoing graph edges from duplicate nodes are re-attached to the
        canonical node. Duplicate nodes are then deleted.

        Args:
            cluster: A Cluster returned by find_clusters().
            dry_run: If True, preview the merge without writing anything.

        Returns:
            Summary dict with canonical_id, merged_ids, and content_preview.
        """
        canonical = cluster.canonical
        canonical_id = canonical["id"]
        dup_ids = [n["id"] for n in cluster.duplicates]

        # Build merged content: canonical content + unique sentences from duplicates
        canonical_content = canonical.get("content", "")
        canonical_sentences = set(
            s.strip() for s in canonical_content.split(".")
            if len(s.strip()) > 20
        )
        extra_sentences: list[str] = []

        for dup in cluster.duplicates:
            dup_content = dup.get("content", "")
            for sentence in dup_content.split("."):
                sentence = sentence.strip()
                if len(sentence) > 20 and sentence not in canonical_sentences:
                    extra_sentences.append(sentence)
                    canonical_sentences.add(sentence)

        merged_content = canonical_content
        if extra_sentences:
            merged_content = canonical_content.rstrip(".") + ". " + ". ".join(extra_sentences)

        # Union all tags
        all_tags: list[str] = list(canonical.get("tags") or [])
        for dup in cluster.duplicates:
            for tag in (dup.get("tags") or []):
                if tag not in all_tags:
                    all_tags.append(tag)

        summary = {
            "canonical_id": canonical_id,
            "merged_ids": dup_ids,
            "canonical_content_before": canonical_content[:120],
            "merged_content_preview": merged_content[:200],
            "tags_after": all_tags,
            "dry_run": dry_run,
        }

        if dry_run:
            return summary

        # --- Perform the actual merge ---

        # 1. Re-attach outgoing edges from duplicates to canonical
        for dup in cluster.duplicates:
            dup_id = dup["id"]
            try:
                relations = self.memory.postgres.get_relations(from_id=dup_id)
                for rel in relations:
                    to_id = rel.get("to_id", "")
                    edge_type = rel.get("edge_type", "")
                    if to_id and edge_type and to_id != canonical_id:
                        try:
                            self.memory.relate(canonical_id, to_id, edge_type)
                        except Exception as e:
                            logger.debug("Could not re-attach edge %s->%s: %s", canonical_id, to_id, e)
            except Exception as e:
                logger.warning("Failed to get relations for %s: %s", dup_id, e)

        # 2. Update canonical node with merged content and tags
        try:
            self.memory.update(
                memory_id=canonical_id,
                content=merged_content,
                tags=all_tags,
            )
        except Exception as e:
            logger.error("Failed to update canonical node %s: %s", canonical_id, e)
            summary["error"] = str(e)
            return summary

        # 3. Delete duplicate nodes
        deleted: list[str] = []
        for dup_id in dup_ids:
            try:
                self.memory.forget(dup_id)
                deleted.append(dup_id)
            except Exception as e:
                logger.warning("Failed to delete duplicate %s: %s", dup_id, e)

        summary["deleted"] = deleted
        logger.info(
            "Merged %d nodes into %s (deleted: %s)",
            len(deleted), canonical_id, deleted,
        )
        return summary

    def _load_nodes_with_embeddings(
        self, node_types: set[str]
    ) -> list[dict[str, Any]]:
        """
        Load all searchable nodes from PostgreSQL and their embeddings from Redis.

        Returns list of node dicts with an extra `_embedding` key.
        Nodes without a Redis entry (no embedding) are skipped.
        """
        nodes = []
        for node_type in node_types:
            try:
                entities = self.memory.postgres.list_entities_by_type(node_type)
                for entity in entities:
                    node_id = entity.get("id")
                    if not node_id:
                        continue

                    # Fetch embedding from Redis
                    try:
                        key = f"{self.memory.redis.prefix}memory:{node_id}"
                        doc = self.memory.redis.redis.json().get(key, "$.embedding")
                        if doc and isinstance(doc, list) and doc[0]:
                            embedding = doc[0]
                            if isinstance(embedding, list) and len(embedding) > 0:
                                # Get full content from Redis or metadata
                                content_doc = self.memory.redis.redis.json().get(
                                    key, "$.content", "$.node_type", "$.importance", "$.tags"
                                )
                                meta = entity.get("metadata") or {}
                                if isinstance(meta, str):
                                    import json
                                    try:
                                        meta = json.loads(meta)
                                    except Exception:
                                        meta = {}

                                nodes.append({
                                    "id": node_id,
                                    "name": entity.get("name", ""),
                                    "node_type": entity.get("node_type", node_type),
                                    "content": meta.get("content", entity.get("name", "")),
                                    "importance": meta.get("importance", 0.5),
                                    "tags": entity.get("tags") or [],
                                    "_embedding": embedding,
                                })
                    except Exception as e:
                        logger.debug("Could not load embedding for %s: %s", node_id, e)
            except Exception as e:
                logger.warning("Could not load %s nodes: %s", node_type, e)

        return nodes
