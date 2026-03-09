-- agentmemory.md — PostgreSQL + Apache AGE schema
-- Runs automatically on first container start via docker-entrypoint-initdb.d
-- Note: AGE extension is already created by the image's 00-create-extension-age.sql

-- -------------------------------------------------------------------------
-- Load AGE for this session and create the knowledge graph
-- -------------------------------------------------------------------------
LOAD 'age';
SET search_path = ag_catalog, "$user", public;

DO $$
BEGIN
    -- Create graph only if it doesn't already exist
    IF NOT EXISTS (
        SELECT 1 FROM ag_catalog.ag_graph WHERE name = 'memory_graph'
    ) THEN
        PERFORM ag_catalog.create_graph('memory_graph');
    END IF;
END
$$;

-- -------------------------------------------------------------------------
-- Relational tables (entity metadata + fast lookups)
-- Reset to public schema for these
-- -------------------------------------------------------------------------
SET search_path = public;

CREATE TABLE IF NOT EXISTS entities (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    node_type   TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata    JSONB NOT NULL DEFAULT '{}',
    tags        TEXT[] NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_entities_node_type ON entities (node_type);
CREATE INDEX IF NOT EXISTS idx_entities_name_fts ON entities USING gin (to_tsvector('english', name));
CREATE INDEX IF NOT EXISTS idx_entities_tags ON entities USING gin (tags);
CREATE INDEX IF NOT EXISTS idx_entities_created_at ON entities (created_at DESC);

CREATE TABLE IF NOT EXISTS relations (
    id          TEXT PRIMARY KEY,
    from_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    to_id       TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    edge_type   TEXT NOT NULL,
    properties  JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_relations_from_id ON relations (from_id);
CREATE INDEX IF NOT EXISTS idx_relations_to_id ON relations (to_id);
CREATE INDEX IF NOT EXISTS idx_relations_edge_type ON relations (edge_type);
CREATE INDEX IF NOT EXISTS idx_relations_from_edge ON relations (from_id, edge_type);

CREATE TABLE IF NOT EXISTS metric_data_points (
    id          TEXT PRIMARY KEY,
    metric_id   TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    value       DOUBLE PRECISION NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL,
    source      TEXT NOT NULL DEFAULT 'manual',
    notes       TEXT
);

CREATE INDEX IF NOT EXISTS idx_metric_dp_metric_id ON metric_data_points (metric_id);
CREATE INDEX IF NOT EXISTS idx_metric_dp_recorded_at ON metric_data_points (metric_id, recorded_at DESC);

-- -------------------------------------------------------------------------
-- Trigger: auto-update updated_at
-- -------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_entities_updated_at ON entities;
CREATE TRIGGER trg_entities_updated_at
    BEFORE UPDATE ON entities
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
