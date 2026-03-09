"""
Pydantic models — the contracts for all memory system entities.

These define the shape of data flowing through the system before any
implementation exists. Tests are written against these contracts.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """All vertex types in the knowledge graph."""
    MEMORY = "Memory"
    LEARNING = "Learning"
    DECISION = "Decision"
    GOAL = "Goal"
    INITIATIVE = "Initiative"
    TASK = "Task"
    PROJECT = "Project"
    PERSON = "Person"
    EXTERNAL_CONTACT = "ExternalContact"
    PREFERENCE = "Preference"
    ENVIRONMENT = "Environment"
    TOOL = "Tool"
    WORKFLOW = "Workflow"
    RESOURCE = "Resource"
    COMPETITOR = "Competitor"
    METRIC = "Metric"
    CUSTOMER_FEEDBACK = "CustomerFeedback"
    TOPIC = "Topic"


class EdgeType(str, Enum):
    """All edge types in the knowledge graph."""
    ABOUT = "ABOUT"
    ACHIEVED_VIA = "ACHIEVED_VIA"
    BROKEN_INTO = "BROKEN_INTO"
    PART_OF = "PART_OF"
    BELONGS_TO = "BELONGS_TO"
    FOR = "FOR"
    WORKS_ON = "WORKS_ON"
    OWNS = "OWNS"
    COLLABORATES_ON = "COLLABORATES_ON"
    INVOLVED_IN = "INVOLVED_IN"
    RELATED_TO = "RELATED_TO"
    PREVENTED = "PREVENTED"
    AFFECTS = "AFFECTS"
    LED_TO = "LED_TO"
    USES = "USES"
    USED_BY = "USED_BY"
    HOSTS = "HOSTS"
    DOCUMENTS = "DOCUMENTS"
    MENTIONS = "MENTIONS"
    TAGGED_WITH = "TAGGED_WITH"
    COMPETES_WITH = "COMPETES_WITH"
    TRACKS = "TRACKS"
    FROM = "FROM"
    SUPERSEDES = "SUPERSEDES"


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemorySource(str, Enum):
    CONVERSATION = "conversation"
    CLI = "cli"
    MCP = "mcp"
    IMPORT = "import"


# ---------------------------------------------------------------------------
# Default importance values per node type
# ---------------------------------------------------------------------------

# Default TTL in days per node type. None = permanent (no decay acceleration).
# Ephemeral nodes (Task, event-like Memory) decay faster and can be GC'd.
DEFAULT_TTL_DAYS: dict[str, int | None] = {
    "Task": 90,
    "Initiative": 180,
    "Memory": None,        # General memories are permanent by default
    "Learning": None,
    "Decision": None,
    "Goal": None,
    "Workflow": None,
    "Preference": None,
    "Project": None,
    "Person": None,
    "ExternalContact": None,
    "Environment": None,
    "Tool": None,
    "Resource": None,
    "Topic": None,
    "Competitor": None,
    "Metric": None,
    "CustomerFeedback": None,
}


DEFAULT_IMPORTANCE: dict[str, float] = {
    "Goal": 0.8,
    "Decision": 0.7,
    "Learning": 0.65,
    "Workflow": 0.65,
    "CustomerFeedback": 0.6,
    "Memory": 0.5,
    "Preference": 0.55,
    "Project": 0.6,
    "Initiative": 0.55,
    "Task": 0.45,
    "Competitor": 0.55,
    "Metric": 0.5,
    "Person": 0.5,
    "ExternalContact": 0.45,
    "Environment": 0.4,
    "Tool": 0.4,
    "Resource": 0.4,
    "Topic": 0.4,
}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseNode(BaseModel):
    """Common fields for all graph nodes."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    node_type: NodeType
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)


class Relation(BaseModel):
    """A directed edge between two nodes."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    from_id: str
    to_id: str
    edge_type: EdgeType
    properties: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Knowledge nodes (have embeddings for semantic search)
# ---------------------------------------------------------------------------


class Memory(BaseNode):
    """General observation, fact, or conversation note."""
    node_type: NodeType = NodeType.MEMORY
    content: str
    source: MemorySource = MemorySource.CONVERSATION
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    embedding: list[float] | None = None


class Learning(BaseNode):
    """A failed experiment or lesson learned — what NOT to do."""
    node_type: NodeType = NodeType.LEARNING
    content: str
    what_failed: str
    why_it_failed: str
    what_to_avoid: str
    source: MemorySource = MemorySource.CONVERSATION
    embedding: list[float] | None = None


class Decision(BaseNode):
    """A key choice made with rationale."""
    node_type: NodeType = NodeType.DECISION
    content: str
    rationale: str
    alternatives_considered: list[str] = Field(default_factory=list)
    source: MemorySource = MemorySource.CONVERSATION
    embedding: list[float] | None = None


class Preference(BaseNode):
    """A project-scoped preference discovered over time."""
    node_type: NodeType = NodeType.PREFERENCE
    content: str
    project_id: str | None = None
    embedding: list[float] | None = None


class Workflow(BaseNode):
    """A reusable process or how-to guide."""
    node_type: NodeType = NodeType.WORKFLOW
    content: str
    steps: list[str] = Field(default_factory=list)
    project_id: str | None = None
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Work structure nodes
# ---------------------------------------------------------------------------


class Project(BaseNode):
    """A software project or product."""
    node_type: NodeType = NodeType.PROJECT
    description: str = ""
    # Project profile — how the agent works in this project
    repo_path: str | None = None
    repo_url: str | None = None
    stack: list[str] = Field(default_factory=list)
    run_cmd: str | None = None
    test_cmd: str | None = None
    deploy_cmd: str | None = None
    branch_strategy: str | None = None
    environments: dict[str, str] = Field(default_factory=dict)  # {"dev": "localhost", "prod": "vm999"}


class Goal(BaseNode):
    """A goal or OKR."""
    node_type: NodeType = NodeType.GOAL
    description: str = ""
    project_id: str | None = None
    owner_id: str | None = None
    due_date: datetime | None = None
    completed: bool = False
    completed_at: datetime | None = None


class Initiative(BaseNode):
    """A campaign or initiative to achieve a goal."""
    node_type: NodeType = NodeType.INITIATIVE
    description: str = ""
    goal_id: str | None = None
    project_id: str | None = None
    status: str = "active"  # active, completed, paused, cancelled


class Task(BaseNode):
    """A concrete work item."""
    node_type: NodeType = NodeType.TASK
    description: str = ""
    status: TaskStatus = TaskStatus.TODO
    initiative_id: str | None = None
    project_id: str | None = None
    assigned_to: str | None = None
    due_date: datetime | None = None
    completed_at: datetime | None = None
    result_summary: str | None = None  # brief audit trail when done


# ---------------------------------------------------------------------------
# People nodes
# ---------------------------------------------------------------------------


class Person(BaseNode):
    """A team member or internal person."""
    node_type: NodeType = NodeType.PERSON
    email: str | None = None
    role: str | None = None


class ExternalContact(BaseNode):
    """A customer, vendor, or collaborator."""
    node_type: NodeType = NodeType.EXTERNAL_CONTACT
    email: str | None = None
    company: str | None = None
    role: str | None = None


# ---------------------------------------------------------------------------
# Infrastructure nodes
# ---------------------------------------------------------------------------


class Environment(BaseNode):
    """A server, VM, or deployment environment."""
    node_type: NodeType = NodeType.ENVIRONMENT
    host: str | None = None
    ip: str | None = None
    env_type: str = "unknown"  # dev, staging, prod, ai-agent


class Tool(BaseNode):
    """A technology or tool in use."""
    node_type: NodeType = NodeType.TOOL
    version: str | None = None
    purpose: str | None = None


class Resource(BaseNode):
    """A URL, doc, repo, or reference."""
    node_type: NodeType = NodeType.RESOURCE
    url: str | None = None
    resource_type: str = "url"  # url, doc, repo, api


class Topic(BaseNode):
    """A thematic grouping."""
    node_type: NodeType = NodeType.TOPIC


# ---------------------------------------------------------------------------
# Market intelligence nodes
# ---------------------------------------------------------------------------


class Competitor(BaseNode):
    """A competing product or company."""
    node_type: NodeType = NodeType.COMPETITOR
    website: str | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    positioning: str | None = None


class MetricDataPoint(BaseModel):
    """A single time-series data point for a metric."""
    value: float
    date: datetime
    source: str = "manual"
    notes: str | None = None


class Metric(BaseNode):
    """A KPI tracked over time (visitors, trials, revenue, etc.)."""
    node_type: NodeType = NodeType.METRIC
    metric_type: str  # visitors, trials, users, revenue, etc.
    unit: str = ""  # $, count, %, etc.
    project_id: str | None = None
    data_points: list[MetricDataPoint] = Field(default_factory=list)


class CustomerFeedback(BaseNode):
    """Feedback received about a product from a customer."""
    node_type: NodeType = NodeType.CUSTOMER_FEEDBACK
    content: str
    sentiment: str | None = None  # positive, negative, neutral
    source: str = "direct"  # direct, feedback1, email, etc.
    contact_id: str | None = None
    project_id: str | None = None
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Search / retrieval models
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    """A single result from hybrid search."""
    node_id: str
    node_type: NodeType
    content: str
    score: float  # combined relevance score 0.0–1.0
    vector_score: float | None = None
    keyword_score: float | None = None
    graph_boost: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResult(BaseModel):
    """Full recall response with results and reasoning trace."""
    query: str
    results: list[SearchResult]
    total: int
    graph_context: list[dict[str, Any]] = Field(default_factory=list)
    latency_ms: int = 0


class MemoryError(BaseModel):
    """Structured error for graceful degradation."""
    error: str
    reason: str
    service: str  # redis, postgres, embeddings


# ---------------------------------------------------------------------------
# Union type for any node
# ---------------------------------------------------------------------------

AnyNode = (
    Memory | Learning | Decision | Preference | Workflow
    | Project | Goal | Initiative | Task
    | Person | ExternalContact
    | Environment | Tool | Resource | Topic
    | Competitor | Metric | CustomerFeedback
)
