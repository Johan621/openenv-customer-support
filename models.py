"""
Pydantic models for the Customer Support Ticket Triage RL Environment.

Defines the typed Action and Observation spaces as required by the OpenEnv spec.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Keep score-like defaults strictly inside (0, 1) for strict validators
EPS = 1e-6


# ---------------------------------------------------------------------------
# Ticket data model
# ---------------------------------------------------------------------------

class TicketData(BaseModel):
    """Represents a single customer support ticket presented to the agent."""

    ticket_id: str = Field(..., description="Unique ticket identifier")
    subject: str = Field(..., description="Brief ticket subject (5-15 words)")
    initial_category: str = Field(
        ...,
        description="Customer-supplied category guess (may be incorrect)",
    )
    description: str = Field(..., description="Full ticket description (20-200 words)")
    customer_sentiment: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment polarity score from -1.0 (negative) to 1.0 (positive)",
    )
    word_count: int = Field(..., ge=0, description="Word count of the description")
    customer_account_age: int = Field(
        ..., ge=0, description="Days since account creation"
    )
    previous_tickets_count: int = Field(
        ..., ge=0, description="Number of previous support tickets from this customer"
    )


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------

RouteCategory = Literal["billing", "technical", "feature", "feedback", "spam"]
UrgencyLevel = Literal["low", "medium", "high", "critical"]
ResolutionDifficulty = Literal["easy", "medium", "hard"]


class TriageAction(BaseModel):
    """
    Action the agent takes for each ticket.
    """

    route_category: RouteCategory = Field(
        ...,
        description="Routing destination: billing | technical | feature | feedback | spam",
    )
    urgency_assessment: UrgencyLevel = Field(
        ...,
        description="Urgency classification: low | medium | high | critical",
    )
    resolution_difficulty: ResolutionDifficulty = Field(
        ...,
        description="Predicted resolution difficulty: easy | medium | hard",
    )
    priority_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Numerical priority 0-100 (higher = more urgent)",
    )

    @field_validator("priority_score")
    @classmethod
    def round_priority(cls, v: float) -> float:
        return round(v, 2)


# ---------------------------------------------------------------------------
# Observation space
# ---------------------------------------------------------------------------

class EpisodeStats(BaseModel):
    """Aggregate statistics for the current episode."""

    total_tickets: int = Field(0, description="Total tickets in this episode")
    processed_tickets: int = Field(0, description="Tickets processed so far")
    correct_routes: int = Field(0, description="Correctly routed tickets")

    # Defaults must be strictly inside (0,1)
    avg_correctness: float = Field(EPS, description="Running average correctness score")
    avg_efficiency: float = Field(EPS, description="Running average efficiency score")
    total_reward: float = Field(EPS, description="Cumulative reward this episode")


class TriageObservation(BaseModel):
    """
    Observation returned by the environment after each step.
    """

    ticket_info: Optional[TicketData] = Field(
        None, description="Current ticket to triage (None when episode is done)"
    )

    # Defaults must be strictly inside (0,1)
    correctness_score: float = Field(
        EPS,
        ge=0.0,
        le=1.0,
        description="Triage correctness for the previous action (0-1)",
    )
    efficiency_score: float = Field(
        EPS,
        ge=0.0,
        le=1.0,
        description="Routing efficiency for the previous action (0-1)",
    )
    task_progress: float = Field(
        EPS,
        ge=0.0,
        le=1.0,
        description="Fraction of tickets processed in the current episode",
    )

    difficulty_level: Literal["easy", "medium", "hard"] = Field(
        "easy", description="Current difficulty level"
    )
    episode_stats: EpisodeStats = Field(
        default_factory=EpisodeStats,
        description="Aggregate episode statistics",
    )
    done: bool = Field(False, description="True when the episode has ended")

    # Treat as score-like too for strict validators
    reward: float = Field(EPS, description="Step reward (partial progress signal)")

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (step_count, episode_count, etc.)",
    )


# ---------------------------------------------------------------------------
# Reset / State API helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Request body for POST /reset."""

    difficulty: Literal["easy", "medium", "hard"] = Field(
        "easy", description="Difficulty level for the new episode"
    )
    seed: Optional[int] = Field(
        None, description="Random seed for reproducibility (optional)"
    )


class StepRequest(BaseModel):
    """Request body for POST /step."""
    action: TriageAction


class EnvironmentState(BaseModel):
    """Snapshot returned by GET /state."""

    session_id: str
    difficulty: Literal["easy", "medium", "hard"]
    step_count: int
    episode_count: int
    done: bool
    episode_stats: EpisodeStats
    current_ticket: Optional[TicketData] = None
