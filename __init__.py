"""
openenv-customer-support
========================

Real-world Customer Support Ticket Triage RL Environment for the
Meta-PyTorch-HuggingFace OpenEnv Hackathon Round 1.
"""

from models import (
    EpisodeStats,
    EnvironmentState,
    ResetRequest,
    StepRequest,
    TicketData,
    TriageAction,
    TriageObservation,
)
from server.customer_support_env import CustomerSupportEnv
from client import CustomerSupportEnvClient

__all__ = [
    "CustomerSupportEnv",
    "CustomerSupportEnvClient",
    "EpisodeStats",
    "EnvironmentState",
    "ResetRequest",
    "StepRequest",
    "TicketData",
    "TriageAction",
    "TriageObservation",
]

__version__ = "0.1.0"
