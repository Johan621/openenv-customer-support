"""
Core Customer Support Triage RL Environment.

Implements the OpenEnv step/reset/state interface with:
- Per-ticket reward shaping
- Partial progress signals
- Episode-level bonus
- State management
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from models import (
    EpisodeStats,
    EnvironmentState,
    TriageAction,
    TriageObservation,
    TicketData,
)
from server.ticket_generator import TicketGenerator, TicketGroundTruth

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reward constants
# ---------------------------------------------------------------------------

CORRECT_ROUTE_REWARD = 0.50
WRONG_ROUTE_PENALTY = -0.20
URGENCY_CORRECT_REWARD = 0.25
URGENCY_OFF_BY_ONE_REWARD = 0.10
DIFFICULTY_CORRECT_REWARD = 0.10
PRIORITY_IN_RANGE_REWARD = 0.05
INEFFICIENT_ROUTING_PENALTY = -0.10
EPISODE_BONUS = 0.40

URGENCY_ORDER = ["low", "medium", "high", "critical"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

# Small epsilon to keep "score-like" values strictly inside (0, 1)
EPS = 1e-6


def clamp_open01(x: float) -> float:
    """Clamp x to the open interval (0, 1)."""
    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1.0 - EPS
    return x


class CustomerSupportEnv:
    """
    Customer Support Ticket Triage RL environment.

    API:
        reset(difficulty, seed) -> TriageObservation
        step(action)            -> TriageObservation
        state()                 -> EnvironmentState
    """

    def __init__(self) -> None:
        self.session_id: str = str(uuid.uuid4())
        self._generator: Optional[TicketGenerator] = None

        # Episode state
        self._difficulty: str = "easy"
        self._tickets: list[TicketData] = []
        self._ground_truths: list[TicketGroundTruth] = []
        self._step_index: int = 0
        self._episode_count: int = 0
        self._done: bool = True

        # Stats
        self._episode_stats: EpisodeStats = EpisodeStats()
        self._all_correct_this_episode: bool = True

    # ------------------------------------------------------------------
    # Lazy load helper
    # ------------------------------------------------------------------

    def _get_generator(self) -> TicketGenerator:
        if self._generator is None:
            self._generator = TicketGenerator()
        return self._generator

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        difficulty: str = "easy",
        seed: Optional[int] = None,
    ) -> TriageObservation:
        """Start a new episode and return the first observation."""
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(
                f"Invalid difficulty: {difficulty!r}. Must be easy/medium/hard."
            )

        self._difficulty = difficulty
        self._step_index = 0
        self._done = False
        self._all_correct_this_episode = True
        self._episode_count += 1

        episode = self._get_generator().generate_episode(difficulty, seed=seed)
        self._tickets = [t for t, _ in episode]
        self._ground_truths = [gt for _, gt in episode]

        n = len(self._tickets)

        # IMPORTANT: keep score-like values strictly in (0, 1)
        self._episode_stats = EpisodeStats(
            total_tickets=n,
            processed_tickets=0,
            correct_routes=0,
            avg_correctness=EPS,
            avg_efficiency=EPS,
            total_reward=EPS,  # was 0.0 (can fail strict validator)
        )

        logger.info(
            "Episode %d started | difficulty=%s | tickets=%d | seed=%s",
            self._episode_count,
            difficulty,
            n,
            seed,
        )

        return TriageObservation(
            ticket_info=self._tickets[0],
            correctness_score=EPS,
            efficiency_score=EPS,
            task_progress=EPS,
            difficulty_level=difficulty,  # type: ignore[arg-type]
            episode_stats=self._episode_stats.model_copy(),
            done=False,
            reward=EPS,
            metadata={
                "step_count": 0,
                "episode_count": self._episode_count,
                "session_id": self.session_id,
                "message": "Episode started. Triage the first ticket.",
            },
        )

    def step(self, action: TriageAction) -> TriageObservation:
        """Process one triage action and return the next observation."""
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if not self._tickets:
            raise RuntimeError("No tickets loaded. Call reset() first.")

        current_idx = self._step_index
        gt = self._ground_truths[current_idx]
        ticket = self._tickets[current_idx]

        # ---- Compute per-ticket scores ----
        correctness, efficiency, reward, was_correct = self._compute_reward(action, gt, ticket)

        # ---- Update episode stats ----
        processed = current_idx + 1
        stats = self._episode_stats
        total_corr = stats.avg_correctness * current_idx + correctness
        total_eff = stats.avg_efficiency * current_idx + efficiency

        if not was_correct:
            self._all_correct_this_episode = False

        stats.processed_tickets = processed
        stats.avg_correctness = total_corr / processed
        stats.avg_efficiency = total_eff / processed
        stats.total_reward += reward
        if action.route_category == gt.correct_route:
            stats.correct_routes += 1

        # Clamp episode-level score fields into (0,1) (validator safety)
        stats.avg_correctness = clamp_open01(float(stats.avg_correctness))
        stats.avg_efficiency = clamp_open01(float(stats.avg_efficiency))
        stats.total_reward = clamp_open01(float(stats.total_reward))

        # ---- Advance step ----
        self._step_index += 1
        n = len(self._tickets)
        progress = self._step_index / n

        # ---- Check done ----
        episode_done = self._step_index >= n

        # Episode bonus
        episode_bonus = 0.0
        if episode_done and self._all_correct_this_episode:
            episode_bonus = EPISODE_BONUS
            reward += episode_bonus
            stats.total_reward += episode_bonus
            # Clamp again after bonus
            stats.total_reward = clamp_open01(float(stats.total_reward))

        self._done = episode_done

        # ---- Next ticket ----
        next_ticket: Optional[TicketData] = None
        if not episode_done:
            next_ticket = self._tickets[self._step_index]

        logger.debug(
            "Step %d/%d | route=%s (want %s) | urgency=%s (want %s) | "
            "correctness=%.2f | reward=%.3f",
            self._step_index,
            n,
            action.route_category,
            gt.correct_route,
            action.urgency_assessment,
            gt.correct_urgency,
            correctness,
            reward,
        )

        # Clamp score-like fields to (0,1) for validator
        safe_correctness = clamp_open01(float(correctness))
        safe_efficiency = clamp_open01(float(efficiency))
        safe_progress = clamp_open01(float(progress)) if not episode_done else (1.0 - EPS)

        # Reward must be strictly (0,1) too (validator safety)
        reward = clamp_open01(float(reward))

        return TriageObservation(
            ticket_info=next_ticket,
            correctness_score=round(safe_correctness, 6),
            efficiency_score=round(safe_efficiency, 6),
            task_progress=round(safe_progress, 6),
            difficulty_level=self._difficulty,  # type: ignore[arg-type]
            episode_stats=stats.model_copy(),
            done=episode_done,
            reward=round(reward, 6),
            metadata={
                "step_count": self._step_index,
                "episode_count": self._episode_count,
                "session_id": self.session_id,
                "episode_bonus": round(episode_bonus, 4),
                "processed_ticket_id": ticket.ticket_id,
                "correct_route": gt.correct_route,
                "correct_urgency": gt.correct_urgency,
            },
        )

    def state(self) -> EnvironmentState:
        """Return a snapshot of the current environment state."""
        current_ticket: Optional[TicketData] = None
        if not self._done and self._tickets and self._step_index < len(self._tickets):
            current_ticket = self._tickets[self._step_index]

        return EnvironmentState(
            session_id=self.session_id,
            difficulty=self._difficulty,  # type: ignore[arg-type]
            step_count=self._step_index,
            episode_count=self._episode_count,
            done=self._done,
            episode_stats=self._episode_stats.model_copy(),
            current_ticket=current_ticket,
        )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        action: TriageAction,
        gt: TicketGroundTruth,
        ticket: TicketData,
    ) -> tuple[float, float, float, bool]:
        """
        Compute (correctness_score, efficiency_score, reward, was_correct).

        Reward structure:
          base_reward = (correctness + efficiency) / 2
          partial_reward = base_reward * 0.6
          + penalties for errors
        """
        reward = 0.0
        correctness_components: list[float] = []
        efficiency_components: list[float] = []

        # ---- Route correctness ----
        route_correct = action.route_category == gt.correct_route
        if route_correct:
            reward += CORRECT_ROUTE_REWARD
            correctness_components.append(1.0)
        else:
            reward += WRONG_ROUTE_PENALTY
            correctness_components.append(0.0)
            # Extra penalty for wildly wrong routing (e.g. spam → billing)
            if {action.route_category, gt.correct_route} in [
                {"spam", "billing"},
                {"spam", "technical"},
                {"feature", "spam"},
            ]:
                reward += INEFFICIENT_ROUTING_PENALTY

        # ---- Urgency correctness ----
        urgency_correct = action.urgency_assessment == gt.correct_urgency
        if urgency_correct:
            reward += URGENCY_CORRECT_REWARD
            correctness_components.append(1.0)
        else:
            # Partial credit for off-by-one
            pred_idx = URGENCY_ORDER.index(action.urgency_assessment)
            true_idx = URGENCY_ORDER.index(gt.correct_urgency)
            if abs(pred_idx - true_idx) == 1:
                reward += URGENCY_OFF_BY_ONE_REWARD
                correctness_components.append(0.5)
            else:
                correctness_components.append(0.0)

        # ---- Resolution difficulty ----
        diff_correct = action.resolution_difficulty == gt.correct_difficulty
        if diff_correct:
            reward += DIFFICULTY_CORRECT_REWARD
            efficiency_components.append(1.0)
        else:
            pred_didx = DIFFICULTY_ORDER.index(action.resolution_difficulty)
            true_didx = DIFFICULTY_ORDER.index(gt.correct_difficulty)
            efficiency_components.append(max(0.0, 1.0 - abs(pred_didx - true_didx) * 0.5))

        # ---- Priority score in range ----
        lo, hi = gt.optimal_priority_range
        if lo <= action.priority_score <= hi:
            reward += PRIORITY_IN_RANGE_REWARD
            efficiency_components.append(1.0)
        else:
            # Partial credit based on distance
            dist = min(abs(action.priority_score - lo), abs(action.priority_score - hi))
            partial = max(0.0, 1.0 - dist / 50.0)
            efficiency_components.append(partial)

        # ---- Sentiment consistency bonus ----
        # Critical/high urgency should have lower sentiment tickets → small bonus if consistent
        if ticket.customer_sentiment < -0.3 and action.urgency_assessment in ("high", "critical"):
            efficiency_components.append(1.0)
        elif ticket.customer_sentiment > 0.5 and action.urgency_assessment in ("low", "medium"):
            efficiency_components.append(1.0)

        correctness_score = (
            sum(correctness_components) / len(correctness_components)
            if correctness_components else 0.0
        )
        efficiency_score = (
            sum(efficiency_components) / len(efficiency_components)
            if efficiency_components else 0.0
        )

        was_correct = route_correct and urgency_correct and diff_correct

        return correctness_score, efficiency_score, reward, was_correct
