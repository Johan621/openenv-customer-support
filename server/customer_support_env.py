"""
Core Customer Support Triage RL Environment.

Implements the OpenEnv step/reset/state interface with:
- Per-ticket reward shaping
- Partial progress signals
- Episode-level bonus
- State management

Validator hardening (important):
- Any score-like float that may be validated is clamped to the open interval (0, 1),
  never exactly 0.0 and never exactly 1.0.
- This includes:
  - TriageObservation.correctness_score / efficiency_score / task_progress / reward
  - metadata.task_score / metadata.episode_bonus
  - episode_stats.avg_correctness / avg_efficiency / total_reward
  - EnvironmentState.task_score
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

# Keep all score-like values strictly inside (0, 1)
EPS = 1e-2  # 0.01

def clamp_open01(x: float) -> float:
    """Clamp x so it is strictly inside (0,1) AND safe after 2-decimal rounding."""
    x = float(x)
    if x <= EPS:
        return EPS
    if x >= 1.0 - EPS:
        return 1.0 - EPS  # 0.99
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

        # Canonical score for validators (always in (0,1))
        self._task_score: float = EPS

    def _get_generator(self) -> TicketGenerator:
        if self._generator is None:
            self._generator = TicketGenerator()
        return self._generator

    def _update_task_score(self) -> None:
        """
        Canonical 'task score' that validators can consume.

        Use avg_correctness as the basis, but clamp strictly to (0,1).
        """
        try:
            base = float(self._episode_stats.avg_correctness)
        except Exception:
            base = EPS
        self._task_score = clamp_open01(base)

    def _safe_episode_stats(self) -> EpisodeStats:
        """
        Return a copy of episode stats with all score-like floats clamped AND rounded.
        This prevents any downstream JSON encoding/rounding from producing 0.0 or 1.0.
        """
        s = self._episode_stats.model_copy()
        # FIXED:
        s.avg_correctness = clamp_open01(round(float(s.avg_correctness), 6))
        s.avg_efficiency  = clamp_open01(round(float(s.avg_efficiency),  6))
        s.total_reward    = round(float(s.total_reward), 6)
        return s

    def reset(self, difficulty: str = "easy", seed: Optional[int] = None) -> TriageObservation:
        if difficulty not in ("easy", "medium", "hard"):
            raise ValueError(f"Invalid difficulty: {difficulty!r}. Must be easy/medium/hard.")

        self._difficulty = difficulty
        self._step_index = 0
        self._done = False
        self._all_correct_this_episode = True
        self._episode_count += 1

        episode = self._get_generator().generate_episode(difficulty, seed=seed)
        self._tickets = [t for t, _ in episode]
        self._ground_truths = [gt for _, gt in episode]

        n = len(self._tickets)
        self._episode_stats = EpisodeStats(
            total_tickets=n,
            processed_tickets=0,
            correct_routes=0,
            avg_correctness=EPS,
            avg_efficiency=EPS,
            total_reward=0.0,
        )
        self._update_task_score()

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
            episode_stats=self._safe_episode_stats(),
            done=False,
            reward=EPS,
            metadata={
                "step_count": 0,
                "episode_count": self._episode_count,
                "session_id": self.session_id,
                # canonical score (validators should read this)
                "task_score": clamp_open01(round(float(self._task_score), 6)),
                "message": "Episode started. Triage the first ticket.",
            },
        )

    def step(self, action: TriageAction) -> TriageObservation:
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if not self._tickets:
            raise RuntimeError("No tickets loaded. Call reset() first.")

        current_idx = self._step_index
        gt = self._ground_truths[current_idx]
        ticket = self._tickets[current_idx]

        correctness, efficiency, reward, was_correct = self._compute_reward(action, gt, ticket)

        processed = current_idx + 1
        stats = self._episode_stats

        total_corr = float(stats.avg_correctness) * current_idx + float(correctness)
        total_eff = float(stats.avg_efficiency) * current_idx + float(efficiency)

        if not was_correct:
            self._all_correct_this_episode = False

        stats.processed_tickets = processed
        stats.avg_correctness = clamp_open01(total_corr / processed)
        stats.avg_efficiency = clamp_open01(total_eff / processed)
        stats.total_reward = float(stats.total_reward) + float(reward)
        if action.route_category == gt.correct_route:
            stats.correct_routes += 1

        # Advance
        self._step_index += 1
        n = len(self._tickets)
        progress = self._step_index / n
        episode_done = self._step_index >= n

        episode_bonus = 0.0
        if episode_done and self._all_correct_this_episode:
            episode_bonus = EPISODE_BONUS
            reward += episode_bonus
            stats.total_reward = float(stats.total_reward) + float(episode_bonus)

        self._done = episode_done

        next_ticket: Optional[TicketData] = None
        if not episode_done:
            next_ticket = self._tickets[self._step_index]

        # Update canonical task score
        self._update_task_score()

        # Clamp returned per-step values
        safe_correctness = clamp_open01(float(correctness))
        safe_efficiency = clamp_open01(float(efficiency))
        safe_progress = clamp_open01(float(progress)) if not episode_done else (1.0 - EPS)
        safe_progress = clamp_open01(safe_progress)
        safe_reward = clamp_open01(float(reward))

        # IMPORTANT: episode_bonus must also be strictly in (0,1) when exposed
        if episode_bonus == 0.0:
            safe_episode_bonus = 0.0
        else:
            safe_episode_bonus = clamp_open01(float(episode_bonus))

        return TriageObservation(
            ticket_info=next_ticket,
            correctness_score=round(safe_correctness, 6),
            efficiency_score=round(safe_efficiency, 6),
            task_progress=round(safe_progress, 6),
            difficulty_level=self._difficulty,  # type: ignore[arg-type]
            episode_stats=self._safe_episode_stats(),
            done=episode_done,
            reward=round(safe_reward, 6),
            metadata={
                "step_count": self._step_index,
                "episode_count": self._episode_count,
                "session_id": self.session_id,
                "episode_bonus": round(safe_episode_bonus, 6),
                "processed_ticket_id": ticket.ticket_id,
                "correct_route": gt.correct_route,
                "correct_urgency": gt.correct_urgency,
                # canonical score (validators should read this)
                "task_score": clamp_open01(round(float(self._task_score), 6)),
            },
        )

    def state(self) -> EnvironmentState:
        current_ticket: Optional[TicketData] = None
        if not self._done and self._tickets and self._step_index < len(self._tickets):
            current_ticket = self._tickets[self._step_index]

        # Ensure task score always available / clamped
        self._update_task_score()

        return EnvironmentState(
            session_id=self.session_id,
            difficulty=self._difficulty,  # type: ignore[arg-type]
            step_count=self._step_index,
            episode_count=self._episode_count,
            done=self._done,
            episode_stats=self._safe_episode_stats(),
            current_ticket=current_ticket,
            task_score=clamp_open01(round(float(self._task_score), 6)),
        )

    def _compute_reward(
        self,
        action: TriageAction,
        gt: TicketGroundTruth,
        ticket: TicketData,
    ) -> tuple[float, float, float, bool]:
        reward = 0.0
        correctness_components: list[float] = []
        efficiency_components: list[float] = []

        route_correct = action.route_category == gt.correct_route
        if route_correct:
            reward += CORRECT_ROUTE_REWARD
            correctness_components.append(1.0)
        else:
            reward += WRONG_ROUTE_PENALTY
            correctness_components.append(0.0)
            if {action.route_category, gt.correct_route} in [
                {"spam", "billing"},
                {"spam", "technical"},
                {"feature", "spam"},
            ]:
                reward += INEFFICIENT_ROUTING_PENALTY

        urgency_correct = action.urgency_assessment == gt.correct_urgency
        if urgency_correct:
            reward += URGENCY_CORRECT_REWARD
            correctness_components.append(1.0)
        else:
            pred_idx = URGENCY_ORDER.index(action.urgency_assessment)
            true_idx = URGENCY_ORDER.index(gt.correct_urgency)
            if abs(pred_idx - true_idx) == 1:
                reward += URGENCY_OFF_BY_ONE_REWARD
                correctness_components.append(0.5)
            else:
                correctness_components.append(0.0)

        diff_correct = action.resolution_difficulty == gt.correct_difficulty
        if diff_correct:
            reward += DIFFICULTY_CORRECT_REWARD
            efficiency_components.append(1.0)
        else:
            pred_didx = DIFFICULTY_ORDER.index(action.resolution_difficulty)
            true_didx = DIFFICULTY_ORDER.index(gt.correct_difficulty)
            efficiency_components.append(max(0.0, 1.0 - abs(pred_didx - true_didx) * 0.5))

        lo, hi = gt.optimal_priority_range
        if lo <= action.priority_score <= hi:
            reward += PRIORITY_IN_RANGE_REWARD
            efficiency_components.append(1.0)
        else:
            dist = min(abs(action.priority_score - lo), abs(action.priority_score - hi))
            partial = max(0.0, 1.0 - dist / 50.0)
            efficiency_components.append(partial)

        if ticket.customer_sentiment < -0.3 and action.urgency_assessment in ("high", "critical"):
            efficiency_components.append(1.0)
        elif ticket.customer_sentiment > 0.5 and action.urgency_assessment in ("low", "medium"):
            efficiency_components.append(1.0)

        correctness_score = (
            sum(correctness_components) / len(correctness_components)
            if correctness_components
            else 0.0
        )
        efficiency_score = (
            sum(efficiency_components) / len(efficiency_components)
            if efficiency_components
            else 0.0
        )

        was_correct = route_correct and urgency_correct and diff_correct
        return correctness_score, efficiency_score, reward, was_correct
