"""
Basic tests for the Customer Support Triage RL Environment.

Tests cover:
- Model validation
- Ticket generation
- Environment reset/step/state
- Reward computation
- All difficulty levels
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure repo root is on path
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from models import (
    EpisodeStats,
    EnvironmentState,
    TriageAction,
    TriageObservation,
    TicketData,
    ResetRequest,
    StepRequest,
)
from server.customer_support_env import CustomerSupportEnv
from server.ticket_generator import TicketGenerator


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestTriageAction:
    def test_valid_action(self):
        action = TriageAction(
            route_category="billing",
            urgency_assessment="high",
            resolution_difficulty="hard",
            priority_score=75.0,
        )
        assert action.route_category == "billing"
        assert action.urgency_assessment == "high"
        assert action.resolution_difficulty == "hard"
        assert action.priority_score == 75.0

    def test_priority_clamped(self):
        """priority_score must be 0-100."""
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            TriageAction(
                route_category="technical",
                urgency_assessment="low",
                resolution_difficulty="easy",
                priority_score=150.0,
            )

    def test_invalid_route(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            TriageAction(
                route_category="unknown_route",
                urgency_assessment="low",
                resolution_difficulty="easy",
                priority_score=10.0,
            )


class TestTicketData:
    def test_sentiment_range(self):
        import pydantic
        with pytest.raises(pydantic.ValidationError):
            TicketData(
                ticket_id="T1",
                subject="Test",
                initial_category="billing",
                description="test description here",
                customer_sentiment=2.0,  # invalid
                word_count=3,
                customer_account_age=100,
                previous_tickets_count=0,
            )


# ---------------------------------------------------------------------------
# Ticket generator tests
# ---------------------------------------------------------------------------

class TestTicketGenerator:
    def test_generate_easy_ticket(self):
        gen = TicketGenerator(seed=42)
        ticket, gt = gen.generate_ticket("easy")
        assert ticket.ticket_id.startswith("TKT-")
        assert ticket.subject
        assert len(ticket.description) > 10
        assert -1.0 <= ticket.customer_sentiment <= 1.0
        assert gt.correct_route in ("billing", "technical", "feature", "feedback", "spam")
        assert gt.correct_urgency in ("low", "medium", "high", "critical")

    def test_generate_episode_sizes(self):
        gen = TicketGenerator(seed=1)
        for difficulty, expected in [("easy", 5), ("medium", 10), ("hard", 15)]:
            episode = gen.generate_episode(difficulty)
            assert len(episode) == expected, f"Expected {expected} tickets for {difficulty}"

    def test_seed_reproducibility(self):
        gen1 = TicketGenerator(seed=99)
        gen2 = TicketGenerator(seed=99)
        ticket1, _ = gen1.generate_ticket("easy")
        ticket2, _ = gen2.generate_ticket("easy")
        assert ticket1.subject == ticket2.subject
        assert ticket1.description == ticket2.description

    def test_hard_has_more_ambiguity(self):
        """Hard episodes should have more tickets with wrong initial_category."""
        gen = TicketGenerator(seed=7)
        easy_ep = gen.generate_episode("easy", seed=7)
        gen2 = TicketGenerator(seed=7)
        hard_ep = gen2.generate_episode("hard", seed=7)

        # Check ground truths exist for all tickets
        for ticket, gt in hard_ep:
            assert gt.correct_route in ("billing", "technical", "feature", "feedback", "spam")


# ---------------------------------------------------------------------------
# Environment tests
# ---------------------------------------------------------------------------

class TestCustomerSupportEnv:
    def test_reset_returns_observation(self):
        env = CustomerSupportEnv()
        obs = env.reset("easy", seed=42)
        assert isinstance(obs, TriageObservation)
        assert obs.ticket_info is not None
        assert obs.done is False
        assert obs.difficulty_level == "easy"
        assert obs.task_progress == 0.0

    def test_full_easy_episode(self):
        env = CustomerSupportEnv()
        obs = env.reset("easy", seed=42)
        total_reward = 0.0
        steps = 0

        while not obs.done:
            action = TriageAction(
                route_category="technical",
                urgency_assessment="medium",
                resolution_difficulty="medium",
                priority_score=50.0,
            )
            obs = env.step(action)
            total_reward += obs.reward
            steps += 1

        assert steps == 5  # easy = 5 tickets
        assert obs.done is True
        assert obs.ticket_info is None
        assert obs.episode_stats.total_tickets == 5
        assert obs.episode_stats.processed_tickets == 5

    def test_full_medium_episode(self):
        env = CustomerSupportEnv()
        obs = env.reset("medium", seed=42)
        steps = 0
        while not obs.done:
            action = TriageAction(
                route_category="billing",
                urgency_assessment="low",
                resolution_difficulty="easy",
                priority_score=20.0,
            )
            obs = env.step(action)
            steps += 1
        assert steps == 10

    def test_full_hard_episode(self):
        env = CustomerSupportEnv()
        obs = env.reset("hard", seed=42)
        steps = 0
        while not obs.done:
            action = TriageAction(
                route_category="technical",
                urgency_assessment="high",
                resolution_difficulty="hard",
                priority_score=80.0,
            )
            obs = env.step(action)
            steps += 1
        assert steps == 15

    def test_step_after_done_raises(self):
        env = CustomerSupportEnv()
        obs = env.reset("easy", seed=1)
        while not obs.done:
            obs = env.step(
                TriageAction(
                    route_category="feedback",
                    urgency_assessment="low",
                    resolution_difficulty="easy",
                    priority_score=15.0,
                )
            )
        with pytest.raises(RuntimeError):
            env.step(
                TriageAction(
                    route_category="feedback",
                    urgency_assessment="low",
                    resolution_difficulty="easy",
                    priority_score=15.0,
                )
            )

    def test_state_method(self):
        env = CustomerSupportEnv()
        env.reset("medium", seed=5)
        state = env.state()
        assert isinstance(state, EnvironmentState)
        assert state.difficulty == "medium"
        assert state.done is False
        assert state.current_ticket is not None

    def test_invalid_difficulty_raises(self):
        env = CustomerSupportEnv()
        with pytest.raises(ValueError):
            env.reset("extreme")

    def test_reward_for_correct_action(self):
        """Correct routing + urgency should yield higher reward."""
        from server.customer_support_env import (
            CORRECT_ROUTE_REWARD,
            URGENCY_CORRECT_REWARD,
            DIFFICULTY_CORRECT_REWARD,
            PRIORITY_IN_RANGE_REWARD,
        )
        env = CustomerSupportEnv()
        env.reset("easy", seed=42)
        gt = env._ground_truths[0]

        correct_action = TriageAction(
            route_category=gt.correct_route,  # type: ignore[arg-type]
            urgency_assessment=gt.correct_urgency,  # type: ignore[arg-type]
            resolution_difficulty=gt.correct_difficulty,  # type: ignore[arg-type]
            priority_score=(gt.optimal_priority_range[0] + gt.optimal_priority_range[1]) / 2,
        )
        obs_correct = env.step(correct_action)

        # A new env for wrong action comparison
        env2 = CustomerSupportEnv()
        env2.reset("easy", seed=42)
        wrong_action = TriageAction(
            route_category="spam",
            urgency_assessment="low",
            resolution_difficulty="easy",
            priority_score=5.0,
        )
        obs_wrong = env2.step(wrong_action)

        assert obs_correct.reward > obs_wrong.reward, (
            f"Correct reward {obs_correct.reward} should exceed wrong reward {obs_wrong.reward}"
        )

    def test_episode_bonus_on_perfect_episode(self):
        """A perfect episode should receive the episode bonus."""
        from server.customer_support_env import EPISODE_BONUS

        env = CustomerSupportEnv()
        obs = env.reset("easy", seed=42)

        last_obs = obs
        while not obs.done:
            gt = env._ground_truths[env._step_index]
            perfect_action = TriageAction(
                route_category=gt.correct_route,  # type: ignore[arg-type]
                urgency_assessment=gt.correct_urgency,  # type: ignore[arg-type]
                resolution_difficulty=gt.correct_difficulty,  # type: ignore[arg-type]
                priority_score=(gt.optimal_priority_range[0] + gt.optimal_priority_range[1]) / 2,
            )
            last_obs = env.step(perfect_action)
            obs = last_obs

        assert last_obs.metadata["episode_bonus"] == EPISODE_BONUS

    def test_task_progress_increases(self):
        env = CustomerSupportEnv()
        obs = env.reset("easy", seed=42)
        prev_progress = 0.0
        while not obs.done:
            action = TriageAction(
                route_category="technical",
                urgency_assessment="medium",
                resolution_difficulty="medium",
                priority_score=50.0,
            )
            obs = env.step(action)
            assert obs.task_progress >= prev_progress
            prev_progress = obs.task_progress
        assert prev_progress == 1.0

    def test_multiple_episodes(self):
        """Environment should support multiple sequential episodes."""
        env = CustomerSupportEnv()
        for ep in range(3):
            obs = env.reset("easy", seed=ep)
            assert not obs.done
            while not obs.done:
                obs = env.step(
                    TriageAction(
                        route_category="technical",
                        urgency_assessment="medium",
                        resolution_difficulty="medium",
                        priority_score=50.0,
                    )
                )
            assert obs.done

        assert env._episode_count == 3


# ---------------------------------------------------------------------------
# Baseline agent tests
# ---------------------------------------------------------------------------

class TestBaselineAgent:
    def test_baseline_runs_on_all_difficulties(self):
        from scripts.baseline_inference import run_evaluation

        for difficulty in ("easy", "medium", "hard"):
            result = run_evaluation(difficulty, n_episodes=2, seed=42)
            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0
            assert result["difficulty"] == difficulty

    def test_baseline_reproducible(self):
        from scripts.baseline_inference import run_evaluation

        r1 = run_evaluation("easy", n_episodes=2, seed=99)
        r2 = run_evaluation("easy", n_episodes=2, seed=99)
        assert r1["score"] == r2["score"]
        assert r1["avg_reward"] == r2["avg_reward"]
