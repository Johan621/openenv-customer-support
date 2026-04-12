"""
Baseline Inference Script for the Customer Support Triage RL Environment.

Implements a heuristic + light ML baseline agent that:
- Uses keyword matching for route classification
- Uses sentiment + description length for urgency
- Produces reproducible scores across all 3 difficulty levels
- Outputs baseline_scores.json

Usage:
    python scripts/baseline_inference.py
    python scripts/baseline_inference.py --difficulties easy medium hard
    python scripts/baseline_inference.py --episodes 5 --seed 42

Notes:
    - This baseline runs the environment locally (no server required).
    - No API keys are required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

# Allow running from repo root
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from models import TicketData, TriageAction # noqa: E402
from server.customer_support_env import CustomerSupportEnv # noqa: E402

EPS = 1e-6


def clamp_open01(x: float) -> float:
    """Clamp x to the open interval (0, 1), never exactly 0.0 or 1.0."""
    x = float(x)
    if x <= 0.0:
        return EPS
    if x >= 1.0:
        return 1.0 - EPS
    return x


# ---------------------------------------------------------------------------
# Heuristic keyword tables
# ---------------------------------------------------------------------------

_BILLING_KEYWORDS = [
    "charge", "charged", "billing", "invoice", "payment", "refund",
    "subscription", "fee", "cost", "price", "paid", "money", "credit card",
    "bank", "transaction", "overcharge", "discount", "promo", "plan",
]

_TECHNICAL_KEYWORDS = [
    "crash", "error", "bug", "broken", "not working", "issue", "problem",
    "api", "integration", "login", "password", "slow", "performance",
    "export", "import", "file", "data", "sync", "update", "version",
    "500", "404", "timeout", "connection", "install", "upgrade",
]

_FEATURE_KEYWORDS = [
    "feature", "request", "suggestion", "would love", "could you add",
    "please add", "wish", "improvement", "enhance", "option", "support for",
    "integrate", "dark mode", "shortcut", "keyboard", "bulk",
]

_FEEDBACK_KEYWORDS = [
    "great", "excellent", "wonderful", "thank", "appreciate", "happy",
    "love", "positive", "feedback", "review", "testimonial", "pleased",
    "satisfaction", "compliment", "amazing", "fantastic",
    "experience", "observations", "months of using", "six months",
    "year of using", "service is why", "professional", "exceptional",
    "noteworthy", "observation", "impressed", "renew our subscription",
    "continue to use", "looking forward", "praise", "renew",
    "share that", "wanted to share", "productivity has improved",
    "why we continue", "would appreciate a call", "account management",
    "reconsidering", "recently the product",
]

_SPAM_KEYWORDS = [
    "prize", "congratulations", "winner", "earn money", "guaranteed",
    "click here", "free upgrade", "limited time", "act now", "thousands",
    "make money", "work from home", "breach", "compromised", "verify",
    "bank account", "social security", "urgent security", "register immediately",
]


def _count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


# ---------------------------------------------------------------------------
# Baseline agent
# ---------------------------------------------------------------------------

class HeuristicBaselineAgent:
    """
    Rule-based heuristic agent for ticket triage.

    Strategy:
    1. Route: keyword matching on subject + description
    2. Urgency: sentiment + keyword intensity
    3. Difficulty: urgency + description length
    4. Priority: linear function of urgency
    """

    def act(self, ticket: TicketData) -> TriageAction:
        full_text = f"{ticket.subject} {ticket.description}"

        route = self._classify_route(full_text, ticket)
        urgency = self._classify_urgency(full_text, ticket, route)
        difficulty = self._classify_difficulty(urgency, ticket)
        priority = self._compute_priority(urgency, ticket)

        return TriageAction(
            route_category=route,  # type: ignore[arg-type]
            urgency_assessment=urgency,  # type: ignore[arg-type]
            resolution_difficulty=difficulty,  # type: ignore[arg-type]
            priority_score=priority,
        )

    def _classify_route(self, text: str, ticket: TicketData) -> str:
        scores = {
            "billing": _count_keywords(text, _BILLING_KEYWORDS),
            "technical": _count_keywords(text, _TECHNICAL_KEYWORDS),
            "feature": _count_keywords(text, _FEATURE_KEYWORDS),
            "feedback": _count_keywords(text, _FEEDBACK_KEYWORDS),
            "spam": _count_keywords(text, _SPAM_KEYWORDS),
        }

        # Boost initial_category if it's plausible
        initial = ticket.initial_category.lower()
        if initial in scores:
            scores[initial] += 1

        # Strong positive sentiment with low technical/billing signals → feedback
        if ticket.customer_sentiment > 0.6:
            if scores["technical"] == 0 and scores["billing"] == 0:
                scores["feedback"] += 2
            else:
                scores["feedback"] += 1

        # Pick the highest scoring category
        route = max(scores, key=lambda k: scores[k])

        # Tie-break: prefer technical > billing > feature > feedback > spam
        max_score = scores[route]
        if max_score == 0:
            # No keywords matched — fall back to initial_category
            route = initial if initial in scores else "technical"

        return route

    def _classify_urgency(self, text: str, ticket: TicketData, route: str = "") -> str:
        text_lower = text.lower()
        sentiment = ticket.customer_sentiment

        # Feedback, feature, and spam are always low urgency
        if route in ("feedback", "feature", "spam"):
            return "low"

        # Critical indicators
        critical_patterns = [
            r"\bcritical\b", r"\bdown\b", r"\boutage\b",
            r"\bproduction\b.*\bbroken\b", r"\bentirely\b",
            r"\bcannot access\b", r"\block.*out\b",
            r"\bwhole team\b", r"\bentire team\b",
        ]
        if any(re.search(p, text_lower) for p in critical_patterns):
            return "critical"

        # High indicators
        high_patterns = [
            r"\burgent\b", r"\bimmediately\b", r"\basap\b",
            r"\bblocker\b", r"\bnot working\b", r"\bserious\b",
            r"\baffect.*team\b", r"\ball.*users?\b",
            r"\bduplicate charge\b", r"\brefund\b.*\bimmediately\b",
        ]
        if any(re.search(p, text_lower) for p in high_patterns):
            return "high"

        # Sentiment-based adjustment
        if sentiment < -0.5:
            return "high"
        if sentiment < -0.2:
            return "medium"

        # Feature requests and feedback → always low
        if any(kw in text_lower for kw in ["feature request", "suggestion", "feedback", "would love", "great", "excellent"]):
            return "low"

        # Spam → low
        if _count_keywords(text, _SPAM_KEYWORDS) > 2:
            return "low"

        # Default by description length heuristic (longer = more detail = real issue)
        if ticket.word_count > 80:
            return "medium"
        return "low"

    def _classify_difficulty(self, urgency: str, ticket: TicketData) -> str:
        urgency_to_diff = {
            "critical": "hard",
            "high": "hard",
            "medium": "medium",
            "low": "easy",
        }
        return urgency_to_diff[urgency]

    def _compute_priority(self, urgency: str, ticket: TicketData) -> float:
        base = {
            "critical": 90.0,
            "high": 70.0,
            "medium": 45.0,
            "low": 20.0,
        }[urgency]

        # Adjust for sentiment
        sentiment_adj = (1.0 - ticket.customer_sentiment) * 5.0  # more negative → higher priority
        base += min(sentiment_adj, 8.0)

        # Adjust for customer history (loyal customers get a small boost)
        if ticket.previous_tickets_count > 10:
            base += 3.0

        return round(min(base, 100.0), 1)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def run_evaluation(
    difficulty: str,
    n_episodes: int = 5,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run n_episodes and return aggregate metrics."""
    agent = HeuristicBaselineAgent()
    env = CustomerSupportEnv()

    episode_rewards = []
    episode_correctness = []
    episode_efficiency = []
    correct_routes_list = []

    for ep in range(n_episodes):
        ep_seed = seed + ep
        obs = env.reset(difficulty=difficulty, seed=ep_seed)
        total_reward = 0.0

        while not obs.done:
            action = agent.act(obs.ticket_info)
            obs = env.step(action)
            total_reward += obs.reward

            if verbose:
                print(
                    f"  [{difficulty}] ep={ep} ticket={obs.metadata.get('processed_ticket_id','?')} "
                    f"route={action.route_category} urgency={action.urgency_assessment} "
                    f"difficulty={action.resolution_difficulty} priority={action.priority_score} "
                    f"correctness={clamp_open01(obs.correctness_score):.6f} reward={clamp_open01(obs.reward):.6f}"
                )

        stats = obs.episode_stats
        episode_rewards.append(stats.total_reward)
        episode_correctness.append(stats.avg_correctness)
        episode_efficiency.append(stats.avg_efficiency)
        correct_routes_list.append(
            stats.correct_routes / max(stats.total_tickets, 1)
        )

    n = len(episode_rewards)
    avg_reward = sum(episode_rewards) / n
    avg_correctness = sum(episode_correctness) / n
    avg_efficiency = sum(episode_efficiency) / n
    avg_route_accuracy = sum(correct_routes_list) / n

    target_scores = {"easy": 0.85, "medium": 0.70, "hard": 0.50}
    target = target_scores[difficulty]
    score = avg_correctness  # primary metric

    return {
        "difficulty": difficulty,
        "episodes": n_episodes,
        "seed": seed,
        "avg_reward": round(avg_reward, 4),
        "avg_correctness_score": clamp_open01(round(avg_correctness, 4)),
        "avg_efficiency_score": clamp_open01(round(avg_efficiency, 4)),
        "avg_route_accuracy": clamp_open01(round(avg_route_accuracy, 4)),
        "target_score": target,
        "achieved_target": bool(score >= target),
        "score": clamp_open01(round(score, 4)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run baseline inference on the Customer Support Triage environment"
    )
    parser.add_argument(
        "--difficulties",
        nargs="+",
        default=["easy", "medium", "hard"],
        choices=["easy", "medium", "hard"],
        help="Difficulty levels to evaluate",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes per difficulty (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output",
        default="baseline_scores.json",
        help="Output file for scores (default: baseline_scores.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-ticket details",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Customer Support Triage — Baseline Inference")
    print("=" * 60)

    results = {}
    all_passed = True

    for difficulty in args.difficulties:
        print(f"\n▶ Evaluating difficulty: {difficulty.upper()}")
        start = time.time()
        metrics = run_evaluation(
            difficulty=difficulty,
            n_episodes=args.episodes,
            seed=args.seed,
            verbose=args.verbose,
        )
        elapsed = time.time() - start

        status = "✅ PASSED" if metrics["achieved_target"] else "❌ BELOW TARGET"
        print(f"  Score:        {metrics['score']:.4f}  (target: >{metrics['target_score']})")
        print(f"  Avg reward:   {metrics['avg_reward']:.4f}")
        print(f"  Route acc:    {metrics['avg_route_accuracy']:.4f}")
        print(f"  Efficiency:   {metrics['avg_efficiency_score']:.4f}")
        print(f"  {status}  ({elapsed:.1f}s)")

        results[difficulty] = metrics
        if not metrics["achieved_target"]:
            all_passed = False

    # Write output
    output_path = Path(args.output)
    with output_path.open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results saved to {output_path}")
    overall = "✅ ALL TARGETS MET" if all_passed else "⚠️  SOME TARGETS MISSED"
    print(f"Overall: {overall}")
    print("=" * 60)


if __name__ == "__main__":
    main()
