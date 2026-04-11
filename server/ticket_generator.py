"""
Realistic ticket generator for the Customer Support Triage RL environment.

Generates procedurally varied tickets with difficulty-appropriate content,
sentiment patterns, urgency signals, and edge cases.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

from models import TicketData


# ---------------------------------------------------------------------------
# Ground-truth labels (used internally for reward computation)
# ---------------------------------------------------------------------------

@dataclass
class TicketGroundTruth:
    """Internal ground-truth labels for reward scoring."""

    ticket_id: str
    correct_route: str
    correct_urgency: str
    correct_difficulty: str
    optimal_priority_range: tuple[float, float]  # (min, max)


# ---------------------------------------------------------------------------
# Ticket template data
# ---------------------------------------------------------------------------

_BILLING_SUBJECTS = [
    "I was charged twice for my monthly subscription",
    "My invoice shows incorrect amount please help",
    "Unable to update payment method card declined",
    "Unexpected charge on my account this month",
    "Refund request for cancelled subscription",
    "Billing cycle changed without my approval",
    "Promo discount not applied to my invoice",
    "Annual plan billed monthly by mistake",
]

_BILLING_DESCRIPTIONS = [
    (
        "I noticed a duplicate charge of $49.99 on my credit card statement dated "
        "last Tuesday. My account shows only one subscription but my bank confirms "
        "two identical charges. Please investigate and issue a refund for the "
        "duplicate charge immediately. I have attached my bank statement as proof.",
        "high",
    ),
    (
        "I updated my subscription to the annual plan last month but I am still "
        "being charged the monthly rate. This has happened for three consecutive "
        "months. I would like the billing corrected and a partial refund for the "
        "overpayments. My account email is the same as this ticket.",
        "medium",
    ),
    (
        "The 20% first-year discount that was promised during signup is not reflected "
        "in my latest invoice. The invoice shows the full price. Could you please "
        "apply the discount and update my billing going forward?",
        "low",
    ),
    (
        "My credit card was declined when I tried to upgrade. The card is valid and "
        "has sufficient funds. I have already tried three different cards with the "
        "same result. This is blocking my team from accessing premium features.",
        "high",
    ),
]

_TECHNICAL_SUBJECTS = [
    "Application crashes on startup after latest update",
    "API integration returns 500 errors consistently",
    "Cannot login two factor authentication not working",
    "Data export feature produces corrupted CSV files",
    "Performance severely degraded since last deployment",
    "Webhook notifications stopped working yesterday",
    "Mobile app freezes when uploading attachments",
    "Integration with Salesforce broken after version update",
]

_TECHNICAL_DESCRIPTIONS = [
    (
        "Since updating to version 3.2.1 yesterday the desktop application crashes "
        "immediately after the splash screen on Windows 11. I have tried reinstalling "
        "three times. The error log shows: RuntimeError: module 'ssl' has no attribute "
        "'wrap_socket'. This is critical as our entire team cannot access the platform.",
        "critical",
    ),
    (
        "Our REST API integration started returning HTTP 500 errors this morning. "
        "The same requests were working fine yesterday. I am sending POST requests "
        "to /api/v2/events with a valid API key. The response body is empty. "
        "This is affecting our production data pipeline.",
        "critical",
    ),
    (
        "The two-factor authentication code I receive via SMS is not being accepted "
        "even though it is correct and within the 30-second window. I have verified "
        "my phone number is correct. I have tried five times and am now locked out.",
        "high",
    ),
    (
        "When I export my project data to CSV the file contains garbled characters "
        "in rows containing special characters such as accented letters and emojis. "
        "The data looks correct inside the application. This affects our monthly "
        "reporting workflow.",
        "medium",
    ),
    (
        "The application has been noticeably slower since last week's deployment. "
        "Page load times have gone from under 2 seconds to over 15 seconds. "
        "This happens consistently for all users in our organisation.",
        "high",
    ),
]

_FEATURE_SUBJECTS = [
    "Request to add bulk export functionality",
    "Would love dark mode option for the dashboard",
    "Suggestion to integrate with Google Calendar",
    "Feature request keyboard shortcuts for power users",
    "Please add multi-language support for our team",
    "Request for advanced filtering in reports section",
    "Suggestion add undo functionality to editor",
    "Feature request custom notification schedules",
]

_FEATURE_DESCRIPTIONS = [
    (
        "It would be very helpful to have a bulk export option that allows selecting "
        "multiple projects and exporting them all at once. Currently I have to export "
        "each project individually which is time consuming for our workflow. "
        "We have over 50 active projects.",
        "low",
    ),
    (
        "Many of our team members work late and would appreciate a dark mode theme "
        "for the dashboard. This is a popular request among our users. "
        "Could you consider adding it in an upcoming release?",
        "low",
    ),
    (
        "Our team heavily uses Google Calendar and it would streamline our workflow "
        "greatly if tasks could be synced directly. We currently have to manually "
        "copy deadlines which leads to missed items.",
        "low",
    ),
    (
        "Power users on our team spend hours in the application daily. Adding "
        "keyboard shortcuts for common actions like creating new items, saving, "
        "and navigating between sections would significantly improve productivity.",
        "low",
    ),
]

_FEEDBACK_SUBJECTS = [
    "Positive feedback about recent customer support",
    "Great experience with the onboarding process",
    "Feedback on new dashboard design changes",
    "Comments about overall product quality",
    "Appreciation for quick resolution of my issue",
    "Suggestion based on six months of usage",
]

_FEEDBACK_DESCRIPTIONS = [
    (
        "I wanted to share that the recent support I received from your team was "
        "exceptional. The agent resolved my billing issue within hours and was "
        "very professional throughout. This kind of service is why we continue "
        "to renew our subscription every year.",
        "low",
    ),
    (
        "The new dashboard design is clean and intuitive. The redesigned navigation "
        "makes it much easier to find features. The performance improvements are "
        "also noticeable. Looking forward to the upcoming features mentioned in "
        "the roadmap.",
        "low",
    ),
    (
        "After six months of using the platform I have some observations. The core "
        "features work well but the mobile experience could be improved. The web "
        "application is excellent and our team productivity has improved by about "
        "30 percent since we started using your service.",
        "low",
    ),
]

_SPAM_SUBJECTS = [
    "Congratulations you have been selected for a prize",
    "Earn money from home guaranteed income opportunity",
    "Your account has been compromised click here now",
    "Special offer limited time free subscription upgrade",
    "Make thousands per week with this simple trick",
    "Important security alert verify your details",
]

_SPAM_DESCRIPTIONS = [
    (
        "You have been specially selected to receive a cash prize of fifty thousand "
        "dollars. To claim your prize please click the link below and provide your "
        "bank account details and social security number. This offer expires in "
        "24 hours. Act now to secure your winnings.",
        "low",
    ),
    (
        "Work from home and earn guaranteed income of five thousand dollars weekly "
        "with minimal effort. No experience required. Simply share our referral "
        "link and collect commissions. Join thousands of successful members today. "
        "Limited spots available so register immediately.",
        "low",
    ),
    (
        "URGENT: Your account security has been breached. Click this link immediately "
        "to verify your credentials and prevent permanent account suspension. "
        "Failure to act within 2 hours will result in data loss.",
        "low",
    ),
]

# ---------------------------------------------------------------------------
# Ambiguous / edge-case templates for Hard difficulty
# ---------------------------------------------------------------------------

_AMBIGUOUS_SUBJECTS = [
    "Issue with my account settings page",
    "Problem that started recently and getting worse",
    "Need help with my account urgently",
    "Something is wrong I do not know what exactly",
    "My account shows different information than expected",
    "Everything stopped working at once suddenly",
]

_AMBIGUOUS_DESCRIPTIONS = [
    (
        "I have been having some issues lately. The page sometimes does not load "
        "correctly and when it does the numbers seem off. I am not sure if this "
        "is a billing issue or a display bug. It started a few days ago and my "
        "colleagues are seeing the same thing. Our account is enterprise tier.",
        "medium",
    ),
    (
        "I received an email saying my subscription was renewed but I cancelled it "
        "last month. I cannot tell if this is an unauthorised charge or just a "
        "notification bug. Either way I am concerned. Please look into this as "
        "soon as possible.",
        "high",
    ),
    (
        "Things were working fine this morning but now several features are not "
        "responding. The dashboard loads but clicking on anything just shows a "
        "loading spinner indefinitely. I cannot tell if this is a server issue "
        "or something on my end. I need this resolved today.",
        "high",
    ),
    (
        "I have been a customer for over two years and recently the product quality "
        "has declined noticeably. Features that used to work reliably now have "
        "intermittent issues. I am reconsidering my annual renewal. I would "
        "appreciate a call from account management.",
        "medium",
    ),
]


# ---------------------------------------------------------------------------
# Category config per difficulty
# ---------------------------------------------------------------------------

_EASY_DISTRIBUTION = [
    ("billing", 0.25),
    ("technical", 0.35),
    ("feature", 0.20),
    ("feedback", 0.15),
    ("spam", 0.05),
]

_MEDIUM_DISTRIBUTION = [
    ("billing", 0.22),
    ("technical", 0.30),
    ("feature", 0.18),
    ("feedback", 0.15),
    ("spam", 0.15),
]

_HARD_DISTRIBUTION = [
    ("billing", 0.20),
    ("technical", 0.25),
    ("feature", 0.15),
    ("feedback", 0.10),
    ("spam", 0.10),
    ("ambiguous", 0.20),
]


_URGENCY_BY_CATEGORY = {
    "billing": ["low", "medium", "high"],
    "technical": ["medium", "high", "critical"],
    "feature": ["low"],
    "feedback": ["low"],
    "spam": ["low"],
    "ambiguous": ["medium", "high"],
}

_DIFFICULTY_BY_URGENCY = {
    "low": "easy",
    "medium": "medium",
    "high": "hard",
    "critical": "hard",
}

_PRIORITY_BY_URGENCY = {
    "low": (10.0, 30.0),
    "medium": (30.0, 60.0),
    "high": (60.0, 85.0),
    "critical": (85.0, 100.0),
}

_SENTIMENT_BY_DIFFICULTY = {
    "easy": (0.3, 1.0),
    "medium": (-0.3, 0.5),
    "hard": (-1.0, 0.3),
}

_ACCOUNT_AGE_BY_DIFFICULTY = {
    "easy": (30, 365),
    "medium": (30, 1095),
    "hard": (365, 2190),
}

_TICKET_HISTORY_BY_DIFFICULTY = {
    "easy": (0, 3),
    "medium": (0, 10),
    "hard": (0, 30),
}


# ---------------------------------------------------------------------------
# Ticket generator class
# ---------------------------------------------------------------------------

class TicketGenerator:
    """
    Procedurally generates realistic customer support tickets.

    Each ticket comes with ground-truth labels used for reward scoring.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    def _weighted_choice(
        self, choices: list[tuple[str, float]]
    ) -> str:
        labels, weights = zip(*choices)
        return self._rng.choices(list(labels), weights=list(weights), k=1)[0]

    def _category_pool(
        self,
        category: str,
        difficulty_level: str,
    ) -> tuple[list[tuple[str, str]], list[str], list[str]]:
        """Return (descriptions_with_urgency, subjects, wrong_categories) for a category."""
        if category == "billing":
            descs = _BILLING_DESCRIPTIONS
            subjs = _BILLING_SUBJECTS
            wrong = ["technical", "feedback"]
        elif category == "technical":
            descs = _TECHNICAL_DESCRIPTIONS
            subjs = _TECHNICAL_SUBJECTS
            wrong = ["billing", "feature"]
        elif category == "feature":
            descs = _FEATURE_DESCRIPTIONS
            subjs = _FEATURE_SUBJECTS
            wrong = ["feedback", "technical"]
        elif category == "feedback":
            descs = _FEEDBACK_DESCRIPTIONS
            subjs = _FEEDBACK_SUBJECTS
            wrong = ["feature", "billing"]
        elif category == "spam":
            descs = _SPAM_DESCRIPTIONS
            subjs = _SPAM_SUBJECTS
            wrong = ["billing", "technical"]
        else:  # ambiguous — treat as billing or technical
            descs = _AMBIGUOUS_DESCRIPTIONS
            subjs = _AMBIGUOUS_SUBJECTS
            wrong = ["billing", "technical", "feature"]
        return descs, subjs, wrong

    def generate_ticket(
        self, difficulty_level: str = "easy", episode_index: int = 0
    ) -> tuple[TicketData, TicketGroundTruth]:
        """Generate a single ticket appropriate for the given difficulty level."""

        distribution = {
            "easy": _EASY_DISTRIBUTION,
            "medium": _MEDIUM_DISTRIBUTION,
            "hard": _HARD_DISTRIBUTION,
        }[difficulty_level]

        # Pick true category
        true_category = self._weighted_choice(distribution)
        effective_category = true_category if true_category != "ambiguous" else self._rng.choice(["billing", "technical"])

        # Pick description and urgency
        descs, subjs, wrong_cats = self._category_pool(true_category, difficulty_level)
        description_text, base_urgency = self._rng.choice(descs)

        urgency_pool = _URGENCY_BY_CATEGORY.get(true_category, ["medium"])
        if difficulty_level == "easy":
            urgency_pool = [u for u in urgency_pool if u != "critical"] or urgency_pool

        # Use the template's urgency if it's allowed; otherwise fall back to random.
        if base_urgency in urgency_pool:
            true_urgency = base_urgency
        else:
            true_urgency = self._rng.choice(urgency_pool)

        # Subject selection
        subject = self._rng.choice(subjs)

        # Initial category (may be wrong for medium/hard)
        if difficulty_level == "easy":
            initial_category = effective_category
        elif difficulty_level == "medium":
            # 30% chance of wrong initial category
            if self._rng.random() < 0.30:
                initial_category = self._rng.choice(wrong_cats)
            else:
                initial_category = effective_category
        else:
            # Hard: 50% chance of wrong initial category
            if self._rng.random() < 0.50:
                initial_category = self._rng.choice(wrong_cats)
            else:
                initial_category = effective_category

        # Sentiment
        sent_range = _SENTIMENT_BY_DIFFICULTY[difficulty_level]
        sentiment = round(
            self._rng.uniform(sent_range[0], sent_range[1]), 3
        )
        # Clamp
        sentiment = max(-1.0, min(1.0, sentiment))

        # Account age
        age_range = _ACCOUNT_AGE_BY_DIFFICULTY[difficulty_level]
        account_age = self._rng.randint(age_range[0], age_range[1])

        # Previous tickets
        hist_range = _TICKET_HISTORY_BY_DIFFICULTY[difficulty_level]
        prev_tickets = self._rng.randint(hist_range[0], hist_range[1])

        word_count = len(description_text.split())
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

        ticket = TicketData(
            ticket_id=ticket_id,
            subject=subject,
            initial_category=initial_category,
            description=description_text,
            customer_sentiment=sentiment,
            word_count=word_count,
            customer_account_age=account_age,
            previous_tickets_count=prev_tickets,
        )

        # Ground truth
        resolution_difficulty = _DIFFICULTY_BY_URGENCY[true_urgency]
        priority_range = _PRIORITY_BY_URGENCY[true_urgency]

        ground_truth = TicketGroundTruth(
            ticket_id=ticket_id,
            correct_route=effective_category,
            correct_urgency=true_urgency,
            correct_difficulty=resolution_difficulty,
            optimal_priority_range=priority_range,
        )

        return ticket, ground_truth

    def generate_episode(
        self, difficulty_level: str = "easy", seed: Optional[int] = None
    ) -> list[tuple[TicketData, TicketGroundTruth]]:
        """Generate a complete episode's worth of tickets."""
        if seed is not None:
            self._rng = random.Random(seed)

        n_tickets = {"easy": 5, "medium": 10, "hard": 15}[difficulty_level]
        return [
            self.generate_ticket(difficulty_level, i) for i in range(n_tickets)
        ]
