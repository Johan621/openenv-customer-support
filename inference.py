from __future__ import annotations
# Ensure stdout is UTF-8 (prevents Windows encoding crashes; safe on Linux too)
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass


import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from client import CustomerSupportEnvClient
from models import TicketData, TriageAction

# -----------------------------------------------------------------------------
# Required env vars (with defaults where required)
# -----------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN (or API_KEY) environment variable is required")

ENV_BASE_URL = os.getenv(
    "ENV_BASE_URL", "https://johan45-openenv-customer-support.hf.space"
).rstrip("/")

DIFFICULTIES = ["easy", "medium", "hard"]
SEED = int(os.getenv("SEED", "42"))
MAX_STEPS_GUARD = int(os.getenv("MAX_STEPS_GUARD", "200"))

# -----------------------------------------------------------------------------
# Printing rules: rewards MUST be formatted to 2 decimals AND strictly between 0 and 1
# Therefore printed rewards must be clamped so they round to 0.01..0.99 (never 0.00/1.00)
# -----------------------------------------------------------------------------
PRINT_MIN = 0.01
PRINT_MAX = 0.99
EPS = PRINT_MIN

def clamp_print_score(x: float) -> float:
    try:
        x = float(x)
    except Exception:
        return EPS
    # handle nan/inf
    if x != x or x == float("inf") or x == float("-inf"):
        return EPS
    if x < PRINT_MIN:
        return PRINT_MIN
    if x > PRINT_MAX:
        return PRINT_MAX
    return x


# -----------------------------------------------------------------------------
# STRICT stdout logs (exact spec)
# -----------------------------------------------------------------------------
def _bool(v: bool) -> str:
    return "true" if v else "false"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    if error is None:
        err = "null"
    else:
        # Avoid any stdout encoding issues; keep content but escape non-encodable chars
        err = str(error).encode("utf-8", "backslashreplace").decode("utf-8")
    r = clamp_print_score(reward)
    # MUST be 2 decimals per hackathon spec
    print(
        f"[STEP] step={step} action={action} reward={r:.2f} done={_bool(done)} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    # MUST be 2 decimals per hackathon spec
    rewards_str = ",".join(f"{clamp_print_score(r):.2f}" for r in rewards)
    print(f"[END] success={_bool(success)} steps={steps} rewards={rewards_str}", flush=True)


# -----------------------------------------------------------------------------
# LLM action selection (high score)
# -----------------------------------------------------------------------------
VALID_ROUTES = {"billing", "technical", "feature", "feedback", "spam"}
VALID_URGENCY = {"low", "medium", "high", "critical"}
VALID_DIFFICULTY = {"easy", "medium", "hard"}


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def heuristic_fallback(ticket: TicketData) -> TriageAction:
    text = f"{ticket.subject} {ticket.description}".lower()
    if any(k in text for k in ["refund", "invoice", "charged", "billing", "payment", "subscription"]):
        route = "billing"
    elif any(k in text for k in ["error", "bug", "crash", "login", "timeout", "500", "404", "broken"]):
        route = "technical"
    elif any(k in text for k in ["feature", "request", "suggestion", "would love", "please add"]):
        route = "feature"
    elif any(k in text for k in ["thank", "great", "excellent", "love", "amazing"]):
        route = "feedback"
    else:
        route = "technical"

    urgency = "high" if any(
        k in text for k in ["critical", "down", "outage", "urgent", "asap", "immediately", "blocker"]
    ) else "low"
    resolution_difficulty = "hard" if urgency in ("high", "critical") else "easy"
    priority_score = 80.0 if urgency in ("high", "critical") else 25.0

    return TriageAction(
        route_category=route,  # type: ignore[arg-type]
        urgency_assessment=urgency,  # type: ignore[arg-type]
        resolution_difficulty=resolution_difficulty,  # type: ignore[arg-type]
        priority_score=priority_score,
    )


def llm_choose_action(client: OpenAI, ticket: TicketData, difficulty: str) -> TriageAction:
    prompt = f"""
Return ONLY JSON with keys:
route_category, urgency_assessment, resolution_difficulty, priority_score

Allowed:
route_category: billing|technical|feature|feedback|spam
urgency_assessment: low|medium|high|critical
resolution_difficulty: easy|medium|hard
priority_score: 0..100

Ticket:
subject: {ticket.subject}
description: {ticket.description}
initial_category: {ticket.initial_category}
customer_sentiment: {ticket.customer_sentiment}
previous_tickets_count: {ticket.previous_tickets_count}
word_count: {ticket.word_count}
difficulty_level: {difficulty}

ONLY JSON. No extra text.
""".strip()

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=220,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception:
        return heuristic_fallback(ticket)

    data = _extract_json(text)
    if not isinstance(data, dict):
        return heuristic_fallback(ticket)

    route = str(data.get("route_category", "")).strip().lower()
    urg = str(data.get("urgency_assessment", "")).strip().lower()
    diff = str(data.get("resolution_difficulty", "")).strip().lower()
    try:
        pr = float(data.get("priority_score", 50.0))
    except Exception:
        pr = 50.0

    fb = heuristic_fallback(ticket)
    if route not in VALID_ROUTES:
        route = str(fb.route_category)
    if urg not in VALID_URGENCY:
        urg = str(fb.urgency_assessment)
    if diff not in VALID_DIFFICULTY:
        diff = str(fb.resolution_difficulty)

    pr = _clamp(pr, 0.0, 100.0)

    return TriageAction(
        route_category=route,  # type: ignore[arg-type]
        urgency_assessment=urg,  # type: ignore[arg-type]
        resolution_difficulty=diff,  # type: ignore[arg-type]
        priority_score=pr,
    )


def run_one(env_client: CustomerSupportEnvClient, llm: OpenAI, difficulty: str, seed: int) -> None:
    task = f"customer_support_triage::{difficulty}"
    env_name = "openenv-customer-support"

    rewards: List[float] = []
    steps_taken = 0
    last_error: Optional[str] = None

    log_start(task=task, env=env_name, model=MODEL_NAME)
    obs = env_client.reset(difficulty=difficulty, seed=seed)

    try:
        for step in range(1, MAX_STEPS_GUARD + 1):
            if obs.done:
                break
            if obs.ticket_info is None:
                last_error = "ticket_info_missing"
                break

            action_obj = llm_choose_action(llm, obs.ticket_info, difficulty)
            obs = env_client.step(action_obj)

            raw_reward = float(obs.reward) if obs.reward is not None else EPS
            reward = clamp_print_score(raw_reward)   # <-- NEW
            rewards.append(reward)     
            steps_taken = step

            action_str = json.dumps(action_obj.model_dump(), ensure_ascii=True, separators=(",", ":"))
            log_step(step=step, action=action_str, reward=reward, done=bool(obs.done), error=last_error)

            if obs.done:
                break

    except Exception as exc:
        last_error = f"{type(exc).__name__}:{exc}"
        # Always emit a STEP line even on exception; never print 0.00
        log_step(step=max(steps_taken, 1), action="exception", reward=EPS, done=True, error=last_error)

    # Determine success from episode stats (not part of strict score constraint)
    stats = obs.episode_stats
    avg_correctness = float(stats.avg_correctness)
    target = {"easy": 0.85, "medium": 0.70, "hard": 0.50}[difficulty]
    success = avg_correctness >= target

    log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN, timeout=30)
    env_client = CustomerSupportEnvClient(base_url=ENV_BASE_URL)
    for d in DIFFICULTIES:
        run_one(env_client, llm, d, SEED)


if __name__ == "__main__":
    main()