---
title: Openenv Customer Support
emoji: 🔥
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---

# OpenEnv Customer Support Triage RL Environment

A Docker-deployed OpenEnv-compatible customer support ticket triage environment + API server.

This Space/repo is set up to:
- build cleanly with Docker
- expose a FastAPI/uvicorn server
- pass `openenv validate` for multi-mode deployment

## Environment Description & Motivation

This environment simulates a realistic customer-support triage workflow. On each step, the agent receives a customer support ticket (subject, description, and metadata) and must:

- **route** the ticket to the correct department,
- **assess urgency**,
- **predict resolution difficulty**,
- assign a **priority score (0–100)**.

Motivation: In real support systems, correct routing and prioritization reduces time-to-resolution, prevents escalations, and improves customer experience. This environment provides a structured RL-style interface (`reset`, `step`, `state`) to train and evaluate such policies.

## Task Description (Expected Difficulty)

Each episode consists of multiple support tickets. For each ticket, the agent must output:

1) `route_category` (department)
2) `urgency_assessment` (urgency level)
3) `resolution_difficulty` (difficulty estimate)
4) `priority_score` (0–100)

Difficulty is selected at reset time:
- **easy**: clearer language, fewer ambiguous cases
- **medium**: more overlap across categories and urgency
- **hard**: noisier text, more ambiguity and edge cases

## Action Space

The action is sent to **POST `/step`** as a JSON body with an `action` object:

```json
{
  "action": {
    "route_category": "billing | technical | feature | feedback | spam",
    "urgency_assessment": "low | medium | high | critical",
    "resolution_difficulty": "easy | medium | hard",
    "priority_score": 0
  }
}
```

Field meanings:
- `route_category`: routing destination (enum)
- `urgency_assessment`: urgency class (enum)
- `resolution_difficulty`: predicted resolution difficulty (enum)
- `priority_score`: numeric priority in **[0, 100]** (higher = more urgent)

## Observation Space

The current observation/state can be fetched via **GET `/state`** (and also appears in responses to `/reset` and `/step`).

It includes (high-level):
- `session_id`
- `difficulty`
- `step_count`, `episode_count`
- `done`
- `episode_stats` (running metrics and reward)
- `current_ticket` (the ticket the agent must act on), typically including:
  - `ticket_id`
  - `subject`
  - `description`
  - additional metadata such as `customer_sentiment`, `word_count`, etc.

## API Endpoints

Base URL (deployed): `https://johan45-openenv-customer-support.hf.space`

System:
- `GET /health` — health check for Docker/load balancers
- `GET /` — root
- `GET /web` — web interface (if enabled)

Environment:
- `POST /reset` — start a new episode (difficulty/seed)
- `POST /step` — take an action for the current ticket
- `GET /state` — get current state (includes `current_ticket`)

Interactive docs:
- `GET /docs`

## Setup & Usage Instructions

### Build (local)
```bash
docker build -t openenv-customer-support .
```

### Run (local)
```bash
docker run --rm -p 7860:7860 openenv-customer-support
```

Open:
- http://localhost:7860/docs

### Example: reset
```bash
curl -X POST "https://johan45-openenv-customer-support.hf.space/reset" \
  -H "Content-Type: application/json" \
  -d '{"difficulty":"easy","seed":0}'
```

### Example: step
```bash
curl -X POST "https://johan45-openenv-customer-support.hf.space/step" \
  -H "Content-Type: application/json" \
  -d '{"action":{"route_category":"technical","urgency_assessment":"high","resolution_difficulty":"medium","priority_score":80}}'
```

### Example: state
```bash
curl "https://johan45-openenv-customer-support.hf.space/state"
```

## Baselines (Scores)

The environment returns performance in `episode_stats` (see `GET /state`), including `total_reward`.

Suggested baselines:
1) **Constant baseline**
   - Always output:
     - `route_category="billing"`
     - `urgency_assessment="low"`
     - `resolution_difficulty="easy"`
     - `priority_score=50`
2) **Random baseline**
   - Sample each enum uniformly, and sample `priority_score` uniformly from [0, 100]
3) **Simple keyword heuristic**
   - Route using keywords (e.g., “refund/invoice” → billing, “error/crash/login” → technical)
   - Map urgency keywords (“urgent/asap/outage”) → high/critical
   - Use higher `priority_score` for high/critical

How to record baseline scores (recommended/reproducible):
1) `POST /reset` with fixed seed (example `{"difficulty":"easy","seed":0}`)
2) Loop tickets: for each ticket call `POST /step` with your baseline policy
3) When episode ends, read `episode_stats.total_reward` from `GET /state`

> Note: Exact numeric baseline reward depends on ticket stream and difficulty. Use a fixed `seed` to make runs reproducible.

## What was updated to make everything work

### 1) Docker build fixes
The Docker image build was stabilized by:
- copying the full project into the image before running `pip install -e .`
- disabling pip’s progress bar inside the container to avoid thread-related build failures on some systems
- simplifying the install step so the build is repeatable

### 2) OpenEnv validation requirements
To satisfy OpenEnv validation checks:
- added the required dependency:
  - `openenv-core>=0.2.0`
- added the required script entry point:
  - `[project.scripts] server = "server.app:main"`
- regenerated `uv.lock` so it matches `pyproject.toml`

## Validation

Run OpenEnv validation from the repository root:
```bash
openenv validate
```

If your repository includes the validator script, you can also run:
```bash
./scripts/validate-submission.sh https://johan45-openenv-customer-support.hf.space .
```

## Project layout (high level)

- `server/` — FastAPI application and environment entry points
- `Dockerfile` — container build used for deployment
- `pyproject.toml` — Python package metadata + dependencies
- `uv.lock` — locked dependency set used by validation

## Do we need `requirements.txt`?

No. This project is deployed via **Docker** and installs dependencies from `pyproject.toml` (with `uv.lock` kept in sync).  
Add a `requirements.txt` only if some external tool explicitly requires it.