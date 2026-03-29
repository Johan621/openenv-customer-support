---
title: Customer Support Triage
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
---

# 🎯 Customer Support Ticket Triage RL Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Hackathon%20Round%201-7c3aed)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)

A **production-ready, real-world Reinforcement Learning environment** for the
Meta-PyTorch-HuggingFace OpenEnv Hackathon Round 1. The environment simulates
customer support ticket triage operations where AI agents learn to efficiently
**classify**, **prioritize**, and **route** support tickets.

---

## 🌐 Real-World Scenario

A SaaS customer support platform receives hundreds of diverse tickets daily.
The agent must efficiently triage incoming tickets by:

- 📖 Reading ticket metadata (subject, category, sentiment, description)
- 🚨 Assessing the true urgency level
- 🔀 Selecting the optimal routing destination
- 📊 Predicting resolution difficulty
- 🔢 Assigning a numerical priority score

This is a **real-world, non-trivial** task that directly maps to operational
challenges faced by support teams worldwide.

---

## 📦 Project Structure

```
openenv-customer-support/
├── __init__.py                     # Module exports
├── README.md                       # This file
├── openenv.yaml                    # OpenEnv manifest
├── pyproject.toml                  # Project metadata and dependencies
├── LICENSE                         # BSD 3-Clause
├── .gitignore
├── Dockerfile                      # Container definition
├── .dockerignore
├── models.py                       # Pydantic models (TriageAction, TriageObservation)
├── client.py                       # HTTP + WebSocket client
├── server/
│   ├── __init__.py
│   ├── app.py                      # FastAPI application
│   ├── customer_support_env.py     # Core environment logic
│   └── ticket_generator.py         # Realistic ticket generation
├── scripts/
│   ├── __init__.py
│   └── baseline_inference.py       # Baseline heuristic agent
├── data/
│   └── sample_tickets.json         # Example tickets
└── tests/
    └── test_environment.py         # Test suite
```

---

## 🎮 Difficulty Levels

| Level  | Tickets/Episode | Ambiguity | Sentiment | Target Score |
|--------|----------------|-----------|-----------|-------------|
| Easy   | 5              | Low       | Positive  | **> 0.85**  |
| Medium | 10             | Moderate  | Mixed     | **> 0.70**  |
| Hard   | 15             | High      | Negative  | **> 0.50**  |

### Easy
- Clear ticket categories with minimal ambiguity
- Positive sentiment bias
- New accounts (simple ticket history)
- Success: All tickets correctly routed with urgency assessment

### Medium
- Mixed categories with ~30% ambiguous initial categories
- Neutral sentiment mix, varied account ages
- Success: ≥ 80% correct routing, reasonable urgency assessment

### Hard
- Ambiguous categories and edge cases, ~50% wrong initial labels
- Complex descriptions requiring inference
- Mixed/negative sentiment, old accounts with ticket history
- Rare categories (spam, escalations)
- Success: ≥ 60% correct routing, handle edge cases appropriately

---

## 🔧 Action Space (`TriageAction`)

The agent outputs a `TriageAction` Pydantic model for each ticket:

```python
class TriageAction(BaseModel):
    route_category: Literal["billing", "technical", "feature", "feedback", "spam"]
    urgency_assessment: Literal["low", "medium", "high", "critical"]
    resolution_difficulty: Literal["easy", "medium", "hard"]
    priority_score: float  # 0.0 - 100.0
```

**Example:**
```json
{
  "route_category": "billing",
  "urgency_assessment": "high",
  "resolution_difficulty": "hard",
  "priority_score": 75.0
}
```

---

## 👀 Observation Space (`TriageObservation`)

The environment returns a `TriageObservation` Pydantic model after each step:

```python
class TriageObservation(BaseModel):
    ticket_info: Optional[TicketData]   # Next ticket (None when done)
    correctness_score: float             # 0-1, triage correctness
    efficiency_score: float              # 0-1, routing efficiency
    task_progress: float                 # 0-1, episode completion
    difficulty_level: str                # "easy" | "medium" | "hard"
    episode_stats: EpisodeStats          # Aggregate statistics
    done: bool                           # Episode termination flag
    reward: float                        # Step reward (partial progress)
    metadata: dict                       # Step count, episode info, etc.
```

### Ticket Data Structure (`TicketData`)

| Field                    | Type    | Description                              |
|--------------------------|---------|------------------------------------------|
| `ticket_id`              | str     | Unique identifier (e.g., `TKT-A1B2C3D4`)|
| `subject`                | str     | Brief subject (5-15 words)               |
| `initial_category`       | str     | Customer's suggested category (may be wrong) |
| `description`            | str     | Full description (20-200 words)          |
| `customer_sentiment`     | float   | Sentiment score (-1.0 to 1.0)            |
| `word_count`             | int     | Word count of description                |
| `customer_account_age`   | int     | Days since account creation              |
| `previous_tickets_count` | int     | Number of prior support tickets          |

---

## 💰 Reward Function

Dense reward structure with partial progress signals:

```
Per-ticket rewards:
  +0.50  Correct route_category
  +0.25  Correct urgency_assessment
  +0.10  Correct resolution_difficulty
  +0.05  priority_score within optimal range

Partial credit:
  +0.10  urgency off-by-one (e.g., medium when high is correct)

Penalties:
  -0.20  Wrong route_category
  -0.10  Wildly wrong routing (e.g., spam → billing)

Episode bonus:
  +0.40  All tickets correctly triaged (route + urgency + difficulty)
```

---

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.10
pip install -r requirements (see pyproject.toml)
```

### Install Dependencies

```bash
pip install "fastapi>=0.111.0" "uvicorn[standard]>=0.29.0" "pydantic>=2.7.0" \
            "websockets>=12.0" "httpx>=0.27.0" "numpy>=1.26.0" \
            "scikit-learn>=1.4.0" "python-dotenv>=1.0.0"
```

### Run the Server

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Interact with the Environment

```python
from server.customer_support_env import CustomerSupportEnv
from models import TriageAction

env = CustomerSupportEnv()

# Start easy episode
obs = env.reset("easy", seed=42)
print("First ticket:", obs.ticket_info.subject)

# Take actions until episode ends
while not obs.done:
    action = TriageAction(
        route_category="technical",
        urgency_assessment="high",
        resolution_difficulty="hard",
        priority_score=75.0,
    )
    obs = env.step(action)
    print(f"Reward: {obs.reward:.3f} | Progress: {obs.task_progress:.0%}")

print("Episode done! Total reward:", obs.episode_stats.total_reward)
```

### HTTP Client

```python
from client import CustomerSupportEnvClient
from models import TriageAction

with CustomerSupportEnvClient("http://localhost:8000") as client:
    obs = client.reset("medium", seed=42)
    while not obs.done:
        action = TriageAction(
            route_category=obs.ticket_info.initial_category,
            urgency_assessment="medium",
            resolution_difficulty="medium",
            priority_score=50.0,
        )
        obs = client.step(action)
```

### REST API

```bash
# Reset episode
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"difficulty": "easy", "seed": 42}'

# Step with action
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": {"route_category": "billing", "urgency_assessment": "high", "resolution_difficulty": "hard", "priority_score": 75.0}}'

# Get state
curl http://localhost:8000/state

# Health check
curl http://localhost:8000/health
```

---

## 📊 Baseline Inference

Run the heuristic baseline agent against all difficulty levels:

```bash
python scripts/baseline_inference.py
```

With options:
```bash
python scripts/baseline_inference.py \
    --difficulties easy medium hard \
    --episodes 5 \
    --seed 42 \
    --output baseline_scores.json \
    --verbose
```

### Expected Baseline Scores

The heuristic agent uses keyword matching + sentiment analysis:

| Difficulty | Target | Typical Baseline |
|------------|--------|-----------------|
| Easy       | > 0.85 | ~0.75-0.85      |
| Medium     | > 0.70 | ~0.60-0.75      |
| Hard       | > 0.50 | ~0.45-0.60      |

### Output Format (`baseline_scores.json`)

```json
{
  "easy": {
    "difficulty": "easy",
    "episodes": 5,
    "seed": 42,
    "avg_reward": 2.15,
    "avg_correctness_score": 0.82,
    "avg_efficiency_score": 0.79,
    "avg_route_accuracy": 0.80,
    "target_score": 0.85,
    "achieved_target": false,
    "score": 0.82
  }
}
```

---

## 🐳 Docker Deployment

### Build

```bash
docker build -t customer-support-env:latest .
```

### Run

```bash
docker run -p 8000:8000 customer-support-env:latest
```

### Verify

```bash
curl http://localhost:8000/health
# → {"status":"ok","environment":"customer-support-triage","version":"0.1.0"}

# Web UI
open http://localhost:8000/web

# API Docs
open http://localhost:8000/docs
```

---

## 🤗 Hugging Face Spaces Deployment

1. Create a new Space at https://huggingface.co/new-space
2. Select **Docker** as the SDK
3. Add `openenv` topic to the Space settings
4. Push code:

```bash
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/customer-support-triage
git push hf main
```

The environment will be available at:
- Web UI: `https://YOUR_USERNAME-customer-support-triage.hf.space/web`
- API: `https://YOUR_USERNAME-customer-support-triage.hf.space/docs`

---

## 🔌 WebSocket Interface

For low-latency agent interaction:

```python
import asyncio
import json
import websockets

async def run_episode():
    async with websockets.connect("ws://localhost:8000/ws/my-session") as ws:
        # Reset
        await ws.send(json.dumps({"type": "reset", "difficulty": "easy", "seed": 42}))
        obs = json.loads(await ws.recv())

        # Step
        action = {
            "type": "step",
            "action": {
                "route_category": "technical",
                "urgency_assessment": "high",
                "resolution_difficulty": "hard",
                "priority_score": 75.0,
            }
        }
        await ws.send(json.dumps(action))
        obs = json.loads(await ws.recv())
        print("Reward:", obs["reward"])

asyncio.run(run_episode())
```

---

## 🧪 Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/ -v
```

---

## 📋 OpenEnv Spec Compliance

| Requirement                          | Status |
|--------------------------------------|--------|
| Typed Pydantic models (Action/Obs)   | ✅     |
| `step()`, `reset()`, `state()` API   | ✅     |
| HTTP endpoints                       | ✅     |
| WebSocket support                    | ✅     |
| `openenv.yaml` manifest              | ✅     |
| 3 difficulty levels with graders     | ✅     |
| Dense reward with partial signals    | ✅     |
| Baseline inference script            | ✅     |
| Reproducible baseline scores         | ✅     |
| Working Dockerfile                   | ✅     |
| Web UI at `/web`                     | ✅     |
| API docs at `/docs`                  | ✅     |
| Hugging Face Spaces ready            | ✅     |

---

## 📜 License

BSD 3-Clause License — see [LICENSE](LICENSE) for details.

---

## 🏆 Hackathon

Built for the **Meta-PyTorch-HuggingFace OpenEnv Hackathon Round 1**.

- Environment type: Real-world text classification + routing
- Task: Customer support ticket triage
- Framework: OpenEnv (FastAPI + Pydantic + WebSocket)