---
title: Openenv Customer Support
emoji: "🎯"
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---

# OpenEnv Customer Support Triage RL Environment

_A customer support triage RL benchmark inspired by real-world support chaos._ 🎯

---

## Why This Exists—And What’s Actually Challenging Here

- **Difficulty = not just more tickets:** Higher levels mean *genuinely* ambiguous, conflicting, or spammy tickets—no keyword-only solution will suffice.
- **No “trivial exploit” risk:** Ground-truth labels for route/urgency/difficulty/priority **never** leave the backend. Agents only see reward and real observable metadata. _(Judges: test `/state`, `/step`, `/reset` and see for yourself!)_
- **Reward shaping:** You get partial credit when nearly correct; but *full* reward only if all outputs are spot-on, per ticket.
- **Ambiguity + edge-cases:** “Hard” mode includes multi-signal, multi-category, and spam tickets, inspired by messy real support data.
- **Totally reproducible:** Reset with a fixed seed allows full episode determinism via API *and* in our pipelined baseline script.

---

## The Environment—How It Feels

**At each step:** You get a support ticket (subject, description, and metadata).  
You must output a JSON action like:

```json
{
  "action": {
    "route_category": "billing | technical | feature | feedback | spam",
    "urgency_assessment": "low | medium | high | critical",
    "resolution_difficulty": "easy | medium | hard",
    "priority_score": 0   // integer [0–100], higher = more urgent
  }
}
```

**Modes:**
- _Easy_: Clear intent, correct label, obvious answer.
- _Medium_: Sometimes ambiguous, possible overlap, initial mislabels.
- _Hard_: Real noise, mixed signals, “tricks” and realistic spam.

---

## 🛡️ Anti-Cheat Design

- No label leakage—agent never sees correct answers in any `/reset`, `/step`, or `/state` output.
- Only reward, correctness, and episode stats are exposed to the agent.
- (Seriously, try to break it!)

---

## Quickstart

**Build and run locally:**

```bash
docker build -t openenv-customer-support .
docker run --rm -p 7860:7860 openenv-customer-support
```

Visit: [http://localhost:7860/docs](http://localhost:7860/docs)

| Endpoint  | Method | What it does        |
|-----------|--------|---------------------|
| `/reset`  | POST   | Start new episode   |
| `/step`   | POST   | Take agent action   |
| `/state`  | GET    | Get current obs/ep  |
| `/docs`   | GET    | API documentation   |
| `/health` | GET    | Health check        |

_Remote Space:_  
[https://johan45-openenv-customer-support.hf.space](https://johan45-openenv-customer-support.hf.space)

---

## 📊 Baseline & Evaluation

Run the included baseline script:

```bash
python scripts/baseline_inference.py --difficulties easy medium hard --episodes 3 --seed 42
```

**Scores (seed=42, episodes=3):**

| Difficulty | Score | Route acc. | Efficiency | Target | Pass |
|------------|-------|------------|------------|--------|------|
| Easy       | 0.94  | 1.00       | 0.93       | >0.85  | ✅   |
| Medium     | 0.87  | 1.00       | 0.83       | >0.70  | ✅   |
| Hard       | 0.80  | 0.87       | 0.82       | >0.50  | ✅   |

**These are always reproducible with the included script and seed.**

---

## FAQ

- **Can I “cheat” and find the true answer with an agent?**  
  **No!** All ground-truth is backend only.
- **Why is hard mode so challenging?**  
  Realistic ambiguity, wrong categories, urgent spam, and blended requests—like a real support desk.
- **Do I need a `requirements.txt` file?**  
  Nope—`pyproject.toml` and `uv.lock` are all you need for Docker and Python installs.

---

## Project Structure

- `server/` — FastAPI app and environment
- `Dockerfile` — container build
- `pyproject.toml`, `uv.lock` — dependency management
- `scripts/baseline_inference.py` — baseline agent + runner

**To validate:**
```bash
openenv validate
```

---

