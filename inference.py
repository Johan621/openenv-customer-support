"""
Customer Support Triage RL - Inference Script

MANDATORY ENV VARS (per hackathon validator):
- API_BASE_URL : OpenAI-compatible endpoint (e.g. https://api-inference.huggingface.co/openai/v1)
- MODEL_NAME   : Model identifier
- HF_TOKEN     : API key (also accepts API_KEY for compatibility)
"""

import os
import json
import time
import requests

from openai import OpenAI

# ----------------------------
# Mandatory env vars
# ----------------------------
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not API_BASE_URL:
    # safe default; validator may provide its own
    API_BASE_URL = "https://api-inference.huggingface.co/openai/v1"
    print(f"⚠️  API_BASE_URL not set. Defaulting to {API_BASE_URL}")

if not MODEL_NAME:
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"
    print(f"⚠️  MODEL_NAME not set. Defaulting to {MODEL_NAME}")

if not API_KEY:
    print("❌ HF_TOKEN (or API_KEY) is not set. Cannot run inference.")
    raise SystemExit(2)

# ----------------------------
# Environment base URL
# ----------------------------
# Not listed as mandatory in the portal, so we keep a default.
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://johan45-openenv-customer-support.hf.space").rstrip("/")

# Runtime controls (keep small for <20 min)
MAX_STEPS_PER_EPISODE = int(os.getenv("MAX_STEPS_PER_EPISODE", "15"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "150"))
DIFFICULTIES = ["easy", "medium", "hard"]

print("🚀 Inference Configuration:")
print(f"  - API_BASE_URL: {API_BASE_URL}")
print(f"  - MODEL_NAME:   {MODEL_NAME}")
print(f"  - ENV_BASE_URL: {ENV_BASE_URL}")
print(f"  - API_KEY:      {'✅ Set' if API_KEY else '❌ Missing'}")

SYSTEM_PROMPT = """You are an expert customer support triage agent.
Analyze the support ticket and decide:
1. route_category: One of [billing, technical, feature, feedback, spam]
2. urgency_assessment: One of [low, medium, high, critical]
3. resolution_difficulty: One of [easy, medium, hard]
4. priority_score: A number between 0-100

Respond in JSON format only:
{"route_category": "...", "urgency_assessment": "...", "resolution_difficulty": "...", "priority_score": 50}
"""

FALLBACK_ACTION = {
    "route_category": "technical",
    "urgency_assessment": "medium",
    "resolution_difficulty": "medium",
    "priority_score": 50,
}


def call_env_endpoint(endpoint, method="GET", data=None):
    url = f"{ENV_BASE_URL}{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data, timeout=15)
        else:
            response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error calling {endpoint}: {e}")
        return None


def generate_triage_action(observation, client: OpenAI):
    ticket = (observation or {}).get("ticket_info") or {}
    ticket_desc = f"Ticket: {ticket.get('subject','N/A')}\nBody: {ticket.get('description','N/A')}\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Triage:\n{ticket_desc}"},
    ]

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = completion.choices[0].message.content or ""
        try:
            action = json.loads(response_text)
            # minimal sanity
            if not isinstance(action, dict):
                return FALLBACK_ACTION
            action.setdefault("priority_score", 50)
            return action
        except Exception:
            return FALLBACK_ACTION
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return FALLBACK_ACTION


def run_episode(difficulty, client: OpenAI):
    print(f"\n🎮 Running episode: {difficulty.upper()}")

    reset_response = call_env_endpoint("/reset", "POST", {"difficulty": difficulty, "seed": 42})
    if not reset_response:
        return {"difficulty": difficulty, "success": False, "score": 0.0, "total_reward": 0.0}

    observation = reset_response
    total_reward = 0.0
    steps_taken = 0

    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        steps_taken = step
        action = generate_triage_action(observation, client)

        step_response = call_env_endpoint("/step", "POST", {"action": action})
        if not step_response:
            break

        observation = step_response
        reward = float(observation.get("reward", 0.0) or 0.0)
        total_reward += reward

        if observation.get("done"):
            break

        time.sleep(0.2)

    # score must be in 0..1 range
    score = max(0.0, min(1.0, float(total_reward)))

    return {
        "difficulty": difficulty,
        "success": True,
        "score": score,
        "total_reward": float(total_reward),
        "steps_taken": steps_taken,
    }


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    results = []
    for difficulty in DIFFICULTIES:
        results.append(run_episode(difficulty, client))
        time.sleep(0.5)

    # Always write outputs (validator-friendly)
    with open("inference_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Also write a simple scores.json (extra safe)
    scores = {r["difficulty"]: float(r.get("score", 0.0)) for r in results}
    with open("scores.json", "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2)

    print("\n✅ Saved inference_results.json and scores.json")
    print("Scores:", scores)


if __name__ == "__main__":
    main()