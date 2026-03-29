"""
Customer Support Triage RL - Inference Script
"""

import os
import json
import time
import requests
from openai import OpenAI

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-chat-hf")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://johan45-openenv-customer-support.hf.space")

MAX_STEPS_PER_EPISODE = 15
TEMPERATURE = 0.7
MAX_TOKENS = 150
DIFFICULTIES = ["easy", "medium", "hard"]

print(f"🚀 Inference Configuration:")
print(f"  - HF Token: {'✅ Set' if HF_TOKEN else '❌ Missing'}")
print(f"  - Model: {MODEL_NAME}")
print(f"  - Environment URL: {ENV_BASE_URL}")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """You are an expert customer support triage agent.
Analyze the support ticket and decide:
1. route_category: One of [billing, technical, feature, feedback, spam]
2. urgency: One of [low, medium, high, critical]
3. priority: An integer between 1-100

Respond in JSON format only:
{"route_category": "...", "urgency": "...", "priority": <number>}"""


def call_env_endpoint(endpoint, method="GET", data=None):
    """Call the environment API endpoint."""
    url = f"{ENV_BASE_URL}{endpoint}"
    try:
        if method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error calling {endpoint}: {e}")
        return {}


def generate_triage_action(observation):
    """Use LLM to generate triage action."""
    ticket = observation.get("ticket", {})
    ticket_desc = f"""
Ticket: {ticket.get('subject', 'N/A')}
Body: {ticket.get('body', 'N/A')}
"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Triage: {ticket_desc}"}
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
        except:
            action = {"route_category": "technical", "urgency": "medium", "priority": 50}
        
        return action
    except Exception as e:
        print(f"❌ LLM Error: {e}")
        return {"route_category": "technical", "urgency": "medium", "priority": 50}


def run_episode(difficulty):
    """Run a single episode."""
    print(f"\n🎮 Running episode: {difficulty.upper()}")
    
    reset_response = call_env_endpoint("/reset", "POST", {"difficulty": difficulty})
    if not reset_response:
        print(f"❌ Failed to reset")
        return {"difficulty": difficulty, "success": False, "score": 0.0}
    
    observation = reset_response
    total_reward = 0.0
    
    for step in range(1, MAX_STEPS_PER_EPISODE + 1):
        print(f"  Step {step}")
        
        action = generate_triage_action(observation)
        print(f"    Action: {action}")
        
        step_response = call_env_endpoint("/step", "POST", {"action": action})
        if not step_response:
            break
        
        observation = step_response
        reward = observation.get("reward", 0.0)
        total_reward += reward
        
        print(f"    Reward: +{reward:.3f}")
        
        if observation.get("done"):
            print(f"    ✅ Done!")
            break
        
        time.sleep(0.5)
    
    score = min(1.0, max(0.0, total_reward))
    
    print(f"  📊 Score: {score:.3f}")
    
    return {
        "difficulty": difficulty,
        "success": True,
        "score": score,
        "total_reward": total_reward
    }


def main():
    print("\n" + "="*60)
    print("🚀 CUSTOMER SUPPORT TRIAGE - INFERENCE")
    print("="*60)
    
    results = []
    for difficulty in DIFFICULTIES:
        result = run_episode(difficulty)
        results.append(result)
        time.sleep(1)
    
    print("\n" + "="*60)
    print("📈 RESULTS")
    print("="*60)
    
    total_score = 0.0
    for result in results:
        if result["success"]:
            print(f"  {result['difficulty'].upper():10} | Score: {result['score']:.3f} ✅")
            total_score += result['score']
    
    avg_score = total_score / len([r for r in results if r["success"]])
    print(f"\n  Average Score: {avg_score:.3f}")
    
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n✅ Results saved!")


if __name__ == "__main__":
    main()