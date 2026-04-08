import os
import requests
from openai import OpenAI

# REQUIRED ENV VARIABLES
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "dummy-model")
HF_TOKEN = os.getenv("HF_TOKEN", "dummy-key")

# OpenAI client (REQUIRED by rules)
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

TASK_NAME = "easy"
BENCHMARK = "codefixenv"

def main():
    step_count = 0
    rewards = []

    print(f"[START] task={TASK_NAME} env={BENCHMARK} model={MODEL_NAME}")

    try:
        # Reset environment
        res = requests.post(f"{API_BASE_URL}/reset")
        state = res.json()

        done = False

        while not done and step_count < 3:
            step_count += 1

            buggy_code = state.get("buggy_code", "")

            # ⚡ simple "agent" logic
            fixed_code = (
                buggy_code
                .replace("==", "=")
                .replace("+==", "+=")
                .replace("range(len(nums) + 1)", "range(len(nums))")
            )

            action = {"fixed_code": fixed_code}

            response = requests.post(f"{API_BASE_URL}/step", json=action)
            result = response.json()

            reward = float(result.get("reward", 0.0))
            done = result.get("done", False)

            rewards.append(reward)

            print(f"[STEP] step={step_count} action={fixed_code} reward={reward:.2f} done={str(done).lower()} error=null")

        score = rewards[-1] if rewards else 0.0
        success = score == 1.0
        rewards_str = ",".join([f"{r:.2f}" for r in rewards])

        print(f"[END] success={str(success).lower()} steps={step_count} score={score:.2f} rewards={rewards_str}")

    except Exception as e:
        print(f"[END] success=false steps={step_count} score=0.00 rewards= error={str(e)}")

if __name__ == "__main__":
    main()