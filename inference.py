import os
import requests
from openai import OpenAI

# Config
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.environ.get("API_KEY", "dummy-key")
SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")
BENCHMARK = "codefixenv"

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

SYSTEM_PROMPT = """You are an expert Python debugger. You will be given buggy Python code and a failing test.
Your job is to return ONLY the fixed Python function — no explanation, no markdown, no backticks.
Just the raw corrected Python code that will make the test pass."""


def call_llm(buggy_code: str, test_cases: str, error_message: str | None) -> str:
    """Ask the LLM to fix the code, protected by try/except."""
    error_context = f"\nLast error:\n{error_message}" if error_message else ""
    user_prompt = f"Fix this Python code so the following tests pass.\n\nBuggy code:\n{buggy_code}\n\nTests that must pass:\n{test_cases}{error_context}\n\nReturn ONLY the fixed function code. No explanation. No markdown."

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=512,
            temperature=0.1,
            timeout=20.0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Prevent crash if LLM API fails, just return original code
        return buggy_code


def run_task(task_id: int):
    """Run one full task episode."""
    task_names = ["", "easy", "medium", "hard"]
    task_name = task_names[task_id]

    # Text-based START log with flush=True
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    step_count = 0
    rewards = []
    score = 0.0

    try:
        # Reset environment for this task
        reset_resp = requests.post(f"{SPACE_URL}/reset", params={"task_id": task_id}, timeout=10.0)
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        done = False

        while not done and step_count < 5:
            step_count += 1

            # LLM generates fixed code
            fixed_code = call_llm(
                buggy_code=obs.get("buggy_code", ""),
                test_cases=obs.get("test_cases", ""),
                error_message=obs.get("error_message")
            )

            # Submit to environment
            step_resp = requests.post(
                f"{SPACE_URL}/step",
                json={"fixed_code": fixed_code},
                timeout=10.0
            )
            step_resp.raise_for_status()
            result = step_resp.json()

            obs = result["observation"]
            reward = float(result["reward"])
            done = result["done"]
            rewards.append(reward)

            # Text-based STEP log with flush=True.
            # Note: Deliberately omitted 'action=code' to prevent newlines from breaking the parsing
            print(f"[STEP] step={step_count} reward={reward:.2f} done={str(done).lower()}", flush=True)

        score = rewards[-1] if rewards else 0.0

        # Text-based END log with flush=True
        print(f"[END] task={task_name} score={score:.2f} steps={step_count}", flush=True)

    except Exception as e:
        # Text-based fallback error log
        print(f"[END] task={task_name} score=0.00 steps={step_count} error=\"{str(e)}\"", flush=True)


def main():
    # Loop through Easy, Medium, Hard
    for task_id in [1, 2, 3]:
        run_task(task_id)


if __name__ == "__main__":
    main()