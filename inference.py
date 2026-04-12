import os
import json
import requests
from openai import OpenAI

# Config
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN = os.environ.get("HF_TOKEN", "dummy-key")
SPACE_URL = os.environ.get("SPACE_URL", "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key"
)

SYSTEM_PROMPT = """You are an expert Python debugger. You will be given buggy Python code and a failing test.
Your job is to return ONLY the fixed Python function — no explanation, no markdown, no backticks.
Just the raw corrected Python code that will make the test pass."""


def call_llm(buggy_code: str, test_cases: str, error_message: str | None) -> str:
    """Ask the LLM to fix the code, protected by try/except."""
    error_context = f"\nLast error:\n{error_message}" if error_message else ""

    user_prompt = f"""Fix this Python code so the following tests pass.

Buggy code:
{buggy_code}

Tests that must pass:
{test_cases}{error_context}

Return ONLY the fixed function code. No explanation. No markdown."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=512,
            temperature=0.1,
            timeout=20.0  # Added timeout so it doesn't hang the evaluator
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Prevent full crash if LLM API is blocked/down
        print(json.dumps({"type": "[LOG]", "message": f"LLM Call Failed: {str(e)}"}))
        return buggy_code  # Return original code to fail this step gracefully


def run_task(task_id: int) -> dict:
    """Run one full task episode with safety limits."""
    step_num = 0
    try:
        # Reset environment for this task
        reset_resp = requests.post(f"{SPACE_URL}/reset", params={"task_id": task_id}, timeout=10.0)
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        print(json.dumps({
            "type": "[START]",
            "task_id": task_id,
            "difficulty": ["", "easy", "medium", "hard"][task_id],
            "buggy_code": obs["buggy_code"]
        }))

        reward = 0.0
        done = False

        # Added step limit (5) to ensure it breaks out correctly
        while not done and step_num < 5:
            step_num += 1

            # LLM generates fixed code
            fixed_code = call_llm(
                buggy_code=obs["buggy_code"],
                test_cases=obs["test_cases"],
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
            reward = result["reward"]
            done = result["done"]

            print(json.dumps({
                "type": "[STEP]",
                "task_id": task_id,
                "step": step_num,
                "reward": reward,
                "done": done,
                "error": obs.get("error_message"),
                "fixed_code_preview": fixed_code[:100]
            }))

        print(json.dumps({
            "type": "[END]",
            "task_id": task_id,
            "final_reward": reward,
            "total_steps": step_num,
            "solved": reward == 1.0
        }))

        return {"task_id": task_id, "reward": reward, "steps": step_num}

    except Exception as e:
        # If anything else fails, print the END tag so the grader parser doesn't break
        print(json.dumps({
            "type": "[END]",
            "task_id": task_id,
            "final_reward": 0.0,
            "total_steps": step_num,
            "solved": False,
            "error": str(e)
        }))
        return {"task_id": task_id, "reward": 0.0, "steps": step_num}


def main():
    print(json.dumps({"type": "[START]", "event": "inference_run_begin", "model": MODEL_NAME}))

    results = []
    for task_id in [1, 2, 3]:
        result = run_task(task_id)
        results.append(result)

    print(json.dumps({
        "type": "[END]",
        "event": "inference_run_complete",
        "results": results,
        "average_reward": sum(r["reward"] for r in results) / len(results)
    }))


if __name__ == "__main__":
    main()