# CodeFixEnv — OpenEnv Hackathon Docs

> A complete real-world OpenEnv environment where an AI agent debugs and fixes broken Python code.
> Built for the OpenEnv Hackathon Round 1 by Abdul Maajid.

---

## Table of Contents

1. [What is CodeFixEnv?](#1-what-is-codefixenv)
2. [Understanding OpenEnv (the framework)](#2-understanding-openenv-the-framework)
3. [Project Structure](#3-project-structure)
4. [The 3 Tasks](#4-the-3-tasks)
5. [Reward Logic](#5-reward-logic)
6. [File-by-File Code Guide](#6-file-by-file-code-guide)
   - [models.py](#61-modelspy)
   - [environment.py](#62-environmentpy)
   - [server.py](#63-serverpy)
   - [openenv.yaml](#64-openenvyaml)
   - [inference.py](#65-inferencepy)
   - [Dockerfile](#66-dockerfile)
   - [requirements.txt](#67-requirementstxt)
   - [README.md](#68-readmemd)
7. [How to Run Locally](#7-how-to-run-locally)
8. [How to Deploy to HF Spaces](#8-how-to-deploy-to-hf-spaces)
9. [Pre-submission Checklist](#9-pre-submission-checklist)
10. [Key Concepts Cheat Sheet](#10-key-concepts-cheat-sheet)

---

## 1. What is CodeFixEnv?

CodeFixEnv is an **agentic RL environment** where an AI agent is given **buggy Python code + a failing test**, and must fix the code so the test passes.

This is a real-world task — it's exactly what tools like GitHub Copilot and Cursor do. It's measurable, gradeable, and has a natural reward signal (did the test pass?).

**Why this is a strong submission:**
- Clearly real-world (not a game or toy)
- Reward is objective and automatable
- 3 tasks with clear difficulty progression
- Partial reward signals (judges love this)

---

## 2. Understanding OpenEnv (the framework)

OpenEnv is built by Meta PyTorch + Hugging Face. It standardizes how AI agents interact with environments during Reinforcement Learning training.

Think of it like a gym for AI agents:

```
Agent reads state → picks an action → environment responds with observation + reward
```

### The 3 mandatory API methods

| Method | What it does | Returns |
|--------|-------------|---------|
| `reset()` | Start a fresh episode. Pick a task. | Initial observation (the buggy code + test) |
| `step(action)` | Agent submits fixed code. Grader runs it. | New observation + reward (0.0–1.0) |
| `state()` | Snapshot of current environment state. | Current task, attempt count, last reward |

### How the server works

Your environment runs as a **FastAPI server**. The agent talks to it over WebSocket or HTTP. Each session is isolated — one agent = one environment instance.

```
POST /reset    → resets the environment, returns starting observation
POST /step     → agent sends action, you grade it, return reward
GET  /state    → returns current state (for debugging + graders)
```

### The OpenEnv spec requires

- Typed Pydantic models for Action, Observation, State
- `openenv.yaml` manifest file
- Dockerfile for containerized deployment
- Deployed to Hugging Face Spaces (must return 200 on ping)
- `inference.py` that runs the full agent loop and prints structured logs

---

## 3. Project Structure

```
codefixenv/
├── server/
│   ├── environment.py      # Core logic: tasks, graders, reward
│   ├── models.py           # Pydantic types: Action, Observation, State
│   ├── server.py           # FastAPI app wiring reset/step/state
│   └── Dockerfile          # Container definition
├── inference.py            # LLM agent script (must be in root)
├── openenv.yaml            # Environment manifest
├── requirements.txt        # Python dependencies
├── docs.md                 # This file
└── README.md               # Public-facing description
```

---

## 4. The 3 Tasks

### Task 1 — Easy (syntax / off-by-one)

The agent receives Python code with a simple syntax error or an off-by-one bug.

```python
# Buggy code given to agent
def sum_list(nums):
    total = 0
    for i in range(len(nums) + 1):   # bug: +1 causes IndexError
        total += nums[i]
    return total

# Test the agent must make pass
assert sum_list([1, 2, 3]) == 6
```

**Grader checks:** Code runs without error + assert passes.

---

### Task 2 — Medium (logic bug)

The agent receives code that runs but returns the wrong answer.

```python
# Buggy code given to agent
def is_palindrome(s):
    return s == s[1:]   # bug: should be s[::-1]

# Test the agent must make pass
assert is_palindrome("racecar") == True
assert is_palindrome("hello") == False
```

**Grader checks:** Both assertions pass. Output matches expected.

---

### Task 3 — Hard (multi-bug + edge cases)

The agent receives code with multiple bugs and must handle edge cases.

```python
# Buggy code given to agent
def merge_sorted(a, b):
    result = []
    i, j = 0, 0
    while i < len(a) and j < len(b):
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1
    # bug: missing remainder appending
    return result

# Tests the agent must make pass
assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
assert merge_sorted([], [1, 2]) == [1, 2]       # edge case
assert merge_sorted([1], []) == [1]             # edge case
```

**Grader checks:** All 3 assertions pass including edge cases.

---

## 5. Reward Logic

Every `step()` returns a reward between 0.0 and 1.0. Partial credit is given — this is what judges look for.

| Situation | Reward |
|-----------|--------|
| Code has syntax error, won't even parse | `0.0` |
| Code runs but throws a runtime error | `0.25` |
| Code runs, main test passes, edge cases fail | `0.5` |
| Code runs, most tests pass | `0.75` |
| All tests including edge cases pass | `1.0` |

### Why partial rewards matter

Without partial rewards, the agent gets no learning signal until it fully solves the task. With partial rewards, it can tell "I'm getting closer" — this is the core of RL.

---

## 6. File-by-File Code Guide

### 6.1 `models.py`

This file defines the data types. Pydantic enforces strict typing — required by OpenEnv spec.

```python
from pydantic import BaseModel
from typing import Optional

class CodeAction(BaseModel):
    """The action the agent takes — submitting fixed code."""
    fixed_code: str

class CodeObservation(BaseModel):
    """What the agent sees at each step."""
    task_id: int                    # which task (1, 2, or 3)
    buggy_code: str                 # the broken code to fix
    test_cases: str                 # test(s) the fixed code must pass
    error_message: Optional[str]    # last error (None on first step)
    attempts: int                   # how many times agent has tried

class CodeState(BaseModel):
    """Full snapshot of the environment (for grading + debug)."""
    task_id: int
    current_observation: CodeObservation
    last_reward: float
    done: bool
    total_attempts: int
```

**Key concepts:**
- `CodeAction` is what the agent sends in `step(action)`
- `CodeObservation` is what the agent receives back
- `CodeState` is what `state()` returns — used by graders

---

### 6.2 `environment.py`

This is the brain. All task logic, graders, and reward computation lives here.

```python
import subprocess
import sys
import textwrap
from models import CodeAction, CodeObservation, CodeState

TASKS = {
    1: {
        "buggy_code": textwrap.dedent("""
            def sum_list(nums):
                total = 0
                for i in range(len(nums) + 1):
                    total += nums[i]
                return total
        """).strip(),
        "test_cases": textwrap.dedent("""
            assert sum_list([1, 2, 3]) == 6
            assert sum_list([]) == 0
        """).strip(),
        "difficulty": "easy"
    },
    2: {
        "buggy_code": textwrap.dedent("""
            def is_palindrome(s):
                return s == s[1:]
        """).strip(),
        "test_cases": textwrap.dedent("""
            assert is_palindrome("racecar") == True
            assert is_palindrome("hello") == False
            assert is_palindrome("a") == True
        """).strip(),
        "difficulty": "medium"
    },
    3: {
        "buggy_code": textwrap.dedent("""
            def merge_sorted(a, b):
                result = []
                i, j = 0, 0
                while i < len(a) and j < len(b):
                    if a[i] < b[j]:
                        result.append(a[i])
                        i += 1
                    else:
                        result.append(b[j])
                        j += 1
                return result
        """).strip(),
        "test_cases": textwrap.dedent("""
            assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
            assert merge_sorted([], [1, 2]) == [1, 2]
            assert merge_sorted([1], []) == [1]
        """).strip(),
        "difficulty": "hard"
    }
}

class CodeFixEnvironment:
    def __init__(self):
        self.task_id = None
        self.attempts = 0
        self.last_reward = 0.0
        self.done = False

    def reset(self, task_id: int = 1) -> CodeObservation:
        """Start a fresh episode with the given task."""
        self.task_id = task_id
        self.attempts = 0
        self.last_reward = 0.0
        self.done = False
        task = TASKS[task_id]
        return CodeObservation(
            task_id=task_id,
            buggy_code=task["buggy_code"],
            test_cases=task["test_cases"],
            error_message=None,
            attempts=0
        )

    def step(self, action: CodeAction) -> tuple[CodeObservation, float, bool]:
        """
        Agent submits fixed code.
        Returns: (observation, reward, done)
        """
        self.attempts += 1
        task = TASKS[self.task_id]
        reward, error_msg = self._grade(action.fixed_code, task["test_cases"])
        self.last_reward = reward
        self.done = (reward == 1.0) or (self.attempts >= 5)

        obs = CodeObservation(
            task_id=self.task_id,
            buggy_code=task["buggy_code"],
            test_cases=task["test_cases"],
            error_message=error_msg,
            attempts=self.attempts
        )
        return obs, reward, self.done

    def state(self) -> CodeState:
        """Return full current state snapshot."""
        task = TASKS[self.task_id]
        obs = CodeObservation(
            task_id=self.task_id,
            buggy_code=task["buggy_code"],
            test_cases=task["test_cases"],
            error_message=None,
            attempts=self.attempts
        )
        return CodeState(
            task_id=self.task_id,
            current_observation=obs,
            last_reward=self.last_reward,
            done=self.done,
            total_attempts=self.attempts
        )

    def _grade(self, fixed_code: str, test_cases: str) -> tuple[float, str | None]:
        """
        Run the fixed code + test cases in a subprocess.
        Returns (reward, error_message).
        """
        full_code = fixed_code + "\n\n" + test_cases

        # Check if code even parses
        try:
            compile(full_code, "<string>", "exec")
        except SyntaxError as e:
            return 0.0, f"SyntaxError: {e}"

        # Run in subprocess for safety
        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=10
            )
        except subprocess.TimeoutExpired:
            return 0.0, "Timeout: code took too long to run"

        if result.returncode != 0:
            stderr = result.stderr.strip()
            # Partial reward: runtime error is better than syntax error
            if "AssertionError" in stderr:
                return 0.5, f"Tests failed: {stderr}"
            return 0.25, f"Runtime error: {stderr}"

        # All tests passed
        return 1.0, None
```

**Key concepts:**
- `TASKS` dict stores all 3 task definitions
- `_grade()` runs code safely in a subprocess — never `eval()` untrusted code
- Reward is computed purely based on what the code does, not what the agent says

---

### 6.3 `server.py`

This wires the environment to FastAPI endpoints so it can run as a server.

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models import CodeAction, CodeObservation, CodeState
from environment import CodeFixEnvironment
import uvicorn

app = FastAPI(title="CodeFixEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# One environment instance per server (hackathon scope — single session)
env = CodeFixEnvironment()

@app.get("/")
def health_check():
    """Judges ping this. Must return 200."""
    return {"status": "ok", "env": "CodeFixEnv"}

@app.post("/reset")
def reset(task_id: int = 1) -> CodeObservation:
    """Reset environment and return starting observation."""
    obs = env.reset(task_id=task_id)
    return obs

@app.post("/step")
def step(action: CodeAction) -> dict:
    """Agent submits fixed code. Returns observation + reward."""
    if env.task_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    obs, reward, done = env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done
    }

@app.get("/state")
def state() -> CodeState:
    """Return current environment state."""
    if env.task_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first")
    return env.state()

@app.get("/tasks")
def list_tasks():
    """List all available tasks (required for graders)."""
    return {
        "tasks": [
            {"id": 1, "difficulty": "easy", "description": "Fix an IndexError in a list sum function"},
            {"id": 2, "difficulty": "medium", "description": "Fix incorrect palindrome logic"},
            {"id": 3, "difficulty": "hard", "description": "Fix incomplete merge sort with edge cases"},
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
```

**Key concepts:**
- `/` must return 200 — judges ping this first
- `/reset` takes a `task_id` param (1, 2, or 3)
- Port `7860` is required for HF Spaces

---

### 6.4 `openenv.yaml`

This is the environment manifest. OpenEnv validators read this file.

```yaml
name: codefixenv
version: "1.0.0"
description: >
  An RL environment where an AI agent debugs and fixes broken Python code.
  The agent receives buggy code and a failing test, and must return fixed code
  that makes the test pass. Difficulty scales across 3 tasks.

author: Abdul Maajid
license: MIT

environment:
  type: code-debugging
  real_world: true

tasks:
  - id: 1
    name: fix_sum_list
    difficulty: easy
    description: Fix an IndexError in a list summation function
    max_attempts: 5
    reward_range: [0.0, 1.0]

  - id: 2
    name: fix_palindrome
    difficulty: medium
    description: Fix incorrect string reversal in palindrome checker
    max_attempts: 5
    reward_range: [0.0, 1.0]

  - id: 3
    name: fix_merge_sorted
    difficulty: hard
    description: Fix incomplete merge sort with missing remainder handling
    max_attempts: 5
    reward_range: [0.0, 1.0]

action_space:
  type: text
  schema:
    fixed_code:
      type: string
      description: The corrected Python function code

observation_space:
  type: structured
  schema:
    task_id:
      type: integer
    buggy_code:
      type: string
    test_cases:
      type: string
    error_message:
      type: string
      nullable: true
    attempts:
      type: integer

api:
  reset: POST /reset
  step: POST /step
  state: GET /state

deployment:
  runtime: docker
  port: 7860
  health_check: GET /
```

---

### 6.5 `inference.py`

This is the LLM agent script. It must be in the repo root. Must use OpenAI client. Must print `[START]`, `[STEP]`, `[END]` logs exactly.

```python
"""
inference.py — CodeFixEnv agent
Uses an LLM to fix buggy Python code across all 3 tasks.

Required env vars:
  API_BASE_URL  - LLM API base URL (e.g. https://api.groq.com/openai/v1)
  MODEL_NAME    - Model identifier (e.g. llama-3.3-70b-versatile)
  HF_TOKEN      - Hugging Face token (for Space authentication)
  SPACE_URL     - HF Space URL (e.g. https://yourusername-codefixenv.hf.space)
"""

import os
import json
import requests
from openai import OpenAI

# Config
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "llama-3.3-70b-versatile")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
SPACE_URL    = os.environ.get("SPACE_URL", "http://localhost:7860")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN if HF_TOKEN else "dummy-key"
)

SYSTEM_PROMPT = """You are an expert Python debugger. You will be given buggy Python code and a failing test.
Your job is to return ONLY the fixed Python function — no explanation, no markdown, no backticks.
Just the raw corrected Python code that will make the test pass."""

def call_llm(buggy_code: str, test_cases: str, error_message: str | None) -> str:
    """Ask the LLM to fix the code."""
    error_context = f"\nLast error:\n{error_message}" if error_message else ""
    
    user_prompt = f"""Fix this Python code so the following tests pass.

Buggy code:
{buggy_code}

Tests that must pass:
{test_cases}{error_context}

Return ONLY the fixed function code. No explanation. No markdown."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=512,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()

def run_task(task_id: int) -> dict:
    """Run one full task episode."""
    
    # Reset environment for this task
    reset_resp = requests.post(f"{SPACE_URL}/reset", params={"task_id": task_id})
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
    step_num = 0

    while not done:
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
            json={"fixed_code": fixed_code}
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
```

**Critical rules:**
- Must be named exactly `inference.py` in root directory
- Must use `OpenAI` client (not requests directly for LLM calls)
- `[START]`, `[STEP]`, `[END]` log format is mandatory — judges parse this
- Must complete in under 20 minutes
- Must work on 2 vCPU / 8GB RAM

---

### 6.6 `Dockerfile`

Place this inside `server/Dockerfile`. Judges do automated `docker build`.

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server/ ./server/
COPY openenv.yaml .

# HF Spaces uses port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["python", "server/server.py"]
```

---

### 6.7 `requirements.txt`

```
fastapi==0.115.0
uvicorn==0.30.0
pydantic==2.7.0
openai==1.30.0
requests==2.32.0
```

Keep it minimal. Judges run this on 2 vCPU / 8GB RAM.

---

### 6.8 `README.md`

```markdown
# CodeFixEnv

An OpenEnv-compatible RL environment for AI code debugging agents.

## What it does

The agent receives broken Python code and a failing test. It must return
fixed code that makes the test pass. Rewards are graded 0.0–1.0 with
partial credit for partial progress.

## Tasks

| ID | Difficulty | Task |
|----|-----------|------|
| 1  | Easy      | Fix IndexError in list sum function |
| 2  | Medium    | Fix palindrome string reversal logic |
| 3  | Hard      | Fix merge sort with edge cases |

## Action Space

```json
{ "fixed_code": "<corrected Python function as string>" }
```

## Observation Space

```json
{
  "task_id": 1,
  "buggy_code": "def sum_list...",
  "test_cases": "assert sum_list...",
  "error_message": null,
  "attempts": 0
}
```

## Reward

| Score | Condition |
|-------|-----------|
| 0.0   | Code won't parse |
| 0.25  | Runtime error |
| 0.5   | Test assertion fails |
| 0.75  | Some tests pass |
| 1.0   | All tests pass |

## Setup

```bash
pip install -r requirements.txt
python server/server.py
```

## Run inference

```bash
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_token_here
export SPACE_URL=https://yourusername-codefixenv.hf.space

python inference.py
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/`      | GET    | Health check |
| `/reset` | POST   | Start new episode |
| `/step`  | POST   | Submit fixed code |
| `/state` | GET    | Current environment state |
| `/tasks` | GET    | List all tasks |
```

---

## 7. How to Run Locally

```bash
# 1. Clone your repo
git clone https://github.com/yourusername/codefixenv
cd codefixenv

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server
python server/server.py
# Server running at http://localhost:7860

# 4. Test reset
curl -X POST "http://localhost:7860/reset?task_id=1"

# 5. Test step
curl -X POST "http://localhost:7860/step" \
  -H "Content-Type: application/json" \
  -d '{"fixed_code": "def sum_list(nums):\n    return sum(nums)"}'

# 6. Test state
curl http://localhost:7860/state

# 7. Run inference (after setting env vars)
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export HF_TOKEN=your_groq_api_key
export SPACE_URL=http://localhost:7860
python inference.py
```

---

## 8. How to Deploy to HF Spaces

### Step 1 — Create the Space

1. Go to huggingface.co → click your profile → New Space
2. Name: `codefixenv`
3. SDK: **Docker**
4. Visibility: Public
5. Click Create Space

### Step 2 — Push your code

```bash
# In your project folder
git init
git add .
git commit -m "initial CodeFixEnv submission"

# Add HF Space as remote
git remote add space https://huggingface.co/spaces/YOURUSERNAME/codefixenv

# Push (use your HF token as password)
git push space main
```

### Step 3 — Set Space secrets

In your HF Space settings → Variables and secrets:
- `API_BASE_URL` = `https://api.groq.com/openai/v1`
- `MODEL_NAME` = `llama-3.3-70b-versatile`
- `HF_TOKEN` = your Groq API key

### Step 4 — Verify deployment

Wait for the Space to build (watch the logs). Once green:

```bash
# Should return {"status": "ok", "env": "CodeFixEnv"}
curl https://YOURUSERNAME-codefixenv.hf.space/

# Should return the first task observation
curl -X POST "https://YOURUSERNAME-codefixenv.hf.space/reset?task_id=1"
```

---

## 9. Pre-submission Checklist

Run through every item before submitting.

### Disqualification checks (must all pass)

- [ ] HF Space is public and live (URL returns 200)
- [ ] `/reset` endpoint responds correctly
- [ ] `openenv.yaml` is present and valid
- [ ] `Dockerfile` builds without error
- [ ] `inference.py` is in root directory
- [ ] `inference.py` uses `OpenAI` client for LLM calls
- [ ] `inference.py` prints `[START]`, `[STEP]`, `[END]` JSON logs
- [ ] Inference script finishes in under 20 minutes
- [ ] All 3 tasks are present and graders return scores in 0.0–1.0
- [ ] Environment runs on 2 vCPU / 8GB RAM (no heavy ML models loaded server-side)

### Quality checks (for scoring)

- [ ] Reward function has partial signals (not just 0 or 1)
- [ ] Tasks have genuine difficulty progression (easy → medium → hard)
- [ ] README explains action space and observation space
- [ ] `openenv.yaml` has all required fields
- [ ] Server handles edge cases (what if `/step` called before `/reset`?)

---

## 10. Key Concepts Cheat Sheet

| Term | Simple meaning |
|------|---------------|
| **RL Environment** | A sandbox where an agent can practice a task repeatedly |
| **Episode** | One run from `reset()` to task completion or max attempts |
| **Observation** | What the agent can see (buggy code + test) |
| **Action** | What the agent does (submits fixed code) |
| **Reward** | Score the agent gets for that action (0.0–1.0) |
| **Grader** | The code that checks if the agent's action was correct |
| **OpenEnv spec** | The standard interface (reset/step/state) all environments must follow |
| **HF Space** | Where your environment runs as a live server (judges ping this) |
| **inference.py** | The agent script — it calls the environment and an LLM in a loop |
| **openenv.yaml** | Config manifest that describes your environment to validators |

---

## Free API Options (no paid keys needed)

| Provider | Get key at | Free tier |
|----------|-----------|-----------|
| **Groq** | console.groq.com | Very generous, fast inference |
| **HF Inference** | huggingface.co/settings/tokens | Free tier available |
| **Together AI** | api.together.xyz | Free credits on signup |

Set `API_BASE_URL` to whichever you use. All are OpenAI-compatible.

---

*Built for OpenEnv Hackathon Round 1 — Deadline: 8 April 2026*
