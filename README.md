# 🚀 CodeFixEnv — AI Code Debugging Environment

## 🧠 Overview

**CodeFixEnv** is an OpenEnv-compatible environment where an AI agent learns to fix buggy Python code using automated testing and reward feedback.

This environment simulates a **real-world developer task**: debugging and correcting code based on failing test cases.

---

## 🎯 Motivation

Debugging is one of the most common tasks in software engineering.
Modern AI tools like code assistants aim to automate this process.

**CodeFixEnv** provides a structured environment to:

* Evaluate AI agents on code fixing ability
* Provide reward-based feedback
* Simulate real debugging workflows

---

## ⚙️ Environment Design

### 🔄 Core API

The environment follows the OpenEnv standard:

* `POST /reset` → Initializes a task and returns buggy code
* `POST /step` → Accepts fixed code and returns reward + status
* `GET /state` → Returns full environment state

---

## 📦 Action Space

### `CodeAction`

```json
{
  "fixed_code": "string"
}
```

The agent submits a corrected version of the buggy code.

---

## 👀 Observation Space

### `CodeObservation`

```json
{
  "task_id": int,
  "buggy_code": "string",
  "test_cases": "string",
  "error_message": "string | null",
  "attempts": int
}
```

The agent receives:

* Original buggy code
* Test cases
* Error feedback
* Attempt count

---

## 🧾 State Representation

### `CodeState`

```json
{
  "task_id": int,
  "current_observation": {...},
  "last_reward": float,
  "done": bool,
  "total_attempts": int
}
```

---

## 🧪 Tasks

### 🟢 Easy

Fix an **IndexError** in a list summation function.

### 🟡 Medium

Fix incorrect logic in a **palindrome checker**.

### 🔴 Hard

Fix incomplete logic in a **merge sorted lists** function.

---

## 🎯 Reward Function

| Condition       | Reward |
| --------------- | ------ |
| Syntax error    | 0.0    |
| Runtime error   | 0.25   |
| Test cases fail | 0.5    |
| All tests pass  | 1.0    |

This provides **dense feedback** for progressive improvement.

---

## 🤖 Baseline Inference

The `inference.py` script simulates an agent:

* Fetches buggy code
* Applies simple fixes
* Sends actions to environment
* Logs results using:

```
[START]
[STEP]
[END]
```

This ensures **reproducible evaluation**.

---

## 🐳 Running Locally (Docker)

```bash
docker build -t codefixenv .
docker run -p 8000:8000 codefixenv
```

Test:

```bash
curl -X POST http://localhost:8000/reset
```

---

## 🔧 Environment Variables

| Variable     | Description      |
| ------------ | ---------------- |
| API_BASE_URL | API endpoint     |
| MODEL_NAME   | Model identifier |
| HF_TOKEN     | API key          |

---

## ☁️ Deployment

This project is deployed as a **Hugging Face Space (Docker)**.

It supports:

* Automated evaluation
* External agent interaction
* OpenEnv validation

---

## ✅ Features

* Real-world task simulation (code debugging)
* Deterministic grading using test cases
* Safe execution via subprocess sandboxing
* Multi-level difficulty tasks
* Structured reward system
* Dockerized deployment

---

## 📌 Submission Checklist

* ✅ OpenEnv API implemented
* ✅ 3 tasks with graders
* ✅ Reward system (0.0 → 1.0)
* ✅ Dockerfile working
* ✅ Inference script working
* ✅ Hugging Face deployment ready

---

## 🚀 Future Improvements

* Integrate real LLM-based agent
* Add more complex debugging tasks
* Introduce multi-step reasoning tasks

---

## 👤 Author

**Abdul Maajid**

---

## 🏁 Conclusion

CodeFixEnv provides a practical and extensible environment for training and evaluating AI agents on real-world debugging tasks.

---

