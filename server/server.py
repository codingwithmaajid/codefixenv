from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from server.models import CodeAction
from server.environment import CodeFixEnvironment
import uvicorn

app = FastAPI(title="CodeFixEnv", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = CodeFixEnvironment()


@app.get("/")
def health_check():
    return {"status": "ok", "env": "CodeFixEnv"}


@app.post("/reset")
def reset(task_id: int = 1):
    return env.reset(task_id=task_id).model_dump()


@app.post("/step")
def step(action: CodeAction):
    if env.task_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    obs, reward, done, _ = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done
    }


@app.get("/state")
def state():
    if env.task_id is None:
        raise HTTPException(status_code=400, detail="Call /reset first")

    return env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": [
            {"id": 1, "difficulty": "easy"},
            {"id": 2, "difficulty": "medium"},
            {"id": 3, "difficulty": "hard"},
        ]
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)