from pydantic import BaseModel
from typing import Optional


class CodeAction(BaseModel):
    fixed_code: str


class CodeObservation(BaseModel):
    task_id: int
    buggy_code: str
    test_cases: str
    error_message: Optional[str] = None
    attempts: int = 0


class CodeState(BaseModel):
    task_id: int
    current_observation: CodeObservation
    last_reward: float = 0.0
    done: bool
    total_attempts: int