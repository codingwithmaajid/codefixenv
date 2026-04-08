import subprocess
import sys
import textwrap
from server.models import CodeAction, CodeObservation, CodeState


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
    }
}


class CodeFixEnvironment:

    def __init__(self):
        self.task_id = None
        self.attempts = 0
        self.last_reward = 0.0
        self.done = False

    def reset(self, task_id: int = 1) -> CodeObservation:
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

    def step(self, action: CodeAction):
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

        return obs, reward, self.done, {}

    def state(self) -> CodeState:
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

    def _grade(self, fixed_code: str, test_cases: str):
        full_code = fixed_code + "\n\n" + test_cases

        try:
            compile(full_code, "<string>", "exec")
        except SyntaxError as e:
            return 0.0, f"SyntaxError: {e}"

        try:
            result = subprocess.run(
                [sys.executable, "-c", full_code],
                capture_output=True,
                text=True,
                timeout=10
            )
        except subprocess.TimeoutExpired:
            return 0.0, "Timeout: code took too long"

        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "AssertionError" in stderr:
                return 0.5, f"Tests failed: {stderr}"
            return 0.25, f"Runtime error: {stderr}"

        return 1.0, None