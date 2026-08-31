import json
from datetime import datetime, timezone
from pathlib import Path


class TraceLogger:
    """Writes one self-contained, evaluation-friendly record per agent run."""

    def __init__(self, output_path: str = "traces/run.json"):
        self.output_path = Path(output_path)
        self.run = {
            "schema_version": "1.0",
            "started_at": self._timestamp(),
            "metadata": {},
            "baseline": None,
            "steps": [],
            "commands": [],
            "final_verification": None,
            "success": None,
            "termination": None,
        }

    def set_metadata(self, **metadata):
        self.run["metadata"].update(metadata)

    def record_command(
        self,
        command: str,
        args: list[str],
        cwd: str | None,
        stdout: str,
        stderr: str,
        exit_code: int,
    ):
        self.run["commands"].append(
            {
                "timestamp": self._timestamp(),
                "command": command,
                "args": args,
                "cwd": cwd,
                "observation": {
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": exit_code,
                },
            }
        )

    def record_baseline(self, result):
        self.run["baseline"] = {"tests": self._observation(result, None)}

    def record_tool_step(
        self,
        tool_name: str,
        args: dict,
        result=None,
        error: str | None = None,
        *,
        actor: str = "model",
        purpose: str | None = None,
        observation: dict | None = None,
        execution: dict | None = None,
    ):
        step = {
            "step": len(self.run["steps"]) + 1,
            "timestamp": self._timestamp(),
            "actor": actor,
            "action": {"type": tool_name, "args": args},
            "observation": observation or self._observation(result, error),
        }
        if purpose:
            step["purpose"] = purpose
        if execution:
            step["execution"] = execution
        self.run["steps"].append(step)

    def record_final_verification(self, tests, diff, error: str | None = None):
        if error is not None:
            self.run["final_verification"] = {"error": error}
            return
        self.run["final_verification"] = {
            "tests": self._observation(tests, None),
            "git_diff": self._observation(diff, None),
            "patch": diff.stdout,
        }
        self.run["success"] = tests.exitCode == 0

    def set_termination(self, reason: str, summary: str | None = None):
        self.run["termination"] = {"reason": reason, "summary": summary}

    @staticmethod
    def _observation(result, error: str | None):
        if result is not None:
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.exitCode,
            }
        if error is not None:
            return {"error": error}
        return None

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.run["completed_at"] = self._timestamp()
        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(self.run, file, indent=2)

    def get_steps(self) -> list[dict]:
        return self.run["steps"]
