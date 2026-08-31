import json
from datetime import datetime, timezone
from pathlib import Path

from src.models import AgentAction


class TraceLogger:
    def __init__(self, output_path: str = "traces/run.json"):
        self.output_path = Path(output_path)
        self.events = []
        self.steps = []

    def record_command(
        self,
        command: str,
        args: list[str],
        cwd: str | None,
        stdout: str,
        stderr: str,
        exit_code: int,
    ):
        self.events.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": "command",
                "command": command,
                "args": args,
                "cwd": cwd,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": exit_code,
            }
        )

    def record_step(
        self,
        action: AgentAction,
        result=None,
        error: str | None = None,
    ):
        observation = self._observation(result, error)

        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reasoning": action.reasoning,
                "action": action.model_dump(),
                "observation": observation,
            }
        )

    def record_tool_step(
        self,
        tool_name: str,
        args: dict,
        result=None,
        error: str | None = None,
    ):
        observation = self._observation(result, error)

        self.steps.append(
            {
                "step": len(self.steps) + 1,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "action": {
                    "type": tool_name,
                    "args": args,
                },
                "observation": observation,
            }
        )

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

    def save(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with self.output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "steps": self.steps,
                    "events": self.events,
                },
                file,
                indent=2,
            )

    def get_steps(self):
        return self.steps
