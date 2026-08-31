from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str

    action: Literal[
        "run_command",
        "read_file",
        "search_files",
        "write_file",
        "finish",
    ]

    command: str | None = None
    command_args: list[str] = Field(default_factory=list)

    path: str | None = None
    query: str | None = None
    content: str | None = None

    summary: str | None = None
