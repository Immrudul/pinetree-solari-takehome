import os
from time import sleep

from openai import OpenAI, RateLimitError


SYSTEM_PROMPT = """
You are an autonomous software debugging agent.

You are debugging a repository inside an isolated Solari sandbox.

Use the provided tools to inspect the repository, read files, search code, run
commands, modify files, run tests, and finish when the bug is fixed and
verified.

Important:
- The provided tools are the ONLY way you can interact with the repository.
- Do not attempt to use container.exec, code_interpreter, browser_search,
  repo_browser, or any undeclared tool.
- Reproduce failures before fixing them when possible.
- Inspect evidence before making assumptions.
- Repository setup and the initial test result are provided by the
  orchestrator.
- Once a failing test has been reproduced, prioritize re-reading the code
  directly related to the failure in a small targeted range. Make a concrete
  hypothesis, modify the relevant code, and rerun the failing test. Avoid
  repeatedly listing files or performing broad searches at this stage.
- Once you identify a concrete suspicious code block consistent with the
  failing test, stop broad repository exploration. Test the smallest relevant
  code change and rerun the failing test.
- Prefer replace_text for a small, targeted edit. Use write_file only to
  create a file or intentionally replace an entire file.
- Do not read the same file region more than twice unless a new failure
  specifically requires it.
- Run tests after changes.
- Only call finish when you have verified the fix.
"""


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a command inside the cloned Solari repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["command", "args"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a specific section of a repository file. Prefer small "
                "targeted ranges of roughly 50-150 lines. Do not request "
                "hundreds of lines unless necessary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                    },
                    "line_start": {
                        "type": "integer",
                        "minimum": 1,
                    },
                    "line_end": {
                        "type": "integer",
                        "minimum": 1,
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search for a specific text string inside repository files. "
                "Do not use this to list files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["query", "path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Replace the contents of a repository file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_text",
            "description": (
                "Make a small targeted edit to an existing file. Use this as "
                "soon as you have identified a specific buggy code block. "
                "Prefer replace_text over further investigation when existing "
                "evidence supports a concrete fix. The old text must match "
                "exactly once."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the repository's Python test suite with pytest.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the task after the fix has been verified.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
                "additionalProperties": False,
            },
        },
    },
]

PROGRESS_TOOL_NAMES = {
    "replace_text",
    "run_tests",
    "finish",
}


def compact_messages(messages):
    if len(messages) <= 10:
        return messages

    return messages[:2] + messages[-8:]


class DebugAgent:
    def __init__(self):
        self.mode = os.getenv("AGENT_MODE", "production").lower()
        if self.mode not in {"production", "evaluation"}:
            raise ValueError(
                "AGENT_MODE must be either 'production' or 'evaluation'."
            )

        provider = os.getenv("LLM_PROVIDER", "gemini").lower()

        if provider == "groq":
            api_key = os.environ["GROQ_API_KEY"]
            base_url = "https://api.groq.com/openai/v1"
            default_model = "qwen/qwen3.6-27b"
        elif provider == "gemini":
            api_key = os.environ["GEMINI_API_KEY"]
            base_url = (
                "https://generativelanguage.googleapis.com/v1beta/openai/"
            )
            default_model = "gemini-3.5-flash-lite"
        elif provider == "ollama":
            api_key = os.getenv("OLLAMA_API_KEY", "ollama")
            base_url = os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434/v1/",
            )
            default_model = "llama3.2:3b"
        else:
            raise ValueError(
                "LLM_PROVIDER must be 'groq', 'gemini', or 'ollama'."
            )

        self.provider = provider
        self.model = os.getenv("LLM_MODEL", default_model)
        default_completion_tokens = "200" if self.mode == "production" else "512"
        self.max_completion_tokens = int(
            os.getenv("MAX_COMPLETION_TOKENS", default_completion_tokens)
        )
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def get_next_response(
        self,
        messages,
        debugging_state: str,
        force_progress: bool = False,
    ):
        messages = compact_messages(messages)
        messages = messages[:2] + [
            {
                "role": "user",
                "content": (
                    "Current factual debugging state from prior tool calls:\n"
                    f"{debugging_state}"
                ),
            }
        ] + messages[2:]
        available_tools = TOOLS
        if force_progress and self.mode == "production":
            available_tools = [
                tool
                for tool in TOOLS
                if tool["function"]["name"] in PROGRESS_TOOL_NAMES
            ]

        for attempt in range(3):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=available_tools,
                    tool_choice="auto",
                    temperature=0.1,
                    max_completion_tokens=self.max_completion_tokens,
                ).choices[0].message
            except RateLimitError as error:
                retry_after_value = error.response.headers.get(
                    "retry-after",
                    "1",
                )
                if retry_after_value.endswith("ms"):
                    retry_after = float(retry_after_value[:-2]) / 1000
                else:
                    retry_after = float(retry_after_value)

                if attempt == 2 or retry_after > 60:
                    raise

                delay = max(retry_after, 0.5) + 0.1
                print(
                    f"{self.provider.title()} rate limit reached. Retrying in "
                    f"{delay:.1f} seconds..."
                )
                sleep(delay)
