from src.sandbox import SolariSandbox


class ActionExecutor:
    def __init__(self, sandbox: SolariSandbox):
        self.sandbox = sandbox

    async def execute(self, tool_name: str, args: dict):
        match tool_name:
            case "run_command":
                return await self.sandbox.run_in_repo(
                    args["command"],
                    args.get("args", []),
                )

            case "read_file":
                path = args["path"]
                line_start = args.get("line_start") or 1
                line_end = args.get("line_end") or line_start + 99

                if line_end - line_start > 199:
                    line_end = line_start + 199

                return await self.sandbox.run_in_repo(
                    "sed",
                    ["-n", f"{line_start},{line_end}p", path],
                )

            case "search_files":
                return await self.sandbox.run_in_repo(
                    "grep",
                    ["-R", "-n", args["query"], args.get("path") or "."],
                )

            case "write_file":
                path = args["path"]
                content = args["content"]
                script = (
                    "from pathlib import Path; "
                    f"Path({path!r}).write_text({content!r})"
                )

                return await self.sandbox.run_in_repo(
                    "python3",
                    ["-c", script],
                )

            case "replace_text":
                path = args["path"]
                old_text = args["old_text"]
                new_text = args["new_text"]
                script = (
                    "from pathlib import Path; "
                    f"path = Path({path!r}); "
                    "text = path.read_text(); "
                    f"old = {old_text!r}; new = {new_text!r}; "
                    "matches = text.count(old); "
                    "assert matches == 1, "
                    "f'expected exactly one matching snippet, found {matches}'; "
                    "path.write_text(text.replace(old, new, 1))"
                )

                return await self.sandbox.run_in_repo(
                    "python3",
                    ["-c", script],
                )

            case "run_tests":
                return await self.sandbox.run_in_repo(
                    "python3",
                    ["-m", "pytest", "-q"],
                )

            case "finish":
                return None

            case _:
                raise ValueError(f"Unknown tool: {tool_name}")
