def trim_output(output: str, max_chars: int = 2000) -> str:
    if len(output) <= max_chars:
        return output

    half = max_chars // 2
    return (
        output[:half]
        + "\n\n... [output truncated] ...\n\n"
        + output[-half:]
    )


def format_result(result) -> str:
    output = result.stdout
    if result.stderr:
        output += "\nSTDERR:\n" + result.stderr
    return output + f"\nExit code: {result.exitCode}"


def summarize_output(output: str, max_chars: int = 360) -> str:
    normalized = " ".join(output.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[:max_chars] + "..."


def build_debugging_state(steps: list[dict]) -> str:
    files_seen = []
    read_ranges = []
    recent_commands = []
    failures = []
    source_evidence = []

    for step in steps:
        action = step["action"]
        tool_name = action["type"]
        args = action["args"]
        observation = step.get("observation") or {}

        if tool_name == "read_file":
            path = args.get("path", "unknown file")
            if path not in files_seen:
                files_seen.append(path)
            start = args.get("line_start")
            end = args.get("line_end")
            line_range = f" lines {start}-{end}" if start else ""
            read_ranges.append(f"{path}{line_range}")
            if observation.get("exit_code") == 0:
                excerpt = trim_output(observation.get("stdout", ""), max_chars=500)
                if excerpt:
                    source_evidence.append(f"{path}{line_range}:\n{excerpt}")
        elif tool_name == "search_files":
            recent_commands.append(
                f"search {args.get('path') or '.'} for {args.get('query', '')!r}"
            )
        elif tool_name == "run_command":
            command = " ".join([args.get("command", "")] + args.get("args", []))
            recent_commands.append(
                f"`{command}` (exit {observation.get('exit_code')})"
            )

        if observation.get("error"):
            failures.append(f"Error: {observation['error']}")
        elif observation.get("exit_code") not in (None, 0):
            output = summarize_output(observation.get("stdout", ""))
            if output:
                failures.append(f"Failure output: {output}")

    if not files_seen and not recent_commands:
        return "No repository actions have run yet."

    state_lines = []
    if files_seen:
        state_lines.append("Files inspected: " + ", ".join(files_seen))
    if read_ranges:
        state_lines.append("Recent file reads: " + ", ".join(read_ranges[-4:]))
    if recent_commands:
        state_lines.append("Recent commands: " + "; ".join(recent_commands[-4:]))
    if failures:
        state_lines.append("Recent failures: " + " | ".join(failures[-2:]))
    if source_evidence:
        state_lines.append(
            "Recent source evidence:\n" + "\n---\n".join(source_evidence[-3:])
        )

    return trim_output("\n".join(f"- {line}" for line in state_lines), 1800)


def normalized_read_range(args: dict) -> tuple[int, int]:
    line_start = args.get("line_start") or 1
    line_end = args.get("line_end") or line_start + 99
    line_end = max(line_start, line_end)
    return line_start, min(line_end, line_start + 199)


def overlapping_read(
    read_history: dict[str, list[dict]],
    path: str,
    line_start: int,
    line_end: int,
) -> dict | None:
    for prior_read in read_history.get(path, []):
        prior_start = prior_read["line_start"]
        prior_end = prior_read["line_end"]
        overlap = max(0, min(line_end, prior_end) - max(line_start, prior_start) + 1)
        shorter_range = min(line_end - line_start + 1, prior_end - prior_start + 1)
        if overlap / shorter_range >= 0.75:
            return prior_read
    return None


async def capture_final_verification(sandbox):
    tests = await sandbox.run_in_repo("python3", ["-m", "pytest", "-q"])
    diff = await sandbox.run_in_repo("git", ["diff", "--"])
    return tests, diff
