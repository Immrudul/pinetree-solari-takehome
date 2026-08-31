import asyncio
import json
import os

from src.actions import ActionExecutor
from src.agent import DebugAgent, SYSTEM_PROMPT
from src.sandbox import SolariSandbox
from src.trace import TraceLogger


REPO_URL = "https://github.com/Immrudul/test-repo"

TASK = """
There is a bug in this repository.

Investigate the repository, reproduce the failure,
identify the root cause, fix the bug, and verify the
solution by running the appropriate tests.
"""


def trim_output(output: str, max_chars: int = 2000) -> str:
    if len(output) <= max_chars:
        return output

    half = max_chars // 2
    return (
        output[:half]
        + "\n\n... [output truncated] ...\n\n"
        + output[-half:]
    )


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
                excerpt = trim_output(
                    observation.get("stdout", ""),
                    max_chars=500,
                )
                if excerpt:
                    source_evidence.append(
                        f"{path}{line_range}:\n{excerpt}"
                    )
        elif tool_name == "search_files":
            recent_commands.append(
                f"search {args.get('path') or '.'} for "
                f"{args.get('query', '')!r}"
            )
        elif tool_name == "run_command":
            command = " ".join(
                [args.get("command", "")] + args.get("args", [])
            )
            exit_code = observation.get("exit_code")
            recent_commands.append(f"`{command}` (exit {exit_code})")

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

    state = "\n".join(f"- {line}" for line in state_lines)
    return trim_output(state, max_chars=1800)


def normalized_read_range(args: dict) -> tuple[int, int]:
    line_start = args.get("line_start") or 1
    line_end = args.get("line_end") or line_start + 99
    line_end = max(line_start, line_end)

    if line_end - line_start > 199:
        line_end = line_start + 199

    return line_start, line_end


def overlapping_read(
    read_history: dict[str, list[dict]],
    path: str,
    line_start: int,
    line_end: int,
) -> dict | None:
    for prior_read in read_history.get(path, []):
        prior_start = prior_read["line_start"]
        prior_end = prior_read["line_end"]
        overlap = max(
            0,
            min(line_end, prior_end) - max(line_start, prior_start) + 1,
        )
        shorter_range = min(
            line_end - line_start + 1,
            prior_end - prior_start + 1,
        )

        if overlap / shorter_range >= 0.75:
            return prior_read

    return None


async def main():
    trace = TraceLogger("traces/agent_run_001.json")
    sandbox = SolariSandbox(trace_logger=trace)

    try:
        await sandbox.start()
        await sandbox.clone_repo(REPO_URL)

        print("Installing repository dependencies...")
        setup_result = await sandbox.run_in_repo(
            "pip3",
            ["install", "-r", "requirements.txt"],
        )
        if setup_result.exitCode != 0:
            raise RuntimeError(
                "Failed to install repository dependencies:\n"
                f"{setup_result.stderr}"
            )

        print("Running the initial test suite...")
        initial_test = await sandbox.run_in_repo(
            "python3",
            ["-m", "pytest", "-q"],
        )
        initial_test_output = initial_test.stdout
        if initial_test.stderr:
            initial_test_output += "\nSTDERR:\n" + initial_test.stderr
        initial_test_output += f"\nExit code: {initial_test.exitCode}"

        executor = ActionExecutor(sandbox)
        demo_fix = os.getenv("DEMO_FIX", "").lower() in {
            "1",
            "true",
            "yes",
        }

        if demo_fix:
            print("Running orchestrator demo fix...")
            demo_args = {
                "path": "solve.py",
                "old_text": (
                    '            solved.append("G")\n'
                    '            solved.append("O")'
                ),
                "new_text": (
                    '            solved.append("L")\n'
                    '            solved.append("F")'
                ),
            }
            demo_result = await executor.execute("replace_text", demo_args)
            trace.record_tool_step(
                "orchestrator_demo_replace_text",
                demo_args,
                result=demo_result,
            )
            if demo_result.exitCode != 0:
                raise RuntimeError(
                    "Demo replacement failed:\n"
                    f"{demo_result.stderr}"
                )

            verification = await executor.execute("run_tests", {})
            trace.record_tool_step(
                "orchestrator_demo_run_tests",
                {},
                result=verification,
            )
            if verification.exitCode != 0:
                raise RuntimeError(
                    "Demo verification failed:\n"
                    f"{verification.stdout}\n{verification.stderr}"
                )

            print("Demo fix verified: all tests pass.")
            return

        agent = DebugAgent()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"{TASK}\n\nRepository setup is complete.\n\n"
                    "Initial test result:\n"
                    f"{trim_output(initial_test_output)}\n\n"
                    "Investigate the failure, fix the bug, rerun the tests, "
                    "and finish when verified."
                ),
            },
        ]
        max_steps = int(os.getenv("MAX_STEPS", "35"))
        read_history: dict[str, list[dict]] = {}
        cached_read_requests: dict[tuple[str, int, int], int] = {}
        blocked_reads = 0

        print(
            f"Agent mode: {agent.mode} "
            f"(completion cap: {agent.max_completion_tokens})"
        )

        for step_num in range(max_steps):
            print(f"\n{'=' * 60}\nSTEP {step_num + 1}\n{'=' * 60}")

            debugging_state = build_debugging_state(trace.get_steps())
            force_progress = blocked_reads >= 2
            if force_progress:
                debugging_state += (
                    "\n- Progress mode is active after repeated redundant reads. "
                    "Repository inspection is temporarily unavailable. "
                    "If you have a concrete hypothesis, use replace_text to make the smallest "
                    "targeted edit, then use run_tests. Only use finish after tests pass."
                )

            response_message = agent.get_next_response(
                messages,
                debugging_state=debugging_state,
                force_progress=force_progress,
            )
            messages.append(response_message)

            tool_calls = response_message.tool_calls
            if not tool_calls:
                print("Model response:", response_message.content)
                continue

            finished = False

            for tool_call in tool_calls:
                tool_name = tool_call.function.name

                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as error:
                    output = f"ERROR: Invalid tool arguments: {error}"
                    trace.record_tool_step(
                        tool_name,
                        {},
                        error=output,
                    )
                else:
                    print("Tool:", tool_name)
                    print("Args:", args)

                    if tool_name == "finish":
                        trace.record_tool_step(tool_name, args)
                        print("\nAgent finished.")
                        print("Summary:", args["summary"])
                        finished = True
                        break

                    cached_read_response = None
                    blocked_read = None
                    if tool_name == "read_file" and agent.mode == "production":
                        path = args["path"]
                        line_start, line_end = normalized_read_range(args)
                        prior_read = overlapping_read(
                            read_history,
                            path,
                            line_start,
                            line_end,
                        )
                        if prior_read:
                            request_key = (path, line_start, line_end)
                            cached_count = cached_read_requests.get(request_key, 0)
                            prior_range = (
                                f"{prior_read['line_start']}-{prior_read['line_end']}"
                            )
                            if cached_count == 0:
                                cached_read_requests[request_key] = 1
                                cached_read_response = (
                                    f"NOTE: {path} lines {line_start}-{line_end} substantially "
                                    "overlap a recently inspected range "
                                    f"({prior_range}). Returning the cached source evidence "
                                    "instead of rerunning the same read.\n\n"
                                    "CACHED SOURCE OUTPUT:\n"
                                    f"{trim_output(prior_read['stdout'], max_chars=1200)}"
                                )
                            else:
                                blocked_read = (
                                f"{path} lines {line_start}-{line_end} substantially "
                                "overlap a recently inspected range "
                                f"({prior_range}) and its cached output was already returned. Use existing "
                                "evidence, inspect a different region, run a test, or "
                                "make a code change."
                                )

                    if cached_read_response:
                        output = cached_read_response
                        trace.record_tool_step(
                            tool_name,
                            args,
                            error="Overlapping read served from cache.",
                        )
                    elif blocked_read:
                        blocked_reads += 1
                        if blocked_reads >= 2:
                            blocked_read += (
                                " You have attempted multiple redundant reads. "
                                "Form a hypothesis and either edit the relevant "
                                "code or run the tests."
                            )
                        output = f"ERROR: {blocked_read}"
                        trace.record_tool_step(
                            tool_name,
                            args,
                            error=blocked_read,
                        )
                    else:
                        try:
                            result = await executor.execute(tool_name, args)
                            trace.record_tool_step(tool_name, args, result=result)

                            if tool_name == "read_file" and result.exitCode == 0:
                                read_history.setdefault(path, []).append(
                                    {
                                        "line_start": line_start,
                                        "line_end": line_end,
                                        "stdout": result.stdout,
                                    }
                                )
                                blocked_reads = 0
                                cached_read_requests.clear()
                            elif (
                                tool_name in {"write_file", "replace_text"}
                                and result.exitCode == 0
                            ):
                                read_history.pop(args["path"], None)
                                blocked_reads = 0
                            elif tool_name == "run_tests":
                                blocked_reads = 0

                            output = result.stdout
                            if result.stderr:
                                output += "\nSTDERR:\n" + result.stderr
                            output += f"\nExit code: {result.exitCode}"

                        except Exception as error:
                            output = f"ERROR: {error}"
                            trace.record_tool_step(
                                tool_name,
                                args,
                                error=str(error),
                            )

                print(output)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": trim_output(output),
                    }
                )

            if finished:
                break

        else:
            print(f"Stopped after {max_steps} steps.")

    finally:
        trace.save()
        await sandbox.stop()


if __name__ == "__main__":
    asyncio.run(main())
