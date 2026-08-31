import asyncio
import json
import os

from src.actions import ActionExecutor
from src.agent import DebugAgent, SYSTEM_PROMPT
from src.evaluator import TraceEvaluator
from src.orchestration import (
    build_debugging_state,
    capture_final_verification,
    format_result,
    normalized_read_range,
    overlapping_read,
    trim_output,
)
from src.sandbox import SolariSandbox
from src.trace import TraceLogger
from openai import RateLimitError


REPO_URL = "https://github.com/Immrudul/test-repo"

TASK = """
There is a bug in this repository.

Investigate the repository, reproduce the failure,
identify the root cause, fix the bug, and verify the
solution by running the appropriate tests.
"""


async def main():
    trace = TraceLogger("traces/agent_run_001.json")
    sandbox = SolariSandbox(trace_logger=trace)
    repo_ready = False
    termination_reason = "runtime_error"
    termination_summary = None
    termination_details = {}
    agent = None

    try:
        await sandbox.start()
        await sandbox.clone_repo(REPO_URL)
        repo_ready = True
        trace.set_metadata(repo_url=REPO_URL)

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
        initial_test_output = format_result(initial_test)
        trace.record_baseline(initial_test)

        executor = ActionExecutor(sandbox)
        demo_fix = os.getenv("DEMO_FIX", "").lower() in {
            "1",
            "true",
            "yes",
        }

        if demo_fix:
            trace.set_metadata(run_kind="deterministic_demo")
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
                "replace_text",
                demo_args,
                result=demo_result,
                actor="orchestrator",
                purpose="deterministic_demo",
            )
            if demo_result.exitCode != 0:
                raise RuntimeError(
                    "Demo replacement failed:\n"
                    f"{demo_result.stderr}"
                )

            verification = await executor.execute("run_tests", {})
            trace.record_tool_step(
                "run_tests",
                {},
                result=verification,
                actor="orchestrator",
                purpose="deterministic_demo",
            )
            if verification.exitCode != 0:
                raise RuntimeError(
                    "Demo verification failed:\n"
                    f"{verification.stdout}\n{verification.stderr}"
                )

            print("Demo fix verified: all tests pass.")
            termination_reason = "demo_completed"
            termination_summary = "Deterministic demo fix verified."
            return

        agent = DebugAgent()
        trace.set_metadata(
            run_kind="model_agent",
            provider=agent.provider,
            model=agent.model,
            agent_mode=agent.mode,
            max_steps=int(os.getenv("MAX_STEPS", "35")),
        )
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
                        termination_reason = "agent_finished"
                        termination_summary = args["summary"]
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
                            purpose="cached_read",
                            observation={
                                "stdout": prior_read["stdout"],
                                "stderr": "",
                                "exit_code": 0,
                                "cached": True,
                            },
                            execution={
                                "actor": "orchestrator",
                                "kind": "cached_read",
                            },
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
                            elif (
                                tool_name == "run_tests"
                                and result.exitCode == 0
                            ):
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
            termination_reason = "max_steps_reached"
            termination_summary = f"Stopped after {max_steps} model steps."

    except RateLimitError:
        termination_reason = "provider_rate_limit"
        termination_summary = "The model provider rate limit was reached."
        termination_details = {
            "provider": agent.provider if agent else None,
            "retryable": True,
        }
        raise
    except Exception as error:
        termination_reason = "runtime_error"
        termination_summary = str(error)
        raise
    finally:
        if repo_ready:
            try:
                final_tests, final_diff = await capture_final_verification(sandbox)
                trace.record_final_verification(final_tests, final_diff)
            except Exception as error:
                trace.record_final_verification(None, None, error=str(error))
        trace.set_termination(
            termination_reason,
            termination_summary,
            **termination_details,
        )
        trace.record_evaluation(TraceEvaluator().evaluate(trace.run))
        trace.save()
        await sandbox.stop()


if __name__ == "__main__":
    asyncio.run(main())
