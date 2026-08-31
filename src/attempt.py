import json
import os

from openai import RateLimitError

from src.actions import ActionExecutor
from src.agent import DebugAgent, SYSTEM_PROMPT
from src.evaluator import TraceEvaluator
from src.orchestration import (
    build_debugging_state,
    capture_final_verification,
    format_result,
    normalized_read_range,
    overlapping_read,
    print_run_summary,
    trim_output,
)
from src.trace import TraceLogger


async def run_agent_attempt(
    sandbox,
    *,
    repo_url: str,
    task: str,
    baseline_result,
    trace_path: str,
    attempt_number: int = 1,
    snapshot_id: str | None = None,
    demo_fix: bool = False,
) -> dict:
    """Run one isolated debugging trajectory and return its finalized trace."""

    trace = TraceLogger(trace_path)
    sandbox.trace_logger = trace
    trace.set_metadata(
        repo_url=repo_url,
        task=task,
        attempt_number=attempt_number,
        snapshot_id=snapshot_id,
    )
    trace.record_baseline(baseline_result)
    termination_reason = "runtime_error"
    termination_summary = None
    termination_details = {}
    agent = None

    try:
        executor = ActionExecutor(sandbox)
        if demo_fix:
            trace.set_metadata(run_kind="deterministic_demo")
            demo_args = {
                "path": "solve.py",
                "old_text": '            solved.append("G")\n            solved.append("O")',
                "new_text": '            solved.append("L")\n            solved.append("F")',
            }
            result = await executor.execute("replace_text", demo_args)
            trace.record_tool_step(
                "replace_text",
                demo_args,
                result=result,
                actor="orchestrator",
                purpose="deterministic_demo",
            )
            if result.exitCode != 0:
                raise RuntimeError(f"Demo replacement failed: {result.stderr}")
            result = await executor.execute("run_tests", {})
            trace.record_tool_step(
                "run_tests",
                {},
                result=result,
                actor="orchestrator",
                purpose="deterministic_demo",
            )
            if result.exitCode != 0:
                raise RuntimeError(f"Demo verification failed: {format_result(result)}")
            termination_reason = "demo_completed"
            termination_summary = "Deterministic demo fix verified."
            return trace.run

        agent = DebugAgent()
        max_steps = int(os.getenv("MAX_STEPS", "35"))
        trace.set_metadata(
            run_kind="model_agent",
            provider=agent.provider,
            model=agent.model,
            agent_mode=agent.mode,
            max_steps=max_steps,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Repository task:\n{task}\n\nRepository setup is complete.\n\n"
                    f"Initial test result:\n{trim_output(format_result(baseline_result))}\n\n"
                    "Investigate the failure, fix the bug, rerun the tests, and finish when verified."
                ),
            },
        ]
        read_history: dict[str, list[dict]] = {}
        cached_read_requests: dict[tuple[str, int, int], int] = {}
        blocked_reads = 0
        print(
            f"Attempt {attempt_number}: agent mode {agent.mode} "
            f"(completion cap: {agent.max_completion_tokens})"
        )

        for step_num in range(max_steps):
            print(f"\n{'=' * 60}\nATTEMPT {attempt_number}, STEP {step_num + 1}\n{'=' * 60}")
            debugging_state = build_debugging_state(trace.get_steps())
            force_progress = blocked_reads >= 2
            if force_progress:
                debugging_state += (
                    "\n- Progress mode is active after repeated redundant reads. "
                    "Repository inspection is temporarily unavailable. Use replace_text, "
                    "then run_tests; only finish after tests pass."
                )

            response_message = agent.get_next_response(
                messages,
                debugging_state=debugging_state,
                force_progress=force_progress,
            )
            messages.append(response_message)
            if not response_message.tool_calls:
                print("Model response:", response_message.content)
                continue

            finished = False
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as error:
                    output = f"ERROR: Invalid tool arguments: {error}"
                    trace.record_tool_step(tool_name, {}, error=output)
                else:
                    intent = args.pop("intent", None)
                    print("Tool:", tool_name)
                    print("Args:", args)
                    if intent:
                        print("Intent:", intent)

                    if tool_name == "finish":
                        trace.record_tool_step(tool_name, args, intent=intent)
                        termination_reason = "agent_finished"
                        termination_summary = args["summary"]
                        finished = True
                        break

                    cached_response, blocked_read, prior_read = _read_policy(
                        tool_name,
                        args,
                        agent.mode,
                        read_history,
                        cached_read_requests,
                    )
                    if cached_response:
                        output = cached_response
                        trace.record_tool_step(
                            tool_name,
                            args,
                            purpose="cached_read",
                            intent=intent,
                            observation={
                                "stdout": prior_read["stdout"],
                                "stderr": "",
                                "exit_code": 0,
                                "cached": True,
                            },
                            execution={"actor": "orchestrator", "kind": "cached_read"},
                        )
                    elif blocked_read:
                        blocked_reads += 1
                        if blocked_reads >= 2:
                            blocked_read += " Form a hypothesis and either edit the relevant code or run the tests."
                        output = f"ERROR: {blocked_read}"
                        trace.record_tool_step(tool_name, args, error=blocked_read, intent=intent)
                    else:
                        try:
                            result = await executor.execute(tool_name, args)
                            trace.record_tool_step(tool_name, args, result=result, intent=intent)
                            if tool_name == "read_file" and result.exitCode == 0:
                                line_start, line_end = normalized_read_range(args)
                                read_history.setdefault(args["path"], []).append(
                                    {"line_start": line_start, "line_end": line_end, "stdout": result.stdout}
                                )
                                blocked_reads = 0
                                cached_read_requests.clear()
                            elif tool_name in {"write_file", "replace_text"} and result.exitCode == 0:
                                read_history.pop(args["path"], None)
                                blocked_reads = 0
                            elif tool_name == "run_tests" and result.exitCode == 0:
                                blocked_reads = 0
                            output = format_result(result)
                        except Exception as error:
                            output = f"ERROR: {error}"
                            trace.record_tool_step(tool_name, args, error=str(error), intent=intent)

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
            termination_reason = "max_steps_reached"
            termination_summary = f"Stopped after {max_steps} model steps."

    except RateLimitError:
        termination_reason = "provider_rate_limit"
        termination_summary = "The model provider rate limit was reached."
        termination_details = {
            "provider": agent.provider if agent else None,
            "retryable": True,
        }
    except Exception as error:
        termination_summary = str(error)
    finally:
        try:
            final_tests, final_diff = await capture_final_verification(sandbox)
            trace.record_final_verification(final_tests, final_diff)
        except Exception as error:
            trace.record_final_verification(None, None, error=str(error))
        trace.set_termination(termination_reason, termination_summary, **termination_details)
        trace.record_evaluation(TraceEvaluator().evaluate(trace.run))
        trace.save()
        print_run_summary(trace)

    return trace.run


def _read_policy(tool_name, args, mode, read_history, cached_requests):
    if tool_name != "read_file" or mode != "production":
        return None, None, None
    line_start, line_end = normalized_read_range(args)
    prior_read = overlapping_read(read_history, args["path"], line_start, line_end)
    if not prior_read:
        return None, None, None
    key = (args["path"], line_start, line_end)
    prior_range = f"{prior_read['line_start']}-{prior_read['line_end']}"
    if cached_requests.get(key, 0) == 0:
        cached_requests[key] = 1
        output = (
            f"NOTE: {args['path']} lines {line_start}-{line_end} substantially overlap "
            f"a recently inspected range ({prior_range}). Returning cached source evidence.\n\n"
            f"CACHED SOURCE OUTPUT:\n{trim_output(prior_read['stdout'], max_chars=1200)}"
        )
        return output, None, prior_read
    return (
        None,
        (
            f"{args['path']} lines {line_start}-{line_end} substantially overlap "
            f"a recently inspected range ({prior_range}) and its cached output was already returned. "
            "Use existing evidence, inspect a different region, run a test, or make a code change."
        ),
        prior_read,
    )
