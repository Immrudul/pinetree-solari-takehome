import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path

from src.attempt import run_agent_attempt
from src.experiment import print_experiment_summary, save_experiment
from src.sandbox import SolariSandbox


DEFAULT_REPO_URL = "https://github.com/Immrudul/test-repo"
DEFAULT_TASK = """
There is a bug in this repository.

Investigate the repository, reproduce the failure,
identify the root cause, fix the bug, and verify the
solution by running the appropriate tests.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Solari debugging agent against a repository task."
    )
    parser.add_argument("--repo", help="Git repository URL. Overrides REPO_URL.")
    parser.add_argument("--task", help="Debugging task. Overrides TASK.")
    parser.add_argument(
        "--attempts",
        type=int,
        help="Independent attempts from one prepared snapshot. Overrides NUM_ATTEMPTS.",
    )
    parser.add_argument(
        "--trace-path",
        help="Trace file for a single attempt. Overrides TRACE_PATH.",
    )
    parser.add_argument(
        "--experiment-id",
        help="Name for a multi-attempt experiment output directory.",
    )
    return parser.parse_args()


async def prepare_benchmark(sandbox: SolariSandbox, repo_url: str):
    await sandbox.start()
    await sandbox.clone_repo(repo_url)

    print("Installing repository dependencies...")
    setup = await sandbox.run_in_repo("pip3", ["install", "-r", "requirements.txt"])
    if setup.exitCode != 0:
        raise RuntimeError(f"Failed to install dependencies:\n{setup.stderr}")

    print("Running the initial test suite...")
    baseline = await sandbox.run_in_repo("python3", ["-m", "pytest", "-q"])
    if baseline.exitCode == 0:
        raise RuntimeError(
            "Benchmark does not reproduce a failing baseline; refusing to run attempts."
        )
    return baseline


async def run_single_attempt(
    sandbox: SolariSandbox,
    *,
    repo_url: str,
    task: str,
    baseline,
    trace_path: str,
    demo_fix: bool,
):
    return await run_agent_attempt(
        sandbox,
        repo_url=repo_url,
        task=task,
        baseline_result=baseline,
        trace_path=trace_path,
        demo_fix=demo_fix,
    )


async def run_fork_experiment(
    base: SolariSandbox,
    *,
    repo_url: str,
    task: str,
    baseline,
    attempts: int,
    experiment_id: str,
):
    snapshot_id = await base.sandbox.snapshot("buggy-ready")
    print(f"Created shared buggy snapshot: {snapshot_id}")
    await base.stop()  # Frees the concurrency slot before restoring children.

    output_dir = Path("traces") / experiment_id
    results = []
    for attempt_number in range(1, attempts + 1):
        trace_path = output_dir / f"attempt_{attempt_number}.json"
        child = SolariSandbox()
        try:
            await child.start_from_snapshot(snapshot_id)
            trace = await run_agent_attempt(
                child,
                repo_url=repo_url,
                task=task,
                baseline_result=baseline,
                trace_path=str(trace_path),
                attempt_number=attempt_number,
                snapshot_id=snapshot_id,
            )
            results.append(
                {
                    "attempt": attempt_number,
                    "trace_path": str(trace_path),
                    "trace": trace,
                }
            )
        finally:
            await child.stop()

    comparison_path = output_dir / "comparison.json"
    experiment = save_experiment(
        str(comparison_path),
        experiment_id=experiment_id,
        repo_url=repo_url,
        task=task,
        snapshot_id=snapshot_id,
        attempts=results,
    )
    print_experiment_summary(experiment, str(comparison_path))


async def main(repo_url: str, task: str, attempts: int, trace_path: str, experiment_id: str):
    if attempts < 1:
        raise ValueError("--attempts must be at least 1.")

    demo_fix = os.getenv("DEMO_FIX", "").lower() in {"1", "true", "yes"}
    if demo_fix and repo_url != DEFAULT_REPO_URL:
        raise ValueError("DEMO_FIX is only available for the bundled default repository.")
    if demo_fix and attempts != 1:
        raise ValueError("DEMO_FIX cannot be used with multiple attempts.")

    base = SolariSandbox()
    try:
        baseline = await prepare_benchmark(base, repo_url)
        if attempts == 1:
            await run_single_attempt(
                base,
                repo_url=repo_url,
                task=task,
                baseline=baseline,
                trace_path=trace_path,
                demo_fix=demo_fix,
            )
        else:
            await run_fork_experiment(
                base,
                repo_url=repo_url,
                task=task,
                baseline=baseline,
                attempts=attempts,
                experiment_id=experiment_id,
            )
            return
    finally:
        await base.stop()


if __name__ == "__main__":
    args = parse_args()
    repo_url = args.repo or os.getenv("REPO_URL", DEFAULT_REPO_URL)
    task = args.task or os.getenv("TASK", DEFAULT_TASK)
    attempts = args.attempts or int(os.getenv("NUM_ATTEMPTS", "1"))
    trace_path = args.trace_path or os.getenv("TRACE_PATH", "traces/agent_run_001.json")
    default_experiment_id = "experiment_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    experiment_id = args.experiment_id or os.getenv("EXPERIMENT_ID", default_experiment_id)
    asyncio.run(main(repo_url, task, attempts, trace_path, experiment_id))
