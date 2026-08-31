import json
from datetime import datetime, timezone
from pathlib import Path


def rank_trace(trace: dict) -> tuple:
    execution = (trace.get("evaluation") or {}).get("execution") or {}
    quality = (trace.get("evaluation") or {}).get("agent_quality") or {}
    scores = quality.get("scores") or {}

    def score(name: str) -> int:
        return (scores.get(name) or {}).get("score") or 0

    return (
        int(execution.get("verified_success") is True),
        score("accuracy"),
        score("evidence"),
        score("efficiency"),
        quality.get("overall") or 0,
    )


def save_experiment(
    output_path: str,
    *,
    experiment_id: str,
    repo_url: str,
    task: str,
    snapshot_id: str,
    attempts: list[dict],
) -> dict:
    ranked = sorted(attempts, key=lambda item: rank_trace(item["trace"]), reverse=True)
    records = []
    for item in attempts:
        trace = item["trace"]
        execution = (trace.get("evaluation") or {}).get("execution") or {}
        quality = (trace.get("evaluation") or {}).get("agent_quality") or {}
        records.append(
            {
                "attempt": item["attempt"],
                "trace_path": item["trace_path"],
                "success": execution.get("verified_success"),
                "overall": quality.get("overall"),
                "steps": len(trace.get("steps", [])),
                "termination": (trace.get("termination") or {}).get("reason"),
            }
        )
    winner = ranked[0]["attempt"] if ranked else None
    experiment = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_url": repo_url,
        "task": task,
        "snapshot_id": snapshot_id,
        "attempt_count": len(attempts),
        "attempts": records,
        "winner": winner,
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    return experiment


def print_experiment_summary(experiment: dict, output_path: str):
    print(f"\n{'=' * 60}\nSOLARI MULTI-TRAJECTORY EXPERIMENT\n{'=' * 60}")
    print("Shared buggy snapshot created.")
    for attempt in experiment["attempts"]:
        status = "success" if attempt["success"] else "incomplete"
        score = attempt["overall"]
        score_text = f"{score:.1f}/10" if isinstance(score, (int, float)) else "not scored"
        print(f"#{attempt['attempt']}  {status:<10} {attempt['steps']:>2} steps  {score_text}")
    print(f"Best trajectory: attempt #{experiment['winner']}")
    print(f"Comparison: {output_path}")
