import json


EDIT_ACTIONS = {"replace_text", "write_file"}
INVESTIGATION_ACTIONS = {"read_file", "search_files", "run_command"}


class TraceEvaluator:
    """Scores verified execution and model-authored trajectory separately."""

    VERSION = "2.0"

    def evaluate(self, trace: dict) -> dict:
        steps = trace.get("steps", [])
        metadata = trace.get("metadata") or {}
        baseline_exit = self._exit_code((trace.get("baseline") or {}).get("tests"))
        final_verification = trace.get("final_verification") or {}
        final_exit = self._exit_code(final_verification.get("tests"))
        patch = final_verification.get("patch", "")

        baseline_failed = baseline_exit not in (None, 0)
        final_passed = final_exit == 0
        patch_present = bool(patch.strip())
        model_steps = [step for step in steps if step.get("actor") == "model"]
        all_edit_steps = self._successful_actions(steps, EDIT_ACTIONS)
        model_edit_steps = self._successful_actions(model_steps, EDIT_ACTIONS)
        first_model_edit_step = self._first_step(model_edit_steps)
        model_investigation_steps = self._actions(
            model_steps, INVESTIGATION_ACTIONS
        )
        model_investigation_steps_before_edit = self._investigation_before_edit(
            model_steps, first_model_edit_step
        )
        model_tests = self._actions(model_steps, {"run_tests"})
        model_tests_after_edit = [
            step
            for step in model_tests
            if first_model_edit_step is not None and step["step"] > first_model_edit_step
            and self._exit_code(step.get("observation")) is not None
        ]
        model_passing_tests_after_edit = [
            step
            for step in model_tests_after_edit
            if self._exit_code(step.get("observation")) == 0
        ]
        failed_model_actions = [
            step for step in model_steps if (step.get("observation") or {}).get("error")
        ]
        cached_model_reads = [
            step for step in model_steps if (step.get("observation") or {}).get("cached")
        ]
        duplicate_action_count = self._duplicate_investigations(model_steps)
        tests_before_first_edit = (
            sum(1 for step in model_tests if step["step"] < first_model_edit_step)
            if first_model_edit_step is not None
            else None
        )

        execution = {
            "baseline_reproduced": baseline_failed,
            "baseline_exit_code": baseline_exit,
            "successful_execution_edit_count": len(all_edit_steps),
            "patch_captured": patch_present,
            "final_verification_passed": final_passed,
            "final_test_exit_code": final_exit,
            "verified_success": final_passed,
            "trace_success_flag": trace.get("success"),
        }
        agent_quality = self._agent_quality(
            run_kind=metadata.get("run_kind", "unknown"),
            model_steps=model_steps,
            model_edit_steps=model_edit_steps,
            model_investigation_steps=model_investigation_steps,
            model_investigation_steps_before_edit=(
                model_investigation_steps_before_edit
            ),
            model_tests_after_edit=model_tests_after_edit,
            model_passing_tests_after_edit=model_passing_tests_after_edit,
            baseline_failed=baseline_failed,
            final_passed=final_passed,
            final_exit=final_exit,
            patch_present=patch_present,
            first_model_edit_step=first_model_edit_step,
            failed_model_actions=failed_model_actions,
            cached_model_reads=cached_model_reads,
            duplicate_action_count=duplicate_action_count,
            tests_before_first_edit=tests_before_first_edit,
        )

        return {
            "evaluator_version": self.VERSION,
            "provenance": metadata.get("run_kind", "unknown"),
            "execution": execution,
            "agent_quality": agent_quality,
        }

    def _agent_quality(
        self,
        *,
        run_kind: str,
        model_steps: list[dict],
        model_edit_steps: list[dict],
        model_investigation_steps: list[dict],
        model_investigation_steps_before_edit: list[dict],
        model_tests_after_edit: list[dict],
        model_passing_tests_after_edit: list[dict],
        baseline_failed: bool,
        final_passed: bool,
        final_exit: int | None,
        patch_present: bool,
        first_model_edit_step: int | None,
        failed_model_actions: list[dict],
        cached_model_reads: list[dict],
        duplicate_action_count: int,
        tests_before_first_edit: int | None,
    ) -> dict:
        if run_kind == "deterministic_demo" or not model_steps:
            reason = (
                "The deterministic demo patch was selected by the orchestrator, "
                "not the model."
                if run_kind == "deterministic_demo"
                else (
                    "No autonomous model-generated actions were recorded; "
                    "only execution quality is evaluated."
                )
            )
            return {
                "scored": False,
                "reason": reason,
                "scores": None,
                "overall": None,
                "deterministic_metrics": {
                    "model_step_count": 0,
                    "model_edit_count": 0,
                },
            }

        logical_flow = 0
        logical_flow += 2 if baseline_failed else 0
        logical_flow += 2 if model_investigation_steps else 0
        logical_flow += 3 if model_edit_steps else 0
        logical_flow += 3 if model_tests_after_edit else 0

        efficiency = 10
        efficiency -= min(4, max(0, len(model_steps) - 8) // 2)
        efficiency -= min(2, len(failed_model_actions))
        efficiency -= min(2, len(cached_model_reads))
        efficiency -= min(2, duplicate_action_count)
        if tests_before_first_edit is not None:
            efficiency -= min(2, max(0, tests_before_first_edit - 1))
        if first_model_edit_step is not None and first_model_edit_step > 8:
            efficiency -= 1

        evidence = 0
        evidence += 3 if baseline_failed else 0
        evidence += 2 if model_investigation_steps else 0
        evidence += 2 if model_edit_steps else 0
        evidence += 1 if final_exit is not None else 0
        evidence += 2 if final_passed else 0

        accuracy = 0
        if model_edit_steps and final_passed:
            accuracy = 8 + (2 if patch_present else 0)
        elif model_edit_steps and final_exit is not None:
            accuracy = 2

        scores = {
            "logical_flow": self._score(
                logical_flow,
                self._logical_flow_reason(
                    baseline_failed,
                    bool(model_investigation_steps),
                    bool(model_edit_steps),
                    bool(model_tests_after_edit),
                ),
            ),
            "clarity": {
                "score": None,
                "reason": (
                    "Not scored: the trace does not yet record concise "
                    "per-action intents."
                ),
            },
            "efficiency": self._score(
                efficiency,
                self._efficiency_reason(
                    len(model_steps),
                    duplicate_action_count,
                    tests_before_first_edit,
                    len(failed_model_actions),
                    len(cached_model_reads),
                ),
            ),
            "evidence": self._score(
                evidence,
                self._evidence_reason(
                    baseline_failed,
                    bool(model_investigation_steps),
                    bool(model_edit_steps),
                    final_exit is not None,
                    final_passed,
                ),
            ),
            "accuracy": self._score(
                accuracy,
                self._accuracy_reason(final_exit, bool(model_edit_steps), patch_present),
            ),
        }
        numeric_scores = [
            item["score"] for item in scores.values() if item["score"] is not None
        ]

        return {
            "scored": True,
            "reason": f"Scored {len(model_steps)} autonomous model actions.",
            "deterministic_metrics": {
                "model_step_count": len(model_steps),
                "model_edit_count": len(model_edit_steps),
                "first_model_edit_step": first_model_edit_step,
                "model_investigation_count": len(model_investigation_steps),
                "model_investigation_count_before_edit": (
                    len(model_investigation_steps_before_edit)
                    if first_model_edit_step is not None
                    else None
                ),
                "model_test_count_after_edit": len(model_tests_after_edit),
                "model_passing_test_count_after_edit": len(
                    model_passing_tests_after_edit
                ),
                "failed_model_action_count": len(failed_model_actions),
                "cached_model_read_count": len(cached_model_reads),
                "duplicate_investigation_count": duplicate_action_count,
                "tests_before_first_model_edit": tests_before_first_edit,
            },
            "scores": scores,
            "overall": round(sum(numeric_scores) / len(numeric_scores), 1),
        }

    @staticmethod
    def _actions(steps: list[dict], action_types: set[str]) -> list[dict]:
        return [
            step
            for step in steps
            if step.get("action", {}).get("type") in action_types
        ]

    def _successful_actions(
        self, steps: list[dict], action_types: set[str]
    ) -> list[dict]:
        return [
            step
            for step in self._actions(steps, action_types)
            if self._exit_code(step.get("observation")) == 0
        ]

    @staticmethod
    def _first_step(steps: list[dict]) -> int | None:
        return steps[0]["step"] if steps else None

    def _investigation_before_edit(
        self, steps: list[dict], first_edit_step: int | None
    ) -> list[dict]:
        if first_edit_step is None:
            return []

        return [
            step
            for step in self._actions(steps, INVESTIGATION_ACTIONS)
            if step["step"] < first_edit_step
        ]

    @staticmethod
    def _duplicate_investigations(steps: list[dict]) -> int:
        seen = set()
        duplicates = 0
        for step in steps:
            action = step.get("action", {})
            if action.get("type") not in {"read_file", "search_files"}:
                continue
            signature = (
                action["type"],
                json.dumps(action.get("args", {}), sort_keys=True, separators=(",", ":")),
            )
            if signature in seen:
                duplicates += 1
            else:
                seen.add(signature)
        return duplicates

    @staticmethod
    def _exit_code(observation: dict | None) -> int | None:
        return observation.get("exit_code") if observation else None

    @staticmethod
    def _score(score: int, reason: str) -> dict:
        return {"score": max(0, min(10, score)), "reason": reason}

    @staticmethod
    def _logical_flow_reason(
        baseline_failed: bool,
        investigated: bool,
        edited: bool,
        verified_after_edit: bool,
    ) -> str:
        stages = []
        if baseline_failed:
            stages.append("failure reproduced")
        if investigated:
            stages.append("investigation recorded")
        if edited:
            stages.append("model edit recorded")
        if verified_after_edit:
            stages.append("model reran tests after editing")
        return "; ".join(stages) or "No observable debugging stages were completed."

    @staticmethod
    def _efficiency_reason(
        model_steps: int,
        duplicates: int,
        tests_before_edit: int | None,
        failures: int,
        cached_reads: int,
    ) -> str:
        test_detail = (
            f"{tests_before_edit} tests before the first model edit"
            if tests_before_edit is not None
            else "no model edit occurred"
        )
        return (
            f"{model_steps} model steps; {duplicates} duplicate investigations; "
            f"{test_detail}; "
            f"{failures} failed actions; {cached_reads} cached reads."
        )

    @staticmethod
    def _evidence_reason(
        baseline_failed: bool,
        investigated: bool,
        edited: bool,
        final_captured: bool,
        final_passed: bool,
    ) -> str:
        evidence = []
        if baseline_failed:
            evidence.append("reproduced baseline failure")
        if investigated:
            evidence.append("model investigation")
        if edited:
            evidence.append("model edit")
        if final_captured:
            evidence.append("final test capture")
        if final_passed:
            evidence.append("passing final verification")
        return "; ".join(evidence) or "No debugging evidence was captured."

    @staticmethod
    def _accuracy_reason(
        final_exit: int | None, model_edited: bool, patch_present: bool
    ) -> str:
        if not model_edited:
            return "No successful model edit was recorded."
        if final_exit == 0 and patch_present:
            return "A model edit produced a captured patch and passing final tests."
        if final_exit == 0:
            return "Final tests passed after a model edit, but no patch was captured."
        if final_exit is not None:
            return "Final tests did not pass after the model edit."
        return "Final tests were not captured."
