"""Ollama-driven custom-tool planning loop.

``AgenticDecisionEngine`` is the LLM-facing decision provider. It does not
grant the model direct process or filesystem access. The model can only select
tools registered with ``ToolDispatcher``; each result is returned to the model
as context, and the deterministic ``DecisionEngine`` validates the final
recommendation before it leaves this module.
"""

from __future__ import annotations

from dataclasses import asdict
import logging
from typing import Any

from src.agent.decision_engine import DecisionEngine
from src.agent.tools import ToolDispatcher
from src.llm.exceptions import OllamaResponseError
from src.llm.ollama_client import AgentToolSelection, BuildingOptimization, OllamaClient
from src.simulator.output_parser import BuildingState


class AgenticDecisionEngine:
    """Use Ollama tool selection followed by deterministic recommendation validation."""

    def __init__(
        self,
        ollama_client: OllamaClient,
        dispatcher: ToolDispatcher,
        max_steps: int = 8,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._ollama_client = ollama_client
        self._dispatcher = dispatcher
        self._validator = DecisionEngine(ollama_client, logger=logger)
        self._max_steps = max_steps
        self._logger = logger or logging.getLogger(__name__)

    def decide(self, building_state: BuildingState) -> BuildingOptimization:
        """Let Ollama select evidence/execution tools until it returns actions."""
        if not isinstance(building_state, BuildingState):
            raise TypeError("building_state must be a BuildingState instance")
        if building_state.simulation_success is False:
            return self._validator.validate_optimization(
                building_state,
                BuildingOptimization("No recommendation: simulation was unsuccessful.", ()),
            )
        state_payload = asdict(building_state)
        # Source paths are Python-owned resources and are intentionally not
        # exposed to the model. The dispatcher binds them after selection.
        state_payload["source_files"] = ()
        context: dict[str, Any] = {"building_state": state_payload}
        history: list[dict[str, Any]] = []
        for step in range(1, self._max_steps + 1):
            # Keep the tool budget bounded, but give the model an explicit
            # finalization signal on the last turn. This preserves LLM-owned
            # recommendation generation while preventing an endless evidence
            # gathering loop.
            latest_result = context.get("latest_tool_result")
            tool_failed = isinstance(latest_result, dict) and latest_result.get("success") is False
            tool_completed = isinstance(latest_result, dict) and latest_result.get("success") is True
            context["final_decision_required"] = step == self._max_steps or tool_failed or tool_completed
            try:
                selection = self._ollama_client.select_tool(
                    self._dispatcher.registry.specifications(),
                    context,
                    history,
                )
            except OllamaResponseError as exc:
                self._logger.error("Final LLM recommendation failed after bounded retries: %s", exc)
                raise
            if selection.action == "recommendation":
                if selection.recommendation is None:
                    raise ValueError("Agent returned a recommendation action without recommendations")
                validated = self._validator.validate_optimization(building_state, selection.recommendation)
                if not validated.actions and self._has_actionable_state(building_state):
                    raise OllamaResponseError(
                        "LLM recommendation was rejected: it contained no safe actions "
                        "for an actionable BuildingState"
                    )
                self._logger.info("Agentic decision completed after %d tool-loop step(s)", step)
                return validated

            trusted_resources = self._dispatcher.trusted_resources()
            if selection.tool_name == "compare_results" and not trusted_resources.get("optimized_state"):
                self._logger.warning(
                    "Ignoring compare_results selection because no trusted optimized_state exists; "
                    "requesting the final recommendation instead."
                )
                context["final_decision_required"] = True
                context["latest_tool_result"] = {
                    "success": False,
                    "error": "compare_results requires a trusted optimized_state, unavailable in this run.",
                }
                history.append({"selection": asdict(selection), "result": context["latest_tool_result"]})
                continue

            result = self._dispatcher.dispatch_intent(
                selection.tool_name or "",
                selection.intent,
                trusted_resources,
            )
            result_payload = result.to_prompt_payload(redact_paths=True)
            history.append({"selection": asdict(selection), "result": result_payload})
            context["latest_tool_result"] = result_payload
            self._logger.info("Agentic tool loop step=%d tool=%s success=%s", step, selection.tool_name, result.success)
            if result.success and "building_state" in result.data:
                context["building_state"] = dict(result.data["building_state"])
                context["building_state"]["source_files"] = ()
            if not result.success:
                context["latest_tool_error"] = result.error
        raise RuntimeError(f"Agent did not produce a recommendation within {self._max_steps} tool steps")

    @staticmethod
    def _has_actionable_state(building_state: BuildingState) -> bool:
        """Return whether the state supports at least one zone recommendation."""
        return any(
            zone.strip() and zone in building_state.thermal.zone_temperatures
            for zone in building_state.occupied_zones
        )
