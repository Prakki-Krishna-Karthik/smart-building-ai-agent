"""Regression tests for the final agent recommendation stage."""

from unittest.mock import Mock

import pytest

from src.agent.tool_loop import AgenticDecisionEngine
from src.llm.exceptions import OllamaResponseError
from src.llm.ollama_client import AgentToolSelection, BuildingOptimization
from src.simulator.output_parser import BuildingState, EnergyState, ThermalState


def test_empty_llm_recommendation_is_reported_as_an_error() -> None:
    """An empty LLM result must never become a deterministic recommendation."""
    client = Mock()
    client.select_tool.return_value = AgentToolSelection(
        action="recommendation",
        reasoning="No recommendation available.",
        recommendation=BuildingOptimization("No recommendation available.", ()),
    )
    dispatcher = Mock()
    dispatcher.registry.specifications.return_value = ()
    dispatcher.trusted_resources.return_value = {}

    state = BuildingState(
        simulation_success=True,
        energy=EnergyState(total_electricity_consumption=156636468279.3707),
        thermal=ThermalState(zone_temperatures={"SPACE1-1": 17.17}),
        zone_names=("SPACE1-1",),
        occupied_zones=("SPACE1-1",),
    )

    with pytest.raises(OllamaResponseError, match="no safe actions"):
        AgenticDecisionEngine(client, dispatcher).decide(state)
