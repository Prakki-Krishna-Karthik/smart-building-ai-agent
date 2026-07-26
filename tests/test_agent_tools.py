"""Tests for the custom agentic tool protocol and built-in tools."""

from pathlib import Path
from unittest.mock import Mock

from src.agent.tools import (
    Tool,
    ToolDispatcher,
    ToolRegistry,
    create_default_tool_dispatcher,
)
from src.models.schemas import SimulationResult
from src.simulator.output_parser import BuildingState, EnergyState


class EchoTool(Tool):
    """Small test tool used to verify registry and dispatcher contracts."""

    name = "echo"
    description = "Return the supplied value."
    input_schema = {"type": "object"}

    def execute(self, arguments):
        return {"value": arguments.get("value")}


def test_registry_rejects_duplicate_tools_and_dispatcher_audits_calls() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    try:
        registry.register(EchoTool())
        raise AssertionError("duplicate registration should fail")
    except ValueError:
        pass

    dispatcher = ToolDispatcher(registry)
    result = dispatcher.dispatch("echo", {"value": "ok"}, call_id="call-1")

    assert result.success is True
    assert result.data == {"value": "ok"}
    assert dispatcher.history()[0].tool_name == "echo"


def test_default_registry_exposes_required_tools(tmp_path: Path) -> None:
    runner = Mock()
    runner.validate_installation.return_value = Path("energyplus")
    runner.run_simulation.return_value = SimulationResult(
        "run-1",
        str(tmp_path / "simulation"),
        return_code=0,
        building_state=BuildingState(
            simulation_success=True,
            energy=EnergyState(total_electricity_consumption=10.0),
        ),
    )
    dispatcher = create_default_tool_dispatcher(runner)

    assert set(dispatcher.registry.names()) == {
        "validate_energyplus",
        "run_energyplus",
        "parse_outputs",
        "inspect_runtime_errors",
        "get_building_state",
        "apply_recommendations",
        "compare_results",
        "generate_report",
    }
    validation = dispatcher.dispatch("validate_energyplus")
    assert validation.success is True


def test_run_and_parse_tools_return_typed_building_state_payload(tmp_path: Path) -> None:
    output = tmp_path / "simulation"
    output.mkdir()
    idf = tmp_path / "building.idf"
    weather = tmp_path / "weather.epw"
    idf.write_text("Version,\n 23.1;\n", encoding="utf-8")
    weather.write_text("weather", encoding="utf-8")
    state = BuildingState(
        simulation_success=True,
        energy=EnergyState(total_electricity_consumption=42.0),
    )
    runner = Mock()
    runner.run_simulation.return_value = SimulationResult("run-1", str(output), return_code=0, building_state=state)
    dispatcher = create_default_tool_dispatcher(runner)

    run = dispatcher.dispatch(
        "run_energyplus",
        {"idf_path": str(idf), "weather_file": str(weather), "output_directory": str(output)},
    )
    parsed = dispatcher.dispatch("parse_outputs", {"output_directory": str(output), "return_code": 0})

    assert run.success is True
    assert parsed.success is True
    assert parsed.data["building_state"]["energy"]["total_electricity_consumption"] == 42.0


def test_intent_dispatch_rejects_unavailable_trusted_resources_before_execution(tmp_path: Path) -> None:
    runner = Mock()
    dispatcher = create_default_tool_dispatcher(runner)

    result = dispatcher.dispatch_intent(
        "run_energyplus",
        "run the trusted baseline model",
        {
            "idf_path": str(tmp_path / "missing.idf"),
            "weather_file": str(tmp_path / "missing.epw"),
            "output_directory": str(tmp_path / "new-output"),
        },
    )

    assert result.success is False
    assert result.error_code == "TRUSTED_RESOURCE_UNAVAILABLE"
    assert runner.run_simulation.called is False
