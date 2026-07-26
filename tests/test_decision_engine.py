"""Unit tests for deterministic AI decision validation."""

from unittest.mock import Mock

from src.agent.decision_engine import DecisionEngine, DecisionSafetyLimits
from src.llm.ollama_client import BuildingOptimization, OptimizationAction
from src.simulator.output_parser import BuildingState, ThermalState


def make_state(*, success: bool | None = True, occupied: tuple[str, ...] = ("Office A",)) -> BuildingState:
    """Create a realistic typed state for decision-engine tests."""
    return BuildingState(
        simulation_success=success,
        zone_names=("Office A", "Empty Storage"),
        occupied_zones=occupied,
        thermal=ThermalState(
            zone_temperatures={"Office A": 25.0, "Empty Storage": 20.0},
            outdoor_temperature=30.0,
            zone_humidity={"Office A": 55.0},
        ),
    )


def action(
    zone: str = "Office A",
    parameter: str = "Cooling Setpoint",
    current: object = 24,
    recommended: object = 22,
    expected: str = "-8%",
) -> OptimizationAction:
    """Build an untrusted model action for validation tests."""
    return OptimizationAction(zone, parameter, current, recommended, "high", expected)


def test_engine_validates_and_enriches_safe_actions() -> None:
    """Safe occupied-zone recommendations receive deterministic metadata."""
    client = Mock()
    client.optimize_building.return_value = BuildingOptimization("Lower cooling demand.", (action(),))
    engine = DecisionEngine(client)

    result = engine.decide(make_state())

    client.optimize_building.assert_called_once()
    assert len(result.actions) == 1
    validated = result.actions[0]
    assert validated.priority == "High"
    assert validated.confidence_score == 0.9
    assert validated.estimated_energy_savings_pct == 8.0
    assert validated.estimated_comfort_impact == "likely_improved"


def test_engine_rejects_unsafe_and_unusable_actions() -> None:
    """Empty, unknown, impossible, and unsupported actions are all filtered."""
    client = Mock()
    client.optimize_building.return_value = BuildingOptimization(
        "mixed recommendations",
        (
            action(zone=""),
            action(zone="Empty Storage"),
            action(recommended=31),
            action(parameter="Cooling Setpoint", current=24, recommended=40),
            action(parameter="HVAC Mode", current="auto", recommended="emergency"),
            action(parameter="Direct Valve Override", current=0, recommended=1),
            action(recommended=22),
        ),
    )
    engine = DecisionEngine(client)

    result = engine.decide(make_state())

    assert len(result.actions) == 1
    assert result.actions[0].zone == "Office A"
    assert result.actions[0].recommended == 22


def test_engine_rejects_conflicting_recommendations() -> None:
    """Conflicting duplicate and heating/cooling setpoints do not escape."""
    client = Mock()
    client.optimize_building.return_value = BuildingOptimization(
        "conflicting",
        (
            action(recommended=22),
            action(recommended=21),
            action(parameter="Heating Setpoint", current=20, recommended=23),
        ),
    )
    result = DecisionEngine(client).decide(make_state())

    assert result.actions == ()


def test_engine_skips_llm_for_empty_or_failed_state() -> None:
    """No recommendation is generated when there is no safe state to analyze."""
    client = Mock()
    engine = DecisionEngine(client)

    empty_result = engine.decide(make_state(occupied=()))
    failed_result = engine.decide(make_state(success=False))

    assert empty_result.actions == ()
    assert failed_result.actions == ()
    client.optimize_building.assert_not_called()


def test_engine_uses_configured_safety_limits() -> None:
    """Custom bounds are enforced without changing the engine implementation."""
    client = Mock()
    client.optimize_building.return_value = BuildingOptimization("test", (action(recommended=23),))
    engine = DecisionEngine(
        client,
        safety_limits=DecisionSafetyLimits(min_temperature_c=20, max_temperature_c=22),
    )

    result = engine.decide(make_state())

    assert result.actions == ()

