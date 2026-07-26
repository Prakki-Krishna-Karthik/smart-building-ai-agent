"""Unit tests for the closed-loop optimization controller."""

from pathlib import Path
from unittest.mock import Mock

from src.controllers.optimization_controller import (
    IDFRecommendationApplier,
    OptimizationController,
)
from src.llm.ollama_client import BuildingOptimization, OptimizationAction
from src.models.schemas import SimulationResult
from src.simulator.output_parser import BuildingState, ComfortState, EnergyState


def state(energy: float, pmv: float = 0.2, ppd: float = 6.0) -> BuildingState:
    """Create a compact successful state for controller tests."""
    return BuildingState(
        simulation_success=True,
        energy=EnergyState(total_electricity_consumption=energy),
        comfort=ComfortState(pmv={"Office A": pmv}, ppd={"Office A": ppd}),
        zone_names=("Office A",),
        occupied_zones=("Office A",),
    )


def recommendation() -> OptimizationAction:
    """Create a validated-style setpoint recommendation."""
    return OptimizationAction("Office A", "Cooling Setpoint", 24, 22, "High", "-8%")


def test_idf_applier_never_overwrites_source_and_applies_markers(tmp_path: Path) -> None:
    """Only the copied IDF marker changes; the original remains byte-for-byte intact."""
    source = tmp_path / "building.idf"
    source.write_text(
        "Version,\n 23.1;\n!- AI_CONTROL Zone=Office A | Parameter=Cooling Setpoint | Value=24\n",
        encoding="utf-8",
    )
    original = source.read_text(encoding="utf-8")
    target = tmp_path / "copy" / "building.idf"

    result = IDFRecommendationApplier().apply(source, target, (recommendation(),))

    assert result.applied == (recommendation(),)
    assert source.read_text(encoding="utf-8") == original
    assert "Value=22" in target.read_text(encoding="utf-8")


def test_idf_applier_rejects_unmarked_targets(tmp_path: Path) -> None:
    """The applier does not guess how to mutate arbitrary IDF objects."""
    source = tmp_path / "building.idf"
    source.write_text("Version,\n 23.1;\n", encoding="utf-8")

    result = IDFRecommendationApplier().apply(source, tmp_path / "copy.idf", (recommendation(),))

    assert result.applied == ()
    assert "no matching AI_CONTROL marker" in result.unapplied_reasons[0]


def test_idf_applier_updates_standard_thermostat_schedule_and_enforces_deadband(tmp_path: Path) -> None:
    """Standard thermostat references are mutable, but invalid deadbands are rejected."""
    source = tmp_path / "building.idf"
    source.write_text(
        """ZoneControl:Thermostat,
  Office Control,
  Office A,
  Control Schedule,
  ThermostatSetpoint:SingleCooling,
  CoolingSetpoint,
  ThermostatSetpoint:SingleHeating,
  HeatingSetpoint,
  ThermostatSetpoint:DualSetpoint,
  DualSetpoint;

ThermostatSetpoint:SingleCooling,
  CoolingSetpoint,
  CoolingSchedule;

ThermostatSetpoint:SingleHeating,
  HeatingSetpoint,
  HeatingSchedule;

ThermostatSetpoint:DualSetpoint,
  DualSetpoint,
  HeatingSchedule,
  CoolingSchedule;

Schedule:Compact,
  CoolingSchedule,
  Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,24.0;

Schedule:Compact,
  HeatingSchedule,
  Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,20.0;
""",
        encoding="utf-8",
    )
    safe_target = tmp_path / "safe.idf"
    safe = IDFRecommendationApplier().apply(
        source,
        safe_target,
        (OptimizationAction("Office A", "Cooling Setpoint", 24, 23, "High", "-2%"),),
    )
    unsafe_target = tmp_path / "unsafe.idf"
    unsafe = IDFRecommendationApplier().apply(
        source,
        unsafe_target,
        (OptimizationAction("Office A", "Cooling Setpoint", 24, 19, "High", "-2%"),),
    )

    assert len(safe.applied) == 1
    assert "Until: 24:00,23" in safe_target.read_text(encoding="utf-8")
    assert unsafe.applied == ()
    assert "deadband" in unsafe.unapplied_reasons[0]


def test_idf_applier_updates_dual_setpoint_schedule(tmp_path: Path) -> None:
    """Dual setpoint objects must resolve the correct heating and cooling schedules."""
    source = tmp_path / "building.idf"
    source.write_text(
        """ZoneControl:Thermostat,
  Office Control,
  Office A,
  Control Schedule,
  ThermostatSetpoint:DualSetpoint,
  DualSetpoint;

ThermostatSetpoint:DualSetpoint,
  DualSetpoint,
  HeatingSchedule,
  CoolingSchedule;

Schedule:Compact,
  HeatingSchedule,
  Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,18.0;

Schedule:Compact,
  CoolingSchedule,
  Temperature,
  Through: 12/31,
  For: AllDays,
  Until: 24:00,24.0;
""",
        encoding="utf-8",
    )

    result = IDFRecommendationApplier().apply(
        source,
        tmp_path / "copy.idf",
        (OptimizationAction("Office A", "Cooling Setpoint", 24, 25, "High", "-2%"),),
    )

    assert len(result.applied) == 1
    copied = (tmp_path / "copy.idf").read_text(encoding="utf-8")
    assert "HeatingSchedule" in copied
    assert "Until: 24:00,25" in copied
    assert "Until: 24:00,18" in copied


def test_controller_runs_iterations_and_writes_reports(tmp_path: Path) -> None:
    """The controller should orchestrate, compare, and persist a closed-loop run."""
    source = tmp_path / "building.idf"
    source.write_text(
        "!- AI_CONTROL Zone=Office A | Parameter=Cooling Setpoint | Value=24\n",
        encoding="utf-8",
    )
    weather = tmp_path / "weather.epw"
    weather.write_text("weather", encoding="utf-8")
    runner = Mock()
    runner.validate_installation.return_value = Path("energyplus")
    runner.run_simulation.side_effect = [
        SimulationResult("baseline", str(tmp_path / "baseline"), building_state=state(100.0)),
        SimulationResult("optimized", str(tmp_path / "optimized"), building_state=state(90.0, 0.1, 4.0)),
    ]
    decision_engine = Mock()
    decision_engine.decide.return_value = BuildingOptimization("Reduce cooling demand.", (recommendation(),))

    result = OptimizationController(
        runner,
        decision_engine,
        report_directory=tmp_path / "reports",
    ).run(source, weather, output_directory=tmp_path / "run", iterations=1)

    assert result.baseline_energy == 100.0
    assert result.optimized_energy == 90.0
    assert result.percentage_energy_savings == 10.0
    assert len(result.recommendations_applied) == 1
    assert result.comfort_comparison.ppd_change == -2.0
    assert Path(result.report_path).is_file()
    assert Path(result.csv_summary_path).is_file()
    assert len(result.iteration_history) == 2
    assert source.read_text(encoding="utf-8").endswith("Value=24\n")
    assert "Value=22" in (tmp_path / "run" / "iteration_01" / "input.idf").read_text(encoding="utf-8")


def test_controller_recovers_from_failed_optimization_iteration(tmp_path: Path) -> None:
    """A failed later simulation is recorded without losing the baseline result."""
    source = tmp_path / "building.idf"
    source.write_text("!- AI_CONTROL Zone=Office A | Parameter=Cooling Setpoint | Value=24\n", encoding="utf-8")
    weather = tmp_path / "weather.epw"
    weather.write_text("weather", encoding="utf-8")
    runner = Mock()
    runner.validate_installation.return_value = Path("energyplus")
    runner.run_simulation.side_effect = [
        SimulationResult("baseline", "baseline", building_state=state(100.0)),
        RuntimeError("simulation failed"),
    ]
    engine = Mock()
    engine.decide.return_value = BuildingOptimization("try", (recommendation(),))

    result = OptimizationController(runner, engine, report_directory=tmp_path / "reports").run(
        source, weather, output_directory=tmp_path / "run", iterations=1
    )

    assert result.baseline_energy == 100.0
    assert result.optimized_energy == 100.0
    assert result.iteration_history[-1].status == "failed"
    assert result.iteration_history[-1].error == "simulation failed"
