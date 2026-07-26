"""Smoke tests for the initial project scaffold."""

import sys
from pathlib import Path

from src.config.config import settings
from src.models.schemas import ControlDecision, SimulationRequest
from src.simulator.energyplus import EnergyPlusRunner
from src.simulator.exceptions import EnergyPlusInputError, EnergyPlusOutputError


def test_settings_have_expected_project_root() -> None:
    """The central settings object should resolve to the repository root."""
    assert settings.project_root.name == "smart-building-ai-agent"


def test_domain_contracts_can_be_constructed() -> None:
    """Shared models should be usable without infrastructure dependencies."""
    request = SimulationRequest("building.idf", "data/output", "demo-run")
    decision = ControlDecision("demo-run", actions={"zone_setpoint": 22.0})
    assert request.run_id == decision.run_id


def test_runner_validates_a_configured_executable() -> None:
    """The runner can validate an explicitly configured cross-platform executable."""
    runner = EnergyPlusRunner(executable=sys.executable, timeout_seconds=10)
    assert runner.validate_installation() == Path(sys.executable).resolve()


def test_runner_rejects_missing_inputs(tmp_path: Path) -> None:
    """Simulation input errors should be explicit and actionable."""
    runner = EnergyPlusRunner(executable=sys.executable)
    try:
        runner.run_simulation(tmp_path / "missing.idf", tmp_path / "missing.epw", tmp_path / "out")
    except EnergyPlusInputError as exc:
        assert "IDF input" in str(exc)
    else:
        raise AssertionError("Expected EnergyPlusInputError")


def test_runner_collects_output_files(tmp_path: Path) -> None:
    """Output collection should include nested regular files in sorted order."""
    output = tmp_path / "output"
    (output / "nested").mkdir(parents=True)
    (output / "z.csv").write_text("z", encoding="utf-8")
    (output / "nested" / "a.eso").write_text("a", encoding="utf-8")
    files = EnergyPlusRunner(executable=sys.executable).collect_output_files(output)
    assert [path.name for path in files] == ["a.eso", "z.csv"]


def test_runner_rejects_missing_output_directory(tmp_path: Path) -> None:
    """Missing output directories should raise a dedicated output exception."""
    try:
        EnergyPlusRunner(executable=sys.executable).collect_output_files(tmp_path / "missing")
    except EnergyPlusOutputError:
        pass
    else:
        raise AssertionError("Expected EnergyPlusOutputError")
