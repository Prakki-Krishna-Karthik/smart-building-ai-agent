"""Run one complete Smart Building AI Agent demonstration.

Run from the repository root with::

    python scripts/demo_run.py

The demo discovers an IDF and EPW under ``data/input``. Set
``DEMO_IDF_PATH`` and ``DEMO_WEATHER_FILE`` when the files are elsewhere. The
script performs one baseline and one optimized simulation, while keeping all
generated files under ``data/output`` and never changing the original IDF.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import sys
from time import perf_counter
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.decision_engine import DecisionEngine  # noqa: E402
from src.config.config import settings  # noqa: E402
from src.controllers.optimization_controller import IDFRecommendationApplier  # noqa: E402
from src.llm.ollama_client import BuildingOptimization, OllamaClient  # noqa: E402
from src.simulator.energyplus import EnergyPlusRunner  # noqa: E402
from src.simulator.output_parser import BuildingState, EnergyPlusOutputParser  # noqa: E402
from src.utils.logging import configure_logging  # noqa: E402


LOGGER = logging.getLogger("demo_run")

DEMO_OUTPUT_REQUESTS = """
! AI_DEMO_OUTPUT_REQUESTS
OutputControl:Files,
  Yes,  !- CSV
  Yes,  !- MTR
  Yes,  !- ESO
  No,   !- EIO
  No,   !- Tabular
  No,   !- SQLite
  No,   !- JSON
  No,   !- AUDIT
  No,   !- Zone Sizing
  No,   !- System Sizing
  No,   !- DXF
  No,   !- BND
  No,   !- RDD
  No,   !- MDD
  No,   !- MTD
  Yes,  !- END
  No,   !- SHD
  No,   !- DFS
  No,   !- GLHE
  No,   !- DelightIn
  No,   !- DelightELdmp
  No,   !- DelightDFdmp
  No,   !- EDD
  No,   !- DBG
  No,   !- PerfLog
  No,   !- SLN
  No,   !- SCI
  No,   !- WRL
  No,   !- Screen
  No;   !- Tarcog

Output:Variable,*,Zone Mean Air Temperature,Hourly;
Output:Variable,*,Zone Air Relative Humidity,Hourly;
Output:Variable,*,Site Outdoor Air Drybulb Temperature,Hourly;
Output:Variable,*,Zone People Occupant Count,Hourly;
Output:Variable,*,Zone Thermal Comfort Fanger Model PMV,Hourly;
Output:Variable,*,Zone Thermal Comfort Fanger Model PPD,Hourly;
Output:Meter,Electricity:Facility,Hourly;
Output:Meter,Electricity:HVAC,Hourly;
Output:Meter,InteriorLights:Electricity,Hourly;
Output:Meter,InteriorEquipment:Electricity,Hourly;
"""


class DemoFailure(RuntimeError):
    """A user-facing failure with a named workflow stage."""

    def __init__(self, stage: str, message: str) -> None:
        super().__init__(message)
        self.stage = stage


def locate_inputs() -> tuple[Path, Path]:
    """Locate the demo IDF and weather file from environment or data/input."""
    idf = settings.demo_idf_path.resolve() if settings.demo_idf_path else None
    weather = settings.demo_weather_file.resolve() if settings.demo_weather_file else None

    if idf is None:
        candidates = sorted(settings.input_directory.rglob("*.idf")) if settings.input_directory.exists() else []
        if not candidates:
            raise DemoFailure(
                "input discovery",
                f"No EnergyPlus IDF was found under {settings.input_directory}. "
                "Add a sample model there or set DEMO_IDF_PATH.",
            )
        idf = candidates[0].resolve()
    if not idf.is_file():
        raise DemoFailure("input discovery", f"Configured IDF does not exist: {idf}")

    if weather is None:
        sibling_weather = sorted(idf.parent.glob("*.epw"))
        input_weather = sorted(settings.input_directory.rglob("*.epw")) if settings.input_directory.exists() else []
        weather_candidates = sibling_weather or input_weather
        if not weather_candidates:
            raise DemoFailure(
                "input discovery",
                f"No EnergyPlus weather file was found under {settings.input_directory}. "
                "Add an EPW file there or set DEMO_WEATHER_FILE.",
            )
        weather = weather_candidates[0].resolve()
    if not weather.is_file():
        raise DemoFailure("input discovery", f"Configured weather file does not exist: {weather}")
    return idf, weather


def prepare_demo_idf(source_idf: Path, target_idf: Path) -> Path:
    """Create a demo-only IDF copy with the CSV/metric output requests.

    The source example is never changed. Existing ``OutputControl:Files`` or
    the demo marker are respected to avoid adding duplicate output objects.
    The injected requests are intentionally limited to the fields consumed by
    ``EnergyPlusOutputParser``.
    """
    content = source_idf.read_text(encoding="utf-8")
    has_output_control = re.search(r"^\s*OutputControl:Files\s*,", content, re.IGNORECASE | re.MULTILINE)
    has_demo_requests = "! AI_DEMO_OUTPUT_REQUESTS" in content
    if not has_output_control and not has_demo_requests:
        target_idf.parent.mkdir(parents=True, exist_ok=True)
        target_idf.write_text(content.rstrip() + "\n\n" + DEMO_OUTPUT_REQUESTS.strip() + "\n", encoding="utf-8")
        LOGGER.info("Prepared demo IDF copy with native CSV output requests: %s", target_idf)
    else:
        target_idf.parent.mkdir(parents=True, exist_ok=True)
        target_idf.write_text(content, encoding="utf-8")
        LOGGER.info("Using existing output-control objects in demo IDF copy: %s", target_idf)
    return target_idf


def average(values: object) -> float | None:
    """Return an average for a numeric mapping's values."""
    if not isinstance(values, dict):
        return None
    numeric = [float(value) for value in values.values() if isinstance(value, (int, float))]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def print_building_state(label: str, state: BuildingState) -> None:
    """Print a compact, judge-friendly state summary."""
    print(f"\n{label} BuildingState")
    print(f"  Simulation Success: {state.simulation_success}")
    print(f"  Warnings: {len(state.simulation_warnings)}")
    print(f"  Errors: {len(state.simulation_errors)}")
    print(f"  Duration (s): {state.simulation_duration_seconds}")
    print(f"  Total Energy: {state.energy.total_electricity_consumption}")
    print(f"  HVAC Energy: {state.energy.hvac_electricity}")
    print(f"  Zones: {', '.join(state.zone_names) or 'none'}")
    print(f"  Occupied Zones: {', '.join(state.occupied_zones) or 'none'}")
    print(f"  Zone Temperatures: {state.thermal.zone_temperatures}")
    print(f"  PMV: {state.comfort.pmv}")
    print(f"  PPD: {state.comfort.ppd}")


def print_recommendations(optimization: BuildingOptimization) -> None:
    """Print validated recommendations returned by the decision engine."""
    print("\nOptimization Recommendations")
    print(f"  Reasoning: {optimization.reasoning}")
    if not optimization.actions:
        print("  No validated recommendations returned.")
        return
    for index, action in enumerate(optimization.actions, start=1):
        print(
            f"  {index}. {action.zone} | {action.parameter}: "
            f"{action.current} -> {action.recommended} | "
            f"Priority={action.priority} | Confidence={action.confidence_score} | "
            f"Estimated Savings={action.estimated_energy_savings_pct}% | "
            f"Comfort={action.estimated_comfort_impact}"
        )


def run_demo() -> int:
    """Execute the end-to-end demo and return a process exit code."""
    started = perf_counter()
    configure_logging(settings.log_directory, settings.log_level)
    print("Smart Building AI Agent - End-to-End Demo")
    print(f"Project: {PROJECT_ROOT}")

    try:
        print("\n[1/12] Verifying EnergyPlus installation...")
        runner = EnergyPlusRunner(logger=LOGGER)
        try:
            executable = runner.validate_installation()
        except Exception as exc:
            raise DemoFailure("EnergyPlus installation", str(exc)) from exc
        print(f"  EnergyPlus: {executable}")

        print("[2/12] Verifying Ollama service...")
        ollama = OllamaClient(logger=LOGGER)
        health = ollama.health_check()
        if not health.available:
            raise DemoFailure("Ollama health check", health.error or "Ollama is not available")
        print(f"  Ollama: running ({health.version or 'version unavailable'})")

        print("[3/12] Verifying configured Ollama model...")
        try:
            models = ollama.list_models()
            ollama.ensure_model_available()
        except Exception as exc:
            raise DemoFailure("Ollama model verification", str(exc)) from exc
        matching = [model.name for model in models if ollama.is_model_available((model,))]
        if not matching:
            matching = [model.name for model in ollama.list_models() if ollama.is_model_available((model,))]
        print(f"  Model: {ollama.model} ({matching[0] if matching else 'available after verification'})")

        print("[4/12] Locating sample EnergyPlus building...")
        idf_path, weather_file = locate_inputs()
        print(f"  IDF: {idf_path}")
        print(f"  Weather: {weather_file}")
        run_root = settings.output_directory / f"demo_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        parser = EnergyPlusOutputParser(logger=LOGGER)

        print("[5/12] Running baseline simulation...")
        prepared_baseline_idf = prepare_demo_idf(idf_path, run_root / "baseline" / "input.idf")
        baseline_result = runner.run_simulation(prepared_baseline_idf, weather_file, run_root / "baseline" / "simulation")
        print("[6/12] Parsing baseline outputs...")
        baseline_state = parser.parse(baseline_result.output_directory, return_code=baseline_result.return_code)
        print_building_state("Baseline", baseline_state)
        if baseline_state.simulation_success is False:
            raise DemoFailure("baseline simulation", "EnergyPlus completed with errors; see the baseline output files and logs")

        print("[7/12] Calling DecisionEngine...")
        engine = DecisionEngine(ollama, logger=LOGGER)
        optimization = engine.decide(baseline_state)
        print_recommendations(optimization)

        print("[8/12] Applying recommendations to a copied IDF...")
        applier = IDFRecommendationApplier(logger=LOGGER)
        modification = applier.apply(prepared_baseline_idf, run_root / "optimized" / "input.idf", optimization.actions)
        print(f"  Recommendations applied: {len(modification.applied)}")
        for reason in modification.unapplied_reasons:
            print(f"  Not applied: {reason}")

        print("[9/12] Running optimized simulation...")
        optimized_result = runner.run_simulation(modification.target_idf, weather_file, run_root / "optimized" / "simulation")
        print("[10/12] Parsing optimized outputs...")
        optimized_state = parser.parse(optimized_result.output_directory, return_code=optimized_result.return_code)
        print_building_state("Optimized", optimized_state)
        if optimized_state.simulation_success is False:
            raise DemoFailure("optimized simulation", "EnergyPlus completed with errors; see optimized output files and logs")

        print("[11/12] Comparing baseline vs optimized metrics...")
        baseline_energy = baseline_state.energy.total_electricity_consumption
        optimized_energy = optimized_state.energy.total_electricity_consumption
        savings = None
        if baseline_energy is not None and optimized_energy is not None and baseline_energy != 0:
            savings = round((baseline_energy - optimized_energy) / baseline_energy * 100.0, 2)
        print("[12/12] Demo complete")
        print("\nSummary")
        print(f"Baseline Energy: {baseline_energy}")
        print(f"Optimized Energy: {optimized_energy}")
        print(f"Energy Savings %: {savings}")
        print(f"PMV Before: {average(baseline_state.comfort.pmv)}")
        print(f"PMV After: {average(optimized_state.comfort.pmv)}")
        print(f"Recommendations Applied: {len(modification.applied)}")
        print(f"Total Execution Time: {perf_counter() - started:.2f}s")
        print(f"Artifacts: {run_root}")
        return 0
    except DemoFailure as exc:
        print(f"\nDEMO FAILED [{exc.stage}]: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\nDEMO FAILED [unexpected error]: {type(exc).__name__}: {exc}", file=sys.stderr)
        if os.getenv("DEMO_DEBUG", "").lower() in {"1", "true", "yes"}:
            traceback.print_exc()
        else:
            print("Set DEMO_DEBUG=1 for a traceback. Check data/logs/application.log for details.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(run_demo())
