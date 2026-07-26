"""Run the existing optimization pipeline against a deliberately inefficient copy.

Usage::

    python scripts/stress_test_run.py --stress-test

Or enable the mode through ``STRESS_TEST=true``. The configured source IDF is
never changed; all artificial inefficiencies are written beneath ``data/output``.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.tool_loop import AgenticDecisionEngine
from src.agent.tools import create_default_tool_dispatcher
from src.config.config import settings
from src.controllers.optimization_controller import OptimizationController
from src.llm.ollama_client import OllamaClient
from src.simulator.energyplus import EnergyPlusRunner
from src.simulator.output_parser import EnergyPlusOutputParser
from src.simulator.stress_test import prepare_stressed_idf
from src.utils.logging import configure_logging


def main() -> int:
    """Prepare the isolated stress model, run optimization, and print a report."""
    enabled = settings.stress_test or "--stress-test" in sys.argv[1:]
    if not enabled:
        print("Stress test is disabled. Set STRESS_TEST=true or pass --stress-test.")
        return 0
    if not settings.demo_idf_path or not settings.demo_weather_file:
        print("Stress test requires DEMO_IDF_PATH and DEMO_WEATHER_FILE.", file=sys.stderr)
        return 1
    configure_logging(settings.log_directory, settings.log_level)
    run_root = settings.output_directory / f"stress_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    stressed_idf, changes = prepare_stressed_idf(settings.demo_idf_path, run_root / "stressed_input.idf", enabled=True)
    print(f"Stressed copy: {stressed_idf}")
    for change in changes:
        print(f"Applied stress test: {change.schedule}: {change.old_value} -> {change.new_value} ({change.occurrences} values)")

    runner = EnergyPlusRunner()
    dispatcher = create_default_tool_dispatcher(runner)
    engine = AgenticDecisionEngine(OllamaClient(), dispatcher, max_steps=8)
    result = OptimizationController(
        runner,
        engine,
        parser=EnergyPlusOutputParser(),
        report_directory=run_root,
        tool_dispatcher=dispatcher,
    ).run(stressed_idf, settings.demo_weather_file, run_root, iterations=1)
    optimization_completed = any(
        record.kind == "optimization" and record.status == "completed"
        for record in result.iteration_history
    )
    print(json.dumps({
        "baseline_energy": result.baseline_energy,
        "optimized_energy": result.optimized_energy,
        "percentage_energy_savings": result.percentage_energy_savings,
        "ai_recommendations": [asdict(action) for action in result.recommendations_applied],
        "applied_recommendations": len(result.recommendations_applied),
        "optimization_completed": optimization_completed,
        "report_path": result.report_path,
        "csv_path": result.csv_summary_path,
        "error": result.error,
    }, indent=2, default=str))
    return 0 if result.error is None and optimization_completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
