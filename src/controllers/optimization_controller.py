"""Closed-loop EnergyPlus optimization orchestration.

This module coordinates the simulation, parsing, decision, safe IDF-copy
mutation, and comparison workflow. It deliberately does not contain Streamlit
code or LLM prompting logic. All infrastructure is injectable, which keeps the
controller testable and allows the hackathon team to replace adapters later.

IDF mutation is intentionally explicit. A recommendation is applied only when
the copied IDF contains a matching marker such as::

    !- AI_CONTROL Zone=Office A | Parameter=Cooling Setpoint | Value=24

or an unambiguous standard ``ZoneControl:Thermostat`` →
``ThermostatSetpoint`` → ``Schedule:Compact`` reference. The original IDF is
never edited. Recommendations without an unambiguous target are recorded as
unapplied and are not guessed into arbitrary IDF objects.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import shutil
from time import perf_counter
from typing import Iterable, Protocol

from src.agent.decision_engine import DecisionSafetyLimits
from src.agent.tools import ToolDispatcher, building_state_from_payload, create_default_tool_dispatcher
from src.config.config import settings
from src.llm.ollama_client import BuildingOptimization, OllamaClient, OptimizationAction
from src.simulator.energyplus import EnergyPlusRunner
from src.simulator.output_parser import BuildingState, EnergyPlusOutputParser


@dataclass(frozen=True)
class ComfortComparison:
    """Baseline/optimized comparison of available comfort indicators."""

    baseline_pmv: float | None = None
    optimized_pmv: float | None = None
    pmv_change: float | None = None
    baseline_ppd: float | None = None
    optimized_ppd: float | None = None
    ppd_change: float | None = None


@dataclass(frozen=True)
class PredictionValidation:
    """Reporting-only comparison of LLM predictions and measured outcomes.

    Energy changes use consumption convention: negative means reduced energy
    consumption and positive means increased consumption.
    """

    llm_estimated_change_pct: float | None = None
    measured_energy_change_pct: float | None = None
    llm_estimated_energy_savings_pct: float | None = None
    measured_energy_savings_pct: float | None = None
    difference_percentage_points: float | None = None
    direction_match: bool | None = None
    magnitude_error_pct: float | None = None
    prediction_status: str = "Not Evaluated"
    reason: str = ""


@dataclass(frozen=True)
class ComfortValidation:
    """Reporting-only comparison of the pre-simulation prediction and outcome."""

    llm_prediction: str = "Not Available"
    measured_result: str = "Not Evaluated"
    agreement: bool = False
    reason: str = ""


@dataclass(frozen=True)
class IterationRecord:
    """Audit record for one baseline or optimization iteration."""

    iteration: int
    kind: str
    status: str
    input_idf: str
    output_directory: str
    energy: float | None = None
    percentage_energy_savings: float | None = None
    recommendations_received: int = 0
    recommendations_applied: tuple[OptimizationAction, ...] = ()
    error: str | None = None
    execution_time_seconds: float = 0.0


@dataclass(frozen=True)
class IDFModificationResult:
    """Result of copying an IDF and applying explicitly targeted actions."""

    target_idf: Path
    applied: tuple[OptimizationAction, ...] = ()
    unapplied_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizationResult:
    """Complete typed result and audit trail for one controller run."""

    baseline_energy: float | None = None
    optimized_energy: float | None = None
    percentage_energy_savings: float | None = None
    comfort_comparison: ComfortComparison = field(default_factory=ComfortComparison)
    measured_comfort_result: str = "Not Evaluated"
    comfort_validation: ComfortValidation = field(default_factory=ComfortValidation)
    prediction_validation: PredictionValidation = field(default_factory=PredictionValidation)
    confidence_score_explanation: str = (
        "Deterministic validation confidence based on supported zone evidence, "
        "finite safe values, and the estimated savings signal; it is not an Ollama probability."
    )
    recommendations_applied: tuple[OptimizationAction, ...] = ()
    iteration_history: tuple[IterationRecord, ...] = ()
    baseline_state: BuildingState | None = None
    optimized_state: BuildingState | None = None
    execution_time_seconds: float = 0.0
    error: str | None = None
    report_path: str | None = None
    csv_summary_path: str | None = None


class DecisionProvider(Protocol):
    """Minimal typed interface required from the decision engine."""

    def decide(self, building_state: BuildingState) -> BuildingOptimization:
        """Return validated recommendations for a building state."""


class IDFRecommendationApplier:
    """Copy IDF files and update only explicit safe control targets."""

    _MARKER = re.compile(
        r"(?P<prefix>!-\s*AI_CONTROL\s+Zone\s*=\s*(?P<zone>[^|]+)\|\s*"
        r"Parameter\s*=\s*(?P<parameter>[^|]+)\|\s*Value\s*=\s*)"
        r"(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+))",
        re.IGNORECASE,
    )

    def __init__(self, safety_limits: DecisionSafetyLimits | None = None, logger: logging.Logger | None = None) -> None:
        """Initialize the marker-based IDF applier."""
        self._limits = safety_limits or DecisionSafetyLimits()
        self._logger = logger or logging.getLogger(__name__)

    def apply(
        self,
        source_idf: str | Path,
        target_idf: str | Path,
        recommendations: Iterable[OptimizationAction],
    ) -> IDFModificationResult:
        """Copy ``source_idf`` and apply safe numeric setpoint markers.

        The target must not be the source and must not already exist. Only
        cooling/heating/temperature setpoints are currently mutated because
        their numeric semantics are unambiguous in the marker contract.
        """
        source = Path(source_idf).expanduser().resolve()
        target = Path(target_idf).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Source IDF does not exist: {source}")
        if source == target:
            raise ValueError("IDF target must be a copy, not the original source")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing IDF copy: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        content = target.read_text(encoding="utf-8")
        applied: list[OptimizationAction] = []
        unapplied: list[str] = []

        for recommendation in recommendations:
            if not self._safe_setpoint(recommendation):
                unapplied.append(f"{recommendation.zone}/{recommendation.parameter}: unsupported or unsafe mutation")
                continue
            content, count = self._replace_marker(content, recommendation)
            if not count:
                content, count = self._replace_thermostat_schedule(content, recommendation)
            if count:
                applied.append(recommendation)
                self._logger.info("Applied IDF recommendation to %s/%s", recommendation.zone, recommendation.parameter)
            else:
                has_thermostat = bool(
                    self._thermostat_schedule_names(
                        content,
                        recommendation.zone.strip().casefold(),
                        "cooling" if "cool" in recommendation.parameter.casefold() else "heating",
                    )
                )
                reason = (
                    "thermostat schedule rejected by safety/deadband validation"
                    if has_thermostat
                    else "no matching AI_CONTROL marker or thermostat reference"
                )
                unapplied.append(f"{recommendation.zone}/{recommendation.parameter}: {reason}")
                self._logger.warning("Could not safely apply %s/%s: %s", recommendation.zone, recommendation.parameter, reason)

        target.write_text(content, encoding="utf-8")
        return IDFModificationResult(target, tuple(applied), tuple(unapplied))

    def _replace_marker(self, content: str, recommendation: OptimizationAction) -> tuple[str, int]:
        """Replace the first matching marker value for one recommendation."""
        replacement_value = str(recommendation.recommended)
        requested_zone = self._canonical_zone_name(recommendation.zone)

        def replace(match: re.Match[str]) -> str:
            if (
                self._canonical_zone_name(match.group("zone")) == requested_zone
                and match.group("parameter").strip().casefold() == recommendation.parameter.strip().casefold()
            ):
                return f"{match.group('prefix')}{replacement_value}"
            return match.group(0)

        updated, count = self._MARKER.subn(replace, content, count=1)
        return updated, count

    def _replace_thermostat_schedule(self, content: str, recommendation: OptimizationAction) -> tuple[str, int]:
        """Resolve a zone thermostat reference and update its compact schedule.

        EnergyPlus examples commonly connect a ``ZoneControl:Thermostat`` to a
        ``ThermostatSetpoint`` object, which then references a
        ``Schedule:Compact``. This method follows those references instead of
        replacing unrelated numeric values. Shared schedules are intentionally
        updated once; the action remains safe but its scope is logged.
        """
        parameter = recommendation.parameter.casefold()
        target_kind = "cooling" if "cool" in parameter else "heating" if "heat" in parameter else None
        if target_kind is None:
            return content, 0
        zone = self._canonical_zone_name(recommendation.zone)
        schedule_names = self._thermostat_schedule_names(content, zone, target_kind)

        if not schedule_names:
            return content, 0
        opposite_kind = "heating" if target_kind == "cooling" else "cooling"
        opposite_names = self._thermostat_schedule_names(content, zone, opposite_kind)
        recommended_value = float(recommendation.recommended)  # type: ignore[arg-type]
        opposite_values = [
            value
            for schedule_name in opposite_names
            for value in self._schedule_values(content, schedule_name)
        ]
        if opposite_values:
            if target_kind == "cooling" and recommended_value <= max(opposite_values):
                self._logger.warning(
                    "Rejected %s/%s: cooling setpoint %.2f is not above heating schedule maximum %.2f",
                    recommendation.zone,
                    recommendation.parameter,
                    recommended_value,
                    max(opposite_values),
                )
                return content, 0
            if target_kind == "heating" and recommended_value >= min(opposite_values):
                self._logger.warning(
                    "Rejected %s/%s: heating setpoint %.2f is not below cooling schedule minimum %.2f",
                    recommendation.zone,
                    recommendation.parameter,
                    recommended_value,
                    min(opposite_values),
                )
                return content, 0
        updated = content
        replacements = 0
        for schedule_name in schedule_names:
            updated, count = self._replace_schedule_values(updated, schedule_name, str(recommendation.recommended))
            replacements += count
        if replacements:
            self._logger.info(
                "Updated %d schedule value(s) for %s/%s via thermostat reference(s)",
                replacements,
                recommendation.zone,
                recommendation.parameter,
            )
        return updated, replacements

    def _thermostat_schedule_names(self, content: str, zone: str, target_kind: str) -> set[str]:
        """Resolve schedule names for a zone's heating or cooling control."""
        thermostat_fields = self._idf_objects(content, "ZoneControl:Thermostat")
        setpoint_fields = self._idf_objects(content, "ThermostatSetpoint:SingleCooling")
        setpoint_fields += self._idf_objects(content, "ThermostatSetpoint:SingleHeating")
        setpoint_fields += self._idf_objects(content, "ThermostatSetpoint:DualSetpoint")
        schedule_names: set[str] = set()
        for fields in thermostat_fields:
            if len(fields) < 3 or fields[1].strip().casefold() != zone:
                continue
            for index in range(3, len(fields) - 1, 2):
                control_type = fields[index].strip().casefold()
                control_name = fields[index + 1].strip()
                if not control_name:
                    continue
                if target_kind == "cooling" and "cooling" not in control_type and "dualsetpoint" not in control_type:
                    continue
                if target_kind == "heating" and "heating" not in control_type and "dualsetpoint" not in control_type:
                    continue
                for setpoint in setpoint_fields:
                    if not setpoint or setpoint[0].strip().casefold() != control_name.casefold():
                        continue
                    # Dual setpoint objects expose heating and cooling schedule
                    # names in fields 1 and 2; single setpoints only expose one.
                    if len(setpoint) > 2:
                        schedule_names.add(setpoint[1 if target_kind == "heating" else 2].strip())
                    elif len(setpoint) > 1:
                        schedule_names.add(setpoint[1].strip())
        return schedule_names

    @staticmethod
    def _canonical_zone_name(zone: str) -> str:
        """Map occupant-output labels such as ``SPACE1-1 PEOPLE 1`` to zones."""
        return re.sub(r"\s+PEOPLE(?:\s+\d+)?$", "", zone.strip(), flags=re.IGNORECASE).casefold()

    @staticmethod
    def _idf_objects(content: str, object_type: str) -> list[list[str]]:
        """Extract IDF object fields with comments removed.

        EnergyPlus example files commonly place the object type and its
        opening comma on separate lines.  Match the declaration boundary
        rather than requiring both tokens to share one line.
        """
        objects = []
        line_pattern = re.compile(rf"^\s*{re.escape(object_type)}\s*,", re.IGNORECASE | re.MULTILINE)
        for match in line_pattern.finditer(content):
            # Object-type names can also appear in ZoneControl:Thermostat
            # fields. A declaration line has no inline field comment.
            if "!-" in match.group(0):
                continue
            semicolon = content.find(";", match.end())
            if semicolon < 0:
                continue
            body = re.sub(r"!.*", "", content[match.end() : semicolon])
            fields = [field.strip() for field in body.split(",")]
            objects.append(fields)
        return objects

    @staticmethod
    def _replace_schedule_values(content: str, schedule_name: str, replacement: str) -> tuple[str, int]:
        """Replace numeric values in ``Until: time,value`` fields of one schedule."""
        object_pattern = re.compile(
            rf"(?P<object>(?:^|\n)\s*Schedule:Compact\s*,\s*{re.escape(schedule_name)}\s*,.*?;)",
            re.IGNORECASE | re.DOTALL,
        )
        value_pattern = re.compile(r"(Until\s*:\s*[^,;]+\s*,\s*)(-?(?:\d+(?:\.\d*)?|\.\d+))", re.IGNORECASE)

        def update_object(match: re.Match[str]) -> str:
            nonlocal replacements
            updated, count = value_pattern.subn(rf"\g<1>{replacement}", match.group("object"))
            replacements += count
            return updated

        replacements = 0
        updated_content = object_pattern.sub(update_object, content, count=1)
        return updated_content, replacements

    @staticmethod
    def _schedule_values(content: str, schedule_name: str) -> list[float]:
        """Read numeric values from a named Schedule:Compact object."""
        object_pattern = re.compile(
            rf"(?:^|\n)\s*Schedule:Compact\s*,\s*{re.escape(schedule_name)}\s*,(?P<object>.*?);",
            re.IGNORECASE | re.DOTALL,
        )
        match = object_pattern.search(content)
        if not match:
            return []
        values = re.findall(r"Until\s*:\s*[^,;]+\s*,\s*(-?(?:\d+(?:\.\d*)?|\.\d+))", match.group("object"), re.IGNORECASE)
        return [float(value) for value in values]

    def _safe_setpoint(self, recommendation: OptimizationAction) -> bool:
        """Check that a recommendation is a finite numeric safe setpoint."""
        parameter = recommendation.parameter.casefold()
        if not any(word in parameter for word in ("setpoint", "temperature", "cooling", "heating")):
            return False
        try:
            value = float(recommendation.recommended)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return self._limits.min_temperature_c <= value <= self._limits.max_temperature_c


class OptimizationController:
    """Coordinate baseline simulation, iterative optimization, and reporting."""

    def __init__(
        self,
        runner: EnergyPlusRunner,
        decision_engine: DecisionProvider,
        parser: EnergyPlusOutputParser | None = None,
        idf_applier: IDFRecommendationApplier | None = None,
        report_directory: str | Path | None = None,
        logger: logging.Logger | None = None,
        tool_dispatcher: ToolDispatcher | None = None,
    ) -> None:
        """Initialize the controller with injectable workflow dependencies.

        EnergyPlus, parsing, IDF mutation, and report persistence are exposed
        to this controller through ``ToolDispatcher``. ``runner`` remains an
        injectable constructor argument for backward compatibility and is used
        only to build the default tool registry.
        """
        self._decision_engine = decision_engine
        self._parser = parser or EnergyPlusOutputParser()
        self._idf_applier = idf_applier or IDFRecommendationApplier()
        self._report_directory = Path(report_directory) if report_directory else settings.output_directory / "reports"
        self._logger = logger or logging.getLogger(__name__)
        self._tools = tool_dispatcher or create_default_tool_dispatcher(
            runner,
            parser=self._parser,
            applier=self._idf_applier,
            logger=self._logger,
        )

    def run(
        self,
        idf_path: str | Path,
        weather_file: str | Path,
        output_directory: str | Path | None = None,
        iterations: int = 5,
    ) -> OptimizationResult:
        """Execute the complete closed-loop workflow.

        Failed baseline validation/simulation returns a reportable failed
        result. Later iteration failures are recorded and the controller
        continues with the last successful state when possible.
        """
        if iterations < 1:
            raise ValueError("iterations must be at least 1")
        source_idf = Path(idf_path).expanduser().resolve()
        weather = Path(weather_file).expanduser().resolve()
        run_root = Path(output_directory).expanduser().resolve() if output_directory else settings.output_directory / f"optimization_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        run_root.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        history: list[IterationRecord] = []
        recommendations_applied: list[OptimizationAction] = []
        baseline_state: BuildingState | None = None
        optimized_state: BuildingState | None = None
        workflow_error: str | None = None

        self._logger.info("Starting closed-loop optimization idf=%s iterations=%d", source_idf, iterations)
        try:
            self._logger.info("Stage 1: validating EnergyPlus installation")
            self._execute_tool("validate_energyplus")
        except Exception as exc:  # adapter errors are persisted as run failures
            self._logger.exception("EnergyPlus installation validation failed")
            result = self._build_result(None, None, None, history, recommendations_applied, started, run_root, error=str(exc))
            self._write_reports(result)
            return result

        baseline_started = perf_counter()
        try:
            self._logger.info("Stage 2: running baseline simulation")
            baseline_output = run_root / "iteration_00_baseline"
            baseline_output.mkdir(parents=True, exist_ok=True)
            baseline_data = self._execute_tool(
                "run_energyplus",
                {
                    "idf_path": str(source_idf),
                    "weather_file": str(weather),
                    "output_directory": str(baseline_output),
                },
            )
            baseline_state = self._parse_tool_state(baseline_data)
            optimized_state = baseline_state
            history.append(self._record(0, "baseline", "completed", source_idf, run_root / "iteration_00_baseline", baseline_state, (), perf_counter() - baseline_started, recommendations_received=0))
            self._write_reports(self._build_result(baseline_state, optimized_state, None, history, recommendations_applied, started, run_root))
        except Exception as exc:
            self._logger.exception("Baseline simulation failed")
            history.append(self._record(0, "baseline", "failed", source_idf, run_root / "iteration_00_baseline", None, (), perf_counter() - baseline_started, str(exc)))
            result = self._build_result(None, None, None, history, recommendations_applied, started, run_root, error=str(exc))
            self._write_reports(result)
            return result

        current_idf = source_idf
        current_state = baseline_state
        for iteration in range(1, iterations + 1):
            iteration_started = perf_counter()
            iteration_output = run_root / f"iteration_{iteration:02d}"
            iteration_output.mkdir(parents=True, exist_ok=True)
            try:
                self._logger.info("Stage 4: requesting recommendations for iteration %d", iteration)
                optimization: BuildingOptimization = self._decision_engine.decide(current_state)
                self._logger.info("Received %d recommendation(s)", len(optimization.actions))
                copy_idf = iteration_output / "input.idf"
                modification = self._idf_applier.apply(current_idf, copy_idf, optimization.actions)
                if modification.unapplied_reasons:
                    self._logger.warning("Unapplied recommendations: %s", modification.unapplied_reasons)
                if not modification.applied:
                    history.append(self._record(iteration, "optimization", "no_applicable_recommendations", copy_idf, iteration_output, current_state, (), perf_counter() - iteration_started, recommendations_received=len(optimization.actions)))
                    self._write_reports(self._build_result(baseline_state, optimized_state, self._energy_savings(baseline_state, optimized_state), history, recommendations_applied, started, run_root))
                    break
                self._logger.info("Stage 6: running optimized simulation for iteration %d", iteration)
                simulation_output = iteration_output / "simulation"
                simulation_output.mkdir(parents=True, exist_ok=True)
                simulation_data = self._execute_tool(
                    "run_energyplus",
                    {
                        "idf_path": str(modification.target_idf),
                        "weather_file": str(weather),
                        "output_directory": str(simulation_output),
                    },
                )
                state = self._parse_tool_state(simulation_data)
                current_idf = modification.target_idf
                current_state = state
                optimized_state = state
                recommendations_applied.extend(modification.applied)
                history.append(self._record(iteration, "optimization", "completed", current_idf, iteration_output, state, modification.applied, perf_counter() - iteration_started, recommendations_received=len(optimization.actions)))
            except Exception as exc:
                self._logger.exception("Optimization iteration %d failed; continuing when possible", iteration)
                workflow_error = workflow_error or str(exc)
                history.append(self._record(iteration, "optimization", "failed", current_idf, iteration_output, current_state, (), perf_counter() - iteration_started, str(exc), recommendations_received=0))
            finally:
                partial = self._build_result(
                    baseline_state,
                    optimized_state,
                    self._energy_savings(baseline_state, optimized_state),
                    history,
                    recommendations_applied,
                    started,
                    run_root,
                    error=workflow_error,
                )
                self._write_reports(partial)

        result = self._build_result(
            baseline_state,
            optimized_state,
            self._energy_savings(baseline_state, optimized_state),
            history,
            recommendations_applied,
            started,
            run_root,
            error=workflow_error,
        )
        self._write_reports(result)
        self._logger.info("Closed-loop optimization complete in %.2fs", result.execution_time_seconds)
        return result

    def _state_from_result(self, result: object) -> BuildingState:
        """Use a compatibility result object to recover a typed state."""
        state = getattr(result, "building_state", None)
        if isinstance(state, BuildingState):
            return state
        output_directory = getattr(result, "output_directory", None)
        if not output_directory:
            raise ValueError("Simulation result did not provide a BuildingState or output directory")
        return self._parser.parse(output_directory, return_code=getattr(result, "return_code", None))

    def _execute_tool(self, name: str, arguments: dict[str, object] | None = None) -> dict[str, object]:
        """Dispatch a tool and raise a recoverable controller error on failure."""
        result = self._tools.dispatch(name, arguments or {})
        if not result.success:
            detail = result.error or "unknown tool failure"
            # Preserve the original exception message for existing controller
            # reports while retaining the tool name in the structured log.
            if detail.startswith("RuntimeError: "):
                detail = detail.removeprefix("RuntimeError: ")
            raise RuntimeError(detail)
        return result.data

    def _parse_tool_state(self, simulation_data: dict[str, object]) -> BuildingState:
        """Use the parser tool after every simulation, even when runner cached state."""
        output_directory = simulation_data.get("output_directory")
        if not isinstance(output_directory, str) or not output_directory:
            raise ValueError("run_energyplus did not return an output_directory")
        parsed = self._execute_tool(
            "parse_outputs",
            {"output_directory": output_directory, "return_code": simulation_data.get("return_code")},
        )
        payload = parsed.get("building_state")
        if not isinstance(payload, dict):
            raise ValueError("parse_outputs did not return a BuildingState payload")
        return building_state_from_payload(payload)

    def _build_result(
        self,
        baseline: BuildingState | None,
        optimized: BuildingState | None,
        savings: float | None,
        history: list[IterationRecord],
        applied: list[OptimizationAction],
        started: float,
        run_root: Path,
        error: str | None = None,
    ) -> OptimizationResult:
        """Construct a result snapshot suitable for intermediate reporting."""
        return OptimizationResult(
            baseline_energy=self._energy(baseline),
            optimized_energy=self._energy(optimized),
            percentage_energy_savings=savings,
            comfort_comparison=self._comfort_comparison(baseline, optimized),
            measured_comfort_result=self._measured_comfort_result(baseline, optimized),
            comfort_validation=self._comfort_validation(applied, baseline, optimized),
            prediction_validation=self._prediction_validation(applied, baseline, optimized, savings),
            recommendations_applied=tuple(applied),
            iteration_history=tuple(history),
            baseline_state=baseline,
            optimized_state=optimized,
            execution_time_seconds=perf_counter() - started,
            error=error,
            report_path=str(run_root / "optimization_report.json"),
            csv_summary_path=str(run_root / "optimization_summary.csv"),
        )

    def _write_reports(self, result: OptimizationResult) -> None:
        """Persist JSON and CSV artifacts through the report-generation tool."""
        report_path = Path(result.report_path or self._report_directory / "optimization_report.json")
        csv_path = Path(result.csv_summary_path or report_path.with_name("optimization_summary.csv"))
        self._execute_tool(
            "generate_report",
            {
                "report_path": str(report_path),
                "csv_summary_path": str(csv_path),
                "report": asdict(result),
            },
        )
        self._logger.info("Wrote optimization reports through tool: %s and %s", report_path, csv_path)

    @staticmethod
    def _energy(state: BuildingState | None) -> float | None:
        return state.energy.total_electricity_consumption if state else None

    def _measured_comfort_result(
        self,
        baseline: BuildingState | None,
        optimized: BuildingState | None,
    ) -> str:
        """Classify measured PMV/PPD change without changing control behavior."""
        if baseline is None or optimized is None or baseline is optimized:
            return "Not Evaluated"
        comparison = self._comfort_comparison(baseline, optimized)
        values = (
            comparison.baseline_pmv,
            comparison.optimized_pmv,
            comparison.baseline_ppd,
            comparison.optimized_ppd,
        )
        if any(value is None for value in values):
            return "Not Evaluated"
        pmv_delta = abs(comparison.optimized_pmv) - abs(comparison.baseline_pmv)  # type: ignore[arg-type]
        ppd_delta = comparison.optimized_ppd - comparison.baseline_ppd  # type: ignore[operator]
        if abs(pmv_delta) < 0.05 and abs(ppd_delta) < 1.0:
            return "Neutral"
        if abs(comparison.optimized_pmv) < abs(comparison.baseline_pmv) and comparison.optimized_ppd < comparison.baseline_ppd:  # type: ignore[arg-type]
            return "Improved"
        if abs(comparison.optimized_pmv) > abs(comparison.baseline_pmv) or comparison.optimized_ppd > comparison.baseline_ppd:  # type: ignore[arg-type]
            return "Degraded"
        return "Neutral"

    def _comfort_validation(
        self,
        applied: Iterable[OptimizationAction],
        baseline: BuildingState | None,
        optimized: BuildingState | None,
    ) -> ComfortValidation:
        """Report agreement between the pre-simulation estimate and measured comfort."""
        actions = list(applied)
        measured = self._measured_comfort_result(baseline, optimized)
        predictions = [
            str(action.llm_predicted_comfort_impact or "").strip().lower()
            for action in actions
        ]
        if any("improv" in value for value in predictions):
            prediction = "Improved"
        elif any("degrad" in value or "wors" in value for value in predictions):
            prediction = "Degraded"
        elif any("neutral" in value for value in predictions):
            prediction = "Neutral"
        else:
            prediction = "Not Available"

        agreement = (prediction == "Improved" and measured == "Improved") or (
            prediction == "Degraded" and measured == "Degraded"
        )
        comparison = self._comfort_comparison(baseline, optimized)
        if measured == "Improved":
            reason = "PPD decreased and PMV moved closer to zero."
        elif measured == "Degraded":
            reasons: list[str] = []
            if comparison.ppd_change is not None and comparison.ppd_change > 0:
                reasons.append("PPD increased")
            if (
                comparison.baseline_pmv is not None
                and comparison.optimized_pmv is not None
                and abs(comparison.optimized_pmv) > abs(comparison.baseline_pmv)
            ):
                reasons.append("PMV moved farther from zero")
            reason = " and ".join(reasons) + "." if reasons else "Comfort metrics degraded."
        elif measured == "Neutral":
            reason = "PMV and PPD changed by less than the neutral threshold or did not move consistently."
        else:
            reason = "PMV/PPD before-and-after data are unavailable."
        return ComfortValidation(
            llm_prediction=prediction,
            measured_result=measured,
            agreement=agreement,
            reason=reason,
        )

    def _prediction_validation(
        self,
        applied: Iterable[OptimizationAction],
        baseline: BuildingState | None,
        optimized: BuildingState | None,
        savings: float | None,
    ) -> PredictionValidation:
        """Compare LLM-declared change direction with measured EnergyPlus data."""
        actions = list(applied)
        if not actions or baseline is None or optimized is None:
            return PredictionValidation(reason="No applied recommendation and completed before/after result are available.")

        numeric_predictions: list[float] = []
        for action in actions:
            match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(action.expected_energy_change))
            if match:
                numeric_predictions.append(float(match.group(0)))
        if numeric_predictions and savings is not None:
            llm_change = round(sum(numeric_predictions) / len(numeric_predictions), 2)
            measured_change = round(-savings, 2)
            llm_savings = round(-llm_change, 2)
            measured_savings = round(-measured_change, 2)
            predicted_direction = 0 if abs(llm_change) < 1e-9 else (-1 if llm_change < 0 else 1)
            measured_direction = 0 if abs(measured_change) < 1e-9 else (-1 if measured_change < 0 else 1)
            direction_match = predicted_direction == measured_direction
            return PredictionValidation(
                llm_estimated_change_pct=llm_change,
                measured_energy_change_pct=measured_change,
                llm_estimated_energy_savings_pct=llm_savings,
                measured_energy_savings_pct=measured_savings,
                difference_percentage_points=round(abs(llm_savings - measured_savings), 2),
                direction_match=direction_match,
                magnitude_error_pct=round(abs(llm_change - measured_change), 2),
                prediction_status="Consistent" if direction_match else "Failed",
                reason=(
                    "LLM and EnergyPlus agree on the direction of energy change."
                    if direction_match
                    else "LLM predicted reduced energy, but EnergyPlus measured increased energy."
                    if llm_change < 0 and measured_change > 0
                    else "LLM and EnergyPlus reported opposite energy-change directions."
                ),
            )

        comparison = self._comfort_comparison(baseline, optimized)
        if comparison.baseline_pmv is None or comparison.optimized_pmv is None or comparison.baseline_ppd is None or comparison.optimized_ppd is None:
            return PredictionValidation(prediction_status="Not Evaluated", reason="Comfort-only prediction has no complete PMV/PPD before-and-after metrics.")
        comfort_match = (
            comparison.optimized_ppd <= comparison.baseline_ppd
            and abs(comparison.optimized_pmv) <= abs(comparison.baseline_pmv)
        )
        return PredictionValidation(
            direction_match=comfort_match,
            prediction_status="Consistent" if comfort_match else "Failed",
            reason=(
                "Comfort metrics improved or remained stable."
                if comfort_match
                else "Comfort-only prediction was not supported by the measured PMV/PPD results."
            ),
        )

    @staticmethod
    def _energy_savings(baseline: BuildingState | None, optimized: BuildingState | None) -> float | None:
        baseline_energy = OptimizationController._energy(baseline)
        optimized_energy = OptimizationController._energy(optimized)
        if baseline_energy is None or optimized_energy is None or baseline_energy == 0:
            return None
        return round((baseline_energy - optimized_energy) / baseline_energy * 100.0, 2)

    @staticmethod
    def _average(values: Iterable[float]) -> float | None:
        values = list(values)
        return round(sum(values) / len(values), 4) if values else None

    def _comfort_comparison(self, baseline: BuildingState | None, optimized: BuildingState | None) -> ComfortComparison:
        """Compare average PMV and PPD across available zones."""
        if not baseline or not optimized:
            return ComfortComparison()
        baseline_pmv = self._average(baseline.comfort.pmv.values())
        optimized_pmv = self._average(optimized.comfort.pmv.values())
        baseline_ppd = self._average(baseline.comfort.ppd.values())
        optimized_ppd = self._average(optimized.comfort.ppd.values())
        return ComfortComparison(
            baseline_pmv,
            optimized_pmv,
            self._difference(optimized_pmv, baseline_pmv),
            baseline_ppd,
            optimized_ppd,
            self._difference(optimized_ppd, baseline_ppd),
        )

    @staticmethod
    def _difference(left: float | None, right: float | None) -> float | None:
        return round(left - right, 4) if left is not None and right is not None else None

    def _record(
        self,
        iteration: int,
        kind: str,
        status: str,
        input_idf: Path,
        output_directory: Path,
        state: BuildingState | None,
        applied: Iterable[OptimizationAction],
        execution_time: float,
        error: str | None = None,
        recommendations_received: int = 0,
    ) -> IterationRecord:
        """Build a normalized iteration audit record."""
        return IterationRecord(
            iteration=iteration,
            kind=kind,
            status=status,
            input_idf=str(input_idf),
            output_directory=str(output_directory),
            energy=self._energy(state),
            recommendations_received=recommendations_received,
            recommendations_applied=tuple(applied),
            execution_time_seconds=round(execution_time, 4),
            error=error,
        )
