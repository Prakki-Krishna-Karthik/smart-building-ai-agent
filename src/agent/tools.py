"""Typed custom agentic tools for the building-optimization workflow.

This module implements a real in-process tool protocol. It is intentionally
not an MCP server: the LLM selects a named tool, ``ToolDispatcher`` validates
and executes it, and the typed result is returned to the LLM. Every call is
logged and retained in an audit history.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field, is_dataclass
import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from src.llm.ollama_client import OptimizationAction
from src.simulator.energyplus import EnergyPlusRunner
from src.simulator.output_parser import (
    BuildingState,
    ComfortState,
    EnergyPlusOutputParser,
    EnergyState,
    ThermalState,
)


@dataclass(frozen=True)
class ToolCall:
    """An immutable record of one requested tool invocation."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    requested_at: str


@dataclass(frozen=True)
class ToolResult:
    """Standard typed result returned by every tool execution."""

    call_id: str
    tool_name: str
    success: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None
    error_details: dict[str, Any] = field(default_factory=dict)
    completed_at: str = ""

    def to_prompt_payload(self, redact_paths: bool = False) -> dict[str, Any]:
        """Return a JSON-safe result for an LLM context window."""
        payload = asdict(self)
        return _redact_paths(payload) if redact_paths else payload


class Tool(ABC):
    """Interface implemented by every callable agent tool."""

    name: str
    description: str
    input_schema: dict[str, Any]

    @abstractmethod
    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Execute validated arguments and return a JSON-safe data object."""

    def specification(self) -> dict[str, Any]:
        """Return the public tool contract supplied to the LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def bind_intent(self, intent: str, trusted_arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Bind a model intent using only Python-owned arguments."""
        del intent
        properties = set(self.input_schema.get("properties", {})) if isinstance(self.input_schema, dict) else set()
        return {key: value for key, value in trusted_arguments.items() if key in properties}


class ToolRegistry:
    """Registry of uniquely named tools available to an agent."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool, rejecting duplicate names."""
        if not isinstance(tool, Tool):
            raise TypeError("tool must implement the Tool interface")
        if tool.name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Return a registered tool or raise a clear lookup error."""
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown agent tool: {name}") from exc

    def specifications(self) -> tuple[dict[str, Any], ...]:
        """Return deterministic tool specifications for model prompts."""
        return tuple(self._tools[name].specification() for name in sorted(self._tools))

    def names(self) -> tuple[str, ...]:
        """Return registered tool names."""
        return tuple(sorted(self._tools))


class ToolDispatcher:
    """Validate, execute, log, and audit calls to registered tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        logger: logging.Logger | None = None,
        trusted_resources: Mapping[str, Any] | None = None,
    ) -> None:
        self.registry = registry
        self._logger = logger or logging.getLogger(__name__)
        self._history: list[ToolCall] = []
        self._trusted_resources = trusted_resources if isinstance(trusted_resources, dict) else dict(trusted_resources or {})

    def dispatch(self, name: str, arguments: Mapping[str, Any] | None = None, call_id: str | None = None) -> ToolResult:
        """Execute one tool and convert failures into typed results."""
        arguments_dict = dict(arguments or {})
        identifier = call_id or f"tool-{len(self._history) + 1:04d}"
        call = ToolCall(identifier, name, arguments_dict, datetime.now(timezone.utc).isoformat())
        self._history.append(call)
        self._logger.info("Tool call started id=%s tool=%s arguments=%s", identifier, name, arguments_dict)
        started = datetime.now(timezone.utc)
        try:
            tool = self.registry.get(name)
            validation_error = self._validate_arguments(tool, arguments_dict, check_resources=True)
            if validation_error is not None:
                self._logger.warning("Tool call rejected before execution id=%s tool=%s: %s", identifier, name, validation_error["message"])
                return ToolResult(
                    identifier,
                    name,
                    False,
                    error=validation_error["message"],
                    error_code=validation_error["code"],
                    error_details=validation_error,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                )
            data = tool.execute(arguments_dict)
            if not isinstance(data, dict):
                raise TypeError("tool results must be dictionaries")
            result = ToolResult(identifier, name, True, _json_safe(data), completed_at=datetime.now(timezone.utc).isoformat())
            self._logger.info("Tool call completed id=%s tool=%s", identifier, name)
            return result
        except Exception as exc:  # tool failures are returned to the agent for recovery
            self._logger.exception("Tool call failed id=%s tool=%s after=%s", identifier, name, datetime.now(timezone.utc) - started)
            return ToolResult(
                identifier,
                name,
                False,
                error=f"{type(exc).__name__}: {exc}",
                error_code="TOOL_EXECUTION_FAILED",
                error_details={"exception_type": type(exc).__name__},
                completed_at=datetime.now(timezone.utc).isoformat(),
            )

    def dispatch_intent(self, name: str, intent: str, trusted_arguments: Mapping[str, Any] | None = None) -> ToolResult:
        """Execute a model-selected tool using only trusted Python arguments."""
        try:
            tool = self.registry.get(name)
        except KeyError as exc:
            return ToolResult(
                call_id=f"tool-{len(self._history) + 1:04d}",
                tool_name=name,
                success=False,
                error=str(exc),
                error_code="UNKNOWN_TOOL",
                error_details={"tool": name},
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
        arguments = tool.bind_intent(intent, trusted_arguments or self._trusted_resources)
        return self.dispatch(name, arguments)

    def update_trusted_resources(self, resources: Mapping[str, Any]) -> None:
        """Update Python-owned resources available for future intent calls."""
        self._trusted_resources.update(resources)

    def trusted_resources(self) -> dict[str, Any]:
        """Return trusted resources for internal binding, not model prompting."""
        return dict(self._trusted_resources)

    @staticmethod
    def _validate_arguments(tool: Tool, arguments: Mapping[str, Any], check_resources: bool = False) -> dict[str, Any] | None:
        """Validate required and unexpected arguments before tool execution."""
        schema = tool.input_schema if isinstance(tool.input_schema, dict) else {}
        required = schema.get("required", [])
        missing = [key for key in required if key not in arguments or arguments[key] in (None, "")]
        if missing:
            return {
                "code": "MISSING_REQUIRED_RESOURCE",
                "message": f"Missing required trusted argument(s): {', '.join(missing)}",
                "missing": missing,
                "tool": tool.name,
            }
        if schema.get("additionalProperties") is False:
            properties = set(schema.get("properties", {}))
            unexpected = sorted(set(arguments) - properties)
            if unexpected:
                return {
                    "code": "UNTRUSTED_ARGUMENT",
                    "message": f"Unexpected argument(s) are not accepted: {', '.join(unexpected)}",
                    "unexpected": unexpected,
                    "tool": tool.name,
                }
        if check_resources:
            file_keys = {"idf_path", "weather_file", "source_idf"}
            directory_keys = {"output_directory"} if tool.name in {"parse_outputs", "get_building_state", "inspect_runtime_errors"} else set()
            for key in sorted(file_keys & set(arguments)):
                path = Path(str(arguments[key])).expanduser()
                if not path.is_file():
                    return {
                        "code": "TRUSTED_RESOURCE_UNAVAILABLE",
                        "message": f"Trusted project resource is unavailable: {key}",
                        "resource": key,
                        "path": str(path),
                        "tool": tool.name,
                    }
            for key in sorted(directory_keys):
                path = Path(str(arguments[key])).expanduser()
                if not path.is_dir():
                    return {
                        "code": "TRUSTED_RESOURCE_UNAVAILABLE",
                        "message": f"Trusted project resource is unavailable: {key}",
                        "resource": key,
                        "path": str(path),
                        "tool": tool.name,
                    }
        return None

    def history(self) -> tuple[ToolCall, ...]:
        """Return an immutable audit history of requested calls."""
        return tuple(self._history)


@dataclass
class ToolContext:
    """Shared adapter dependencies and state cache for built-in tools."""

    runner: EnergyPlusRunner
    parser: EnergyPlusOutputParser
    applier: Any
    states: dict[str, BuildingState] = field(default_factory=dict)
    trusted_resources: dict[str, Any] = field(default_factory=dict)


def _state_payload(state: BuildingState) -> dict[str, Any]:
    return asdict(state)


def building_state_from_payload(payload: Mapping[str, Any]) -> BuildingState:
    """Reconstruct a typed BuildingState received from a tool result."""
    energy = payload.get("energy", {})
    thermal = payload.get("thermal", {})
    comfort = payload.get("comfort", {})
    return BuildingState(
        simulation_success=payload.get("simulation_success"),
        simulation_warnings=tuple(payload.get("simulation_warnings", ())),
        simulation_errors=tuple(payload.get("simulation_errors", ())),
        simulation_duration_seconds=payload.get("simulation_duration_seconds"),
        energy=EnergyState(**energy),
        thermal=ThermalState(**thermal),
        comfort=ComfortState(**comfort),
        zone_names=tuple(payload.get("zone_names", ())),
        occupied_zones=tuple(payload.get("occupied_zones", ())),
        hvac_operating_state=dict(payload.get("hvac_operating_state", {})),
        eso_available=bool(payload.get("eso_available", False)),
        eso_record_count=int(payload.get("eso_record_count", 0)),
        source_files=tuple(payload.get("source_files", ())),
    )


class _ContextTool(Tool):
    def __init__(self, context: ToolContext) -> None:
        self.context = context


class ValidateEnergyPlusTool(_ContextTool):
    name = "validate_energyplus"
    description = "Validate that the configured EnergyPlus executable is installed and runnable."
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        executable = self.context.runner.validate_installation()
        return {"executable": str(executable), "validated": True}


class RunEnergyPlusTool(_ContextTool):
    name = "run_energyplus"
    description = "Run EnergyPlus for an IDF and EPW into a new output directory."
    input_schema = {
        "type": "object",
        "properties": {
            "idf_path": {"type": "string"},
            "weather_file": {"type": "string"},
            "output_directory": {"type": "string"},
        },
        "required": ["idf_path", "weather_file", "output_directory"],
        "additionalProperties": False,
    }

    def bind_intent(self, intent: str, trusted_arguments: Mapping[str, Any]) -> dict[str, Any]:
        """Bind a run intent and allocate a fresh deterministic output folder."""
        del intent
        arguments = super().bind_intent("", trusted_arguments)
        output_directory = arguments.get("output_directory")
        if output_directory:
            base = Path(str(output_directory)).expanduser()
            if base.exists():
                stamp = datetime.now(timezone.utc).strftime("agent_tool_%Y%m%dT%H%M%S%fZ")
                arguments["output_directory"] = str(base / stamp)
        return arguments

    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        required = ("idf_path", "weather_file", "output_directory")
        if any(not str(arguments.get(key, "")).strip() for key in required):
            raise ValueError("idf_path, weather_file, and output_directory are required")
        requested_output_directory = str(arguments["output_directory"])
        result = self.context.runner.run_simulation(arguments["idf_path"], arguments["weather_file"], requested_output_directory)
        state = getattr(result, "building_state", None)
        output_directory = str(getattr(result, "output_directory", requested_output_directory))
        if isinstance(state, BuildingState):
            self.context.states[output_directory] = state
            self.context.states[requested_output_directory] = state
        self.context.trusted_resources.update(
            {
                "idf_path": str(arguments["idf_path"]),
                "weather_file": str(arguments["weather_file"]),
                "output_directory": output_directory,
                "return_code": getattr(result, "return_code", None),
            }
        )
        return {
            "run_id": str(getattr(result, "run_id", Path(requested_output_directory).name)),
            # The requested directory is the trusted workflow handle. A real
            # EnergyPlus runner writes there; retaining it also keeps injected
            # test adapters from leaking an unrelated path into the workflow.
            "output_directory": requested_output_directory,
            "actual_output_directory": output_directory,
            "return_code": getattr(result, "return_code", None),
            # Keep tool context bounded; complete process output remains in the
            # runner result and application logs for later inspection.
            "stdout_tail": str(getattr(result, "stdout", ""))[-4000:],
            "stderr_tail": str(getattr(result, "stderr", ""))[-4000:],
            "output_files": list(getattr(result, "output_files", ())),
        }


class ParseOutputsTool(_ContextTool):
    name = "parse_outputs"
    description = "Parse EnergyPlus output artifacts into a typed BuildingState."
    input_schema = {
        "type": "object",
        "properties": {"output_directory": {"type": "string"}, "return_code": {"type": ["integer", "null"]}},
        "required": ["output_directory"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        output_directory = str(arguments.get("output_directory", ""))
        if not output_directory:
            raise ValueError("output_directory is required")
        state = self.context.states.get(output_directory)
        if state is None:
            state = self.context.parser.parse(output_directory, return_code=arguments.get("return_code"))
            self.context.states[output_directory] = state
        state_payload = _state_payload(state)
        if "baseline_state" not in self.context.trusted_resources:
            self.context.trusted_resources["baseline_state"] = state_payload
        else:
            self.context.trusted_resources["optimized_state"] = state_payload
        return {"output_directory": output_directory, "building_state": state_payload}


class InspectRuntimeErrorsTool(_ContextTool):
    name = "inspect_runtime_errors"
    description = "Inspect EnergyPlus diagnostics and return warnings, severe errors, and fatal errors."
    input_schema = {
        "type": "object",
        "properties": {"output_directory": {"type": "string"}},
        "required": ["output_directory"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        output_directory = str(arguments.get("output_directory", ""))
        if not output_directory:
            raise ValueError("output_directory is required")
        path = Path(output_directory) / "eplusout.err"
        if not path.is_file():
            return {"output_directory": output_directory, "warnings": [], "errors": [], "available": False}
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        warnings = [line for line in lines if "** warning" in line.casefold()]
        errors = [line for line in lines if "** severe" in line.casefold() or "** fatal" in line.casefold()]
        return {"output_directory": output_directory, "warnings": warnings, "errors": errors, "available": True}


class GetBuildingStateTool(ParseOutputsTool):
    name = "get_building_state"
    description = "Return the latest structured BuildingState for an EnergyPlus output directory."


class ApplyRecommendationsTool(_ContextTool):
    name = "apply_recommendations"
    description = "Apply validated recommendations to a new IDF copy without overwriting the source."
    input_schema = {
        "type": "object",
        "properties": {
            "source_idf": {"type": "string"},
            "target_idf": {"type": "string"},
            "recommendations": {"type": "array"},
        },
        "required": ["source_idf", "target_idf", "recommendations"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        raw_actions = arguments.get("recommendations")
        if not isinstance(raw_actions, list):
            raise ValueError("recommendations must be an array")
        actions = [OptimizationAction(**item) for item in raw_actions if isinstance(item, dict)]
        modification = self.context.applier.apply(arguments["source_idf"], arguments["target_idf"], actions)
        return {
            "target_idf": str(modification.target_idf),
            "applied": [asdict(action) for action in modification.applied],
            "unapplied_reasons": list(modification.unapplied_reasons),
        }


class CompareResultsTool(Tool):
    name = "compare_results"
    description = "Compare baseline and optimized BuildingState energy and comfort metrics."
    input_schema = {
        "type": "object",
        "properties": {"baseline_state": {"type": "object"}, "optimized_state": {"type": "object"}},
        "required": ["baseline_state", "optimized_state"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        baseline = building_state_from_payload(arguments["baseline_state"])
        optimized = building_state_from_payload(arguments["optimized_state"])
        before = baseline.energy.total_electricity_consumption
        after = optimized.energy.total_electricity_consumption
        savings = None if before in (None, 0) or after is None else round((before - after) / before * 100, 2)
        baseline_pmv = _average(baseline.comfort.pmv.values())
        optimized_pmv = _average(optimized.comfort.pmv.values())
        baseline_ppd = _average(baseline.comfort.ppd.values())
        optimized_ppd = _average(optimized.comfort.ppd.values())
        return {
            "baseline_energy": before,
            "optimized_energy": after,
            "percentage_energy_savings": savings,
            "comfort_comparison": {
                "baseline_pmv": baseline_pmv,
                "optimized_pmv": optimized_pmv,
                "pmv_change": _difference(baseline_pmv, optimized_pmv),
                "baseline_ppd": baseline_ppd,
                "optimized_ppd": optimized_ppd,
                "ppd_change": _difference(baseline_ppd, optimized_ppd),
            },
        }


class GenerateReportTool(Tool):
    name = "generate_report"
    description = "Persist a JSON optimization report and optional CSV iteration summary."
    input_schema = {
        "type": "object",
        "properties": {
            "report_path": {"type": "string"},
            "csv_summary_path": {"type": "string"},
            "report": {"type": "object"},
        },
        "required": ["report_path", "report"],
        "additionalProperties": False,
    }

    def execute(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        report_path = Path(str(arguments["report_path"])).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = _json_safe(arguments["report"])
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        csv_path_value = arguments.get("csv_summary_path")
        csv_path = None
        if csv_path_value:
            csv_path = Path(str(csv_path_value)).expanduser().resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            records = report.get("iteration_history", []) if isinstance(report, dict) else []
            records = records if isinstance(records, list) else []
            fields = ["iteration", "kind", "status", "energy", "percentage_energy_savings", "recommendations_received", "recommendations_applied", "error", "execution_time_seconds"]
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for record in records:
                    row = {field: record.get(field) for field in fields if field != "recommendations_applied"}
                    applied = record.get("recommendations_applied", [])
                    row["recommendations_applied"] = len(applied) if isinstance(applied, list) else 0
                    writer.writerow(row)
        return {"report_path": str(report_path), "csv_summary_path": str(csv_path) if csv_path else None}


def create_default_tool_dispatcher(
    runner: EnergyPlusRunner,
    parser: EnergyPlusOutputParser | None = None,
    applier: Any | None = None,
    logger: logging.Logger | None = None,
) -> ToolDispatcher:
    """Create the production registry for all required custom agent tools."""
    if parser is None:
        parser = EnergyPlusOutputParser(logger=logger)
    if applier is None:
        from src.controllers.optimization_controller import IDFRecommendationApplier

        applier = IDFRecommendationApplier(logger=logger)
    from src.config.config import settings

    context = ToolContext(
        runner=runner,
        parser=parser,
        applier=applier,
        trusted_resources={
            "idf_path": str(settings.demo_idf_path) if settings.demo_idf_path else None,
            "weather_file": str(settings.demo_weather_file) if settings.demo_weather_file else None,
            "output_directory": str(settings.output_directory),
        },
    )
    registry = ToolRegistry()
    for tool in (
        ValidateEnergyPlusTool(context),
        RunEnergyPlusTool(context),
        ParseOutputsTool(context),
        InspectRuntimeErrorsTool(context),
        GetBuildingStateTool(context),
        ApplyRecommendationsTool(context),
        CompareResultsTool(),
        GenerateReportTool(),
    ):
        registry.register(tool)
    return ToolDispatcher(registry, logger=logger, trusted_resources=context.trusted_resources)


def _average(values: Any) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def _difference(before: float | None, after: float | None) -> float | None:
    return None if before is None or after is None else round(after - before, 4)


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _redact_paths(value: Any, key: str = "") -> Any:
    """Remove repository-specific paths before values enter LLM context."""
    path_keys = {
        "path", "paths", "idf_path", "weather_file", "output_directory",
        "target_idf", "report_path", "csv_summary_path", "source_files", "output_files",
    }
    if isinstance(value, Mapping):
        return {name: ("<trusted-resource>" if name in path_keys else _redact_paths(item, name)) for name, item in value.items()}
    if isinstance(value, list):
        return [_redact_paths(item, key) for item in value]
    if isinstance(value, tuple):
        return [_redact_paths(item, key) for item in value]
    if isinstance(value, str) and (":\\" in value or value.startswith("/") or ".idf" in value.lower() or ".epw" in value.lower()):
        return "<trusted-resource>"
    return value
