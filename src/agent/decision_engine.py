"""Deterministic safety gate around LLM-generated building recommendations.

``DecisionEngine`` is the application boundary between parsed EnergyPlus state
and the LLM. It refuses to ask the model for recommendations when there is no
usable occupied-zone state, then validates every model action against explicit
building, temperature, actuator, and conflict rules. Only actions that pass
all checks are returned to callers. This module contains no EnergyPlus process
logic and no AI prompting logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Iterable

from src.config.config import settings
from src.llm.ollama_client import (
    BuildingOptimization,
    OllamaClient,
    OptimizationAction,
)
from src.simulator.output_parser import BuildingState


@dataclass(frozen=True)
class DecisionSafetyLimits:
    """Configurable bounds used by deterministic action validation."""

    min_temperature_c: float = settings.decision_min_temperature_c
    max_temperature_c: float = settings.decision_max_temperature_c
    min_fan_percent: float = settings.decision_min_fan_percent
    max_fan_percent: float = settings.decision_max_fan_percent
    max_temperature_change_c: float = 6.0


class DecisionEngine:
    """Validate and enrich LLM recommendations before they reach controllers.

    The engine is deliberately conservative. It only permits recognized HVAC
    parameters, known occupied zones, finite numeric values, safe temperature
    and fan ranges, and non-conflicting actions. LLM-provided priorities are
    ignored and replaced with deterministic priorities.
    """

    _TEMPERATURE_WORDS = ("temperature", "setpoint", "cooling", "heating")
    _FAN_WORDS = ("fan", "airflow", "air flow", "damper")
    _MODE_WORDS = ("mode", "operating state", "operation")
    _ALLOWED_MODES = {"auto", "automatic", "cooling", "heating", "off", "on", "standby", "occupied"}

    def __init__(
        self,
        ollama_client: OllamaClient,
        safety_limits: DecisionSafetyLimits | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the engine with an injectable Ollama client and limits."""
        self._ollama_client = ollama_client
        self._limits = safety_limits or DecisionSafetyLimits()
        self._logger = logger or logging.getLogger(__name__)
        self._validate_limits()

    def decide(self, building_state: BuildingState) -> BuildingOptimization:
        """Return only safe, validated recommendations for ``building_state``.

        Empty/unoccupied state and unsuccessful simulations are resolved to an
        empty typed response without calling the LLM. For usable state, the
        LLM receives the original ``BuildingState`` unchanged; every returned
        action then passes through deterministic validation and enrichment.
        """
        if not isinstance(building_state, BuildingState):
            raise TypeError("building_state must be a BuildingState instance")
        eligible_zones = self._eligible_zones(building_state)
        if building_state.simulation_success is False:
            self._logger.warning("Skipping decision: EnergyPlus simulation was unsuccessful")
            return BuildingOptimization("No recommendation: simulation was unsuccessful.", ())
        if not eligible_zones:
            self._logger.info("Skipping decision: no occupied zones are available")
            return BuildingOptimization("No recommendation: no occupied zones are available.", ())

        self._logger.info("Requesting LLM optimization for occupied zones: %s", sorted(eligible_zones))
        optimization = self._ollama_client.optimize_building(building_state)
        return self.validate_optimization(building_state, optimization, eligible_zones=eligible_zones)

    def validate_optimization(
        self,
        building_state: BuildingState,
        optimization: BuildingOptimization,
        eligible_zones: set[str] | None = None,
    ) -> BuildingOptimization:
        """Validate an already-produced optimization response.

        This public seam lets an agentic tool loop use the same deterministic
        safety gate after the LLM has gathered evidence through tools.
        """
        if not isinstance(optimization, BuildingOptimization):
            raise TypeError("optimization must be a BuildingOptimization instance")
        eligible_zones = eligible_zones if eligible_zones is not None else self._eligible_zones(building_state)
        valid = self._validate_actions(optimization.actions, building_state, eligible_zones)
        self._logger.info(
            "Decision validation complete: received=%d accepted=%d rejected=%d",
            len(optimization.actions),
            len(valid),
            len(optimization.actions) - len(valid),
        )
        return BuildingOptimization(reasoning=optimization.reasoning, actions=tuple(valid))

    def _validate_actions(
        self,
        actions: Iterable[OptimizationAction],
        state: BuildingState,
        eligible_zones: set[str],
    ) -> list[OptimizationAction]:
        """Validate actions, remove conflicts, then attach deterministic metadata."""
        candidates: list[OptimizationAction] = []
        for action in actions:
            validated = self._validate_action(action, state, eligible_zones)
            if validated is not None:
                candidates.append(validated)

        conflicts = self._find_conflicts(candidates)
        if conflicts:
            self._logger.warning("Rejecting %d conflicting recommendation(s)", len(conflicts))
        return [action for action in candidates if id(action) not in conflicts]

    def _validate_action(
        self,
        action: OptimizationAction,
        state: BuildingState,
        eligible_zones: set[str],
    ) -> OptimizationAction | None:
        """Return an enriched action or ``None`` when a safety rule fails."""
        zone = self._canonical_zone(action.zone, eligible_zones)
        parameter = action.parameter.strip()
        lower_parameter = parameter.lower()
        if zone is None:
            return self._reject(action, "unknown, empty, or unoccupied zone")
        if not parameter or not self._is_supported_parameter(lower_parameter):
            return self._reject(action, "unsupported or empty HVAC parameter")

        current = self._coerce_number(action.current)
        recommended = self._coerce_number(action.recommended)
        if self._is_temperature_parameter(lower_parameter):
            if current is None or recommended is None:
                return self._reject(action, "temperature actions require finite numeric values")
            if not self._within(current, self._limits.min_temperature_c, self._limits.max_temperature_c):
                return self._reject(action, "current temperature is outside safe bounds")
            if not self._within(recommended, self._limits.min_temperature_c, self._limits.max_temperature_c):
                return self._reject(action, "recommended temperature is outside safe bounds")
            if abs(recommended - current) > self._limits.max_temperature_change_c:
                return self._reject(action, "temperature change is unrealistically large")
        elif self._is_fan_parameter(lower_parameter):
            if recommended is None or not self._within(recommended, self._limits.min_fan_percent, self._limits.max_fan_percent):
                return self._reject(action, "fan/airflow recommendation is outside safe bounds")
            if current is not None and not self._within(current, self._limits.min_fan_percent, self._limits.max_fan_percent):
                return self._reject(action, "current fan/airflow value is outside safe bounds")
        elif self._is_mode_parameter(lower_parameter):
            mode = str(action.recommended).strip().lower()
            if mode not in self._ALLOWED_MODES:
                return self._reject(action, f"unsafe HVAC mode: {mode}")
            recommended = mode
        else:
            return self._reject(action, "parameter is not covered by a safety rule")

        savings = self._estimate_energy_savings(action, current, recommended, lower_parameter)
        comfort = self._estimate_comfort_impact(zone, state, recommended, lower_parameter)
        priority = self._assign_priority(current, recommended, lower_parameter, comfort)
        confidence = self._confidence_score(zone, state, action, current, recommended, savings)
        return OptimizationAction(
            zone=zone,
            parameter=parameter,
            current=self._preserve_number(current, action.current),
            recommended=self._preserve_number(recommended, action.recommended),
            priority=priority,
            expected_energy_change=action.expected_energy_change.strip(),
            confidence_score=confidence,
            estimated_energy_savings_pct=savings,
            llm_predicted_comfort_impact=comfort,
        )

    def _find_conflicts(self, actions: list[OptimizationAction]) -> set[int]:
        """Find duplicate parameter and contradictory setpoint recommendations."""
        conflicts: set[int] = set()
        grouped: dict[tuple[str, str], list[OptimizationAction]] = {}
        for action in actions:
            grouped.setdefault((action.zone.casefold(), self._parameter_family(action.parameter)), []).append(action)
        for group in grouped.values():
            values = {repr(action.recommended) for action in group}
            if len(values) > 1:
                conflicts.update(id(action) for action in group)
            elif len(group) > 1:
                conflicts.update(id(action) for action in group[1:])

        by_zone: dict[str, dict[str, OptimizationAction]] = {}
        for action in actions:
            lower = action.parameter.lower()
            if self._is_temperature_parameter(lower):
                zone_actions = by_zone.setdefault(action.zone.casefold(), {})
                if "cool" in lower:
                    zone_actions["cooling"] = action
                elif "heat" in lower:
                    zone_actions["heating"] = action
        for zone_actions in by_zone.values():
            cooling = self._coerce_number(zone_actions.get("cooling").recommended) if zone_actions.get("cooling") else None
            heating = self._coerce_number(zone_actions.get("heating").recommended) if zone_actions.get("heating") else None
            if cooling is not None and heating is not None and heating >= cooling:
                conflicts.update(id(zone_actions[key]) for key in ("cooling", "heating"))
        return conflicts

    def _estimate_energy_savings(
        self,
        action: OptimizationAction,
        current: float | str | None,
        recommended: float | str | None,
        parameter: str,
    ) -> float:
        """Estimate savings percentage from model hint or conservative heuristic."""
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*%", action.expected_energy_change or "")
        if match:
            return round(max(0.0, -float(match.group(1))), 2)
        if not isinstance(current, (int, float)) or not isinstance(recommended, (int, float)):
            return 0.0
        delta = float(recommended) - float(current)
        if "cool" in parameter:
            return round(max(0.0, delta * 2.0), 2)
        if "heat" in parameter:
            return round(max(0.0, -delta * 2.0), 2)
        if self._is_fan_parameter(parameter):
            return round(max(0.0, -delta * 0.1), 2)
        return 0.0

    def _estimate_comfort_impact(self, zone: str, state: BuildingState, recommended: object, parameter: str) -> str:
        """Estimate qualitative comfort impact without claiming a simulation result."""
        if not self._is_temperature_parameter(parameter):
            return "neutral"
        recommended_number = self._coerce_number(recommended)
        current_zone_temperature = state.thermal.zone_temperatures.get(zone)
        if recommended_number is None or current_zone_temperature is None:
            return "unknown"
        distance_before = abs(current_zone_temperature - 22.0)
        distance_after = abs(recommended_number - 22.0)
        if distance_after < distance_before:
            return "likely_improved"
        if distance_after > distance_before:
            return "possible_worsening"
        return "neutral"

    def _assign_priority(self, current: object, recommended: object, parameter: str, comfort: str) -> str:
        """Assign a deterministic Critical/High/Medium/Low priority."""
        current_number = self._coerce_number(current)
        recommended_number = self._coerce_number(recommended)
        delta = abs(recommended_number - current_number) if current_number is not None and recommended_number is not None else 0.0
        if delta >= 4.0 or comfort == "possible_worsening":
            return "Critical"
        if delta >= 2.0:
            return "High"
        if delta >= 0.5 or self._is_mode_parameter(parameter):
            return "Medium"
        return "Low"

    @staticmethod
    def _confidence_score(
        zone: str,
        state: BuildingState,
        action: OptimizationAction,
        current: object,
        recommended: object,
        savings: float,
    ) -> float:
        """Calculate a bounded deterministic confidence score."""
        score = 0.9
        if zone not in state.thermal.zone_temperatures:
            score -= 0.1
        if current is None or recommended is None:
            score -= 0.1
        if savings == 0.0 and not action.expected_energy_change:
            score -= 0.05
        return round(max(0.0, min(1.0, score)), 2)

    def _eligible_zones(self, state: BuildingState) -> set[str]:
        """Return canonical zones known to be occupied."""
        known = {zone.casefold(): zone for zone in state.zone_names if zone.strip()}
        return {known[zone.casefold()] for zone in state.occupied_zones if zone.casefold() in known}

    @staticmethod
    def _canonical_zone(zone: str, eligible_zones: set[str]) -> str | None:
        cleaned = zone.strip()
        if not cleaned:
            return None
        return next((known for known in eligible_zones if known.casefold() == cleaned.casefold()), None)

    def _is_supported_parameter(self, parameter: str) -> bool:
        return any(word in parameter for word in self._TEMPERATURE_WORDS + self._FAN_WORDS + self._MODE_WORDS)

    def _is_temperature_parameter(self, parameter: str) -> bool:
        return any(word in parameter for word in self._TEMPERATURE_WORDS)

    def _is_fan_parameter(self, parameter: str) -> bool:
        return any(word in parameter for word in self._FAN_WORDS)

    def _is_mode_parameter(self, parameter: str) -> bool:
        return any(word in parameter for word in self._MODE_WORDS)

    @staticmethod
    def _parameter_family(parameter: str) -> str:
        lower = parameter.casefold()
        if "cool" in lower:
            return "cooling_setpoint"
        if "heat" in lower:
            return "heating_setpoint"
        if "setpoint" in lower or "temperature" in lower:
            return "temperature_setpoint"
        if "fan" in lower or "airflow" in lower or "air flow" in lower or "damper" in lower:
            return "airflow"
        if "mode" in lower or "operating" in lower:
            return "mode"
        return lower

    @staticmethod
    def _coerce_number(value: object) -> float | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _preserve_number(number: float | str | None, original: object) -> float | int | str | None:
        if number is None:
            return original if isinstance(original, str) else None
        if isinstance(original, int) and not isinstance(original, bool):
            return int(number) if number.is_integer() else number
        return number

    @staticmethod
    def _within(value: float, minimum: float, maximum: float) -> bool:
        return minimum <= value <= maximum

    def _reject(self, action: OptimizationAction, reason: str) -> None:
        self._logger.warning(
            "Rejected unsafe LLM action zone=%r parameter=%r reason=%s",
            action.zone,
            action.parameter,
            reason,
        )
        return None

    def _validate_limits(self) -> None:
        if self._limits.min_temperature_c >= self._limits.max_temperature_c:
            raise ValueError("Minimum temperature must be below maximum temperature")
        if self._limits.min_fan_percent > self._limits.max_fan_percent:
            raise ValueError("Minimum fan value must not exceed maximum fan value")
