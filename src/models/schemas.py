"""Shared domain contracts for the autonomous building control loop.

This module intentionally contains only lightweight data structures. It is the
stable boundary between EnergyPlus adapters, LLM decision makers, controllers,
and the dashboard. Business rules and optimization algorithms belong in their
respective application services, not in these transport/domain contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulator.output_parser import BuildingState


@dataclass(frozen=True)
class SimulationRequest:
    """Describe one simulation execution without implementing execution logic."""

    input_file: str
    output_directory: str
    run_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationResult:
    """Represent the process result and files produced by a simulation run."""

    run_id: str
    output_directory: str
    completed_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    return_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    output_files: tuple[str, ...] = ()
    building_state: BuildingState | None = None


@dataclass(frozen=True)
class ControlDecision:
    """Represent an LLM/controller proposal for a future control interval."""

    run_id: str
    actions: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float | None = None
