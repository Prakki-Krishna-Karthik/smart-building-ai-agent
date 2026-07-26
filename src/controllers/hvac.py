"""Safe HVAC control adapter for EnergyPlus input models.

The adapter deliberately writes to a destination copy and delegates the
EnergyPlus-specific text transformation to ``IDFRecommendationApplier``.
"""

from __future__ import annotations

from pathlib import Path

from src.llm.ollama_client import OptimizationAction
from src.models.schemas import ControlDecision, SimulationRequest
from src.controllers.optimization_controller import IDFModificationResult, IDFRecommendationApplier


class HVACController:
    """Apply already-validated HVAC actions without overwriting source IDFs."""

    def __init__(self, applier: IDFRecommendationApplier | None = None) -> None:
        """Initialize with the standard safe IDF applier."""
        self._applier = applier or IDFRecommendationApplier()

    def apply(self, decision: ControlDecision, request: SimulationRequest) -> SimulationRequest:
        """Apply typed control actions to a copied IDF and return a new request.

        ``ControlDecision.actions`` may contain ``OptimizationAction`` objects
        or serialized action dictionaries at this compatibility boundary.
        Unsupported values are rejected rather than silently applied.
        """
        actions: list[OptimizationAction] = []
        for action in decision.actions.values() if isinstance(decision.actions, dict) else ():
            if not isinstance(action, OptimizationAction):
                raise TypeError("HVACController requires OptimizationAction values")
            actions.append(action)
        source = Path(request.input_file)
        target = Path(request.output_directory) / f"{source.stem}_controlled{source.suffix}"
        modification: IDFModificationResult = self._applier.apply(source, target, actions)
        if modification.unapplied_reasons:
            raise ValueError("Unsafe or unapplied HVAC recommendations: " + "; ".join(modification.unapplied_reasons))
        return SimulationRequest(
            input_file=str(modification.target_idf),
            output_directory=request.output_directory,
            run_id=request.run_id,
            metadata={**request.metadata, "applied_actions": len(modification.applied)},
        )
