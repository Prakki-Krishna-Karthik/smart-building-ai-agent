"""Application facade for the autonomous optimization workflow.

The detailed workflow lives in :class:`OptimizationController`; this facade
keeps the ``agent`` package useful to callers that want an application-level
entry point without importing dashboard or transport code.
"""

from __future__ import annotations

from pathlib import Path

from src.controllers.optimization_controller import OptimizationController, OptimizationResult


class BuildingOptimizationAgent:
    """Coordinate closed-loop optimization through an injected controller."""

    def __init__(self, controller: OptimizationController) -> None:
        """Create an application facade around ``controller``."""
        self._controller = controller

    def run_iteration(
        self,
        idf_path: str | Path,
        weather_file: str | Path,
        output_directory: str | Path | None = None,
    ) -> OptimizationResult:
        """Run the configured closed loop using one controller iteration."""
        return self._controller.run(idf_path, weather_file, output_directory, iterations=1)

    def run(
        self,
        idf_path: str | Path,
        weather_file: str | Path,
        output_directory: str | Path | None = None,
        iterations: int = 5,
    ) -> OptimizationResult:
        """Run the complete closed loop for ``iterations`` iterations."""
        return self._controller.run(idf_path, weather_file, output_directory, iterations=iterations)
