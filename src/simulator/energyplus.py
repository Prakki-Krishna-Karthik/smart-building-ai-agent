"""Cross-platform EnergyPlus process integration.

``EnergyPlusRunner`` is the infrastructure boundary for EnergyPlus. It is
responsible for locating and validating the executable, starting a simulation,
capturing process output, exposing status, and collecting generated artifacts.
It deliberately does not parse EnergyPlus output data or make control/AI
decisions; those responsibilities belong to higher-level application services.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
import re
import shutil
import subprocess
from typing import Literal, Sequence

from src.config.config import settings
from src.models.schemas import SimulationResult
from src.simulator.output_parser import BuildingState, EnergyPlusOutputParser
from src.simulator.exceptions import (
    EnergyPlusInputError,
    EnergyPlusInstallationError,
    EnergyPlusOutputError,
    EnergyPlusSimulationError,
    EnergyPlusSimulationTimeout,
)


SimulationState = Literal["not_started", "running", "completed", "failed", "timed_out"]


_PARSER_OUTPUT_REQUESTS = """
! AI_RUNNER_OUTPUT_REQUESTS
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

_COMFORT_MODEL_MARKER = "! AI_RUNNER_FANGER_COMFORT"
_COMFORT_SCHEDULES = """
! AI_RUNNER_FANGER_COMFORT
Schedule:Constant,
  AI Comfort Work Efficiency,  !- Name
  Any Number,                   !- Schedule Type Limits Name
  0.0;                          !- Hourly Value

Schedule:Constant,
  AI Comfort Clothing,          !- Name
  Any Number,                   !- Schedule Type Limits Name
  0.5;                          !- Hourly Value

Schedule:Constant,
  AI Comfort Air Velocity,      !- Name
  Any Number,                   !- Schedule Type Limits Name
  0.1;                          !- Hourly Value
"""

_FANGER_FIELDS = """
,
No,
EnclosureAveraged,
,
AI Comfort Work Efficiency,
ClothingInsulationSchedule,
,
AI Comfort Clothing,
AI Comfort Air Velocity,
Fanger;
"""


@dataclass(frozen=True)
class SimulationStatus:
    """A point-in-time view of the current EnergyPlus process."""

    state: SimulationState
    return_code: int | None = None
    pid: int | None = None
    stdout: str = ""
    stderr: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    output_directory: Path | None = None


class EnergyPlusRunner:
    """Run EnergyPlus simulations through a controlled subprocess boundary.

    The executable can be supplied directly, configured with
    ``ENERGYPLUS_EXECUTABLE``, or discovered from ``PATH``. An installation
    directory can be supplied with ``ENERGYPLUS_INSTALLATION_PATH``; this may
    point either to the executable itself or to a directory containing it.
    No operating-system-specific paths are embedded in this class.

    A runner instance is intended to execute one simulation at a time. Calling
    ``run_simulation`` while another process is active raises an error so output
    and status cannot be accidentally mixed between runs.
    """

    def __init__(
        self,
        executable: str | Path | None = None,
        installation_path: str | Path | None = None,
        timeout_seconds: int | None = None,
        logger: logging.Logger | None = None,
        output_parser: EnergyPlusOutputParser | None = None,
    ) -> None:
        """Initialize a runner with optional overrides for application settings."""
        self._configured_executable = Path(executable) if executable else None
        self._installation_path = (
            Path(installation_path)
            if installation_path
            else settings.energyplus_installation_path
        )
        self._executable: Path | None = None
        self._timeout_seconds = timeout_seconds or settings.simulation_timeout_seconds
        self._logger = logger or logging.getLogger(__name__)
        self._output_parser = output_parser or EnergyPlusOutputParser(self._logger)
        self._process: subprocess.Popen[str] | None = None
        self._status = SimulationStatus(state="not_started")

    def locate_energyplus(self) -> Path:
        """Locate the EnergyPlus executable from configuration or ``PATH``.

        Returns:
            The resolved executable path.

        Raises:
            EnergyPlusInstallationError: If no candidate can be found.
        """
        configured = self._configured_executable
        installation = self._installation_path
        candidates: list[Path] = []

        if configured:
            candidates.append(configured)
        elif installation:
            if installation.is_file():
                candidates.append(installation)
            else:
                executable_name = settings.energyplus_executable
                candidates.extend(
                    installation / name
                    for name in (executable_name, "energyplus", "energyplus.exe")
                )

        for candidate in candidates:
            if candidate.is_file() and self._is_executable_candidate(candidate):
                resolved = candidate.resolve()
                self._logger.info("EnergyPlus executable found at %s", resolved)
                self._executable = resolved
                return resolved

        command = settings.energyplus_executable
        located = shutil.which(command)
        if located:
            resolved = Path(located).resolve()
            self._logger.info("EnergyPlus executable found on PATH at %s", resolved)
            self._executable = resolved
            return resolved

        searched = ", ".join(str(path) for path in candidates) or command
        raise EnergyPlusInstallationError(
            f"EnergyPlus executable was not found. Searched: {searched}. "
            "Set ENERGYPLUS_INSTALLATION_PATH or ENERGYPLUS_EXECUTABLE."
        )

    @staticmethod
    def _is_executable_candidate(candidate: Path) -> bool:
        """Return whether a file is executable on the current platform."""
        return candidate.suffix.lower() == ".exe" or bool(shutil.which(str(candidate)))

    def validate_installation(self) -> Path:
        """Locate EnergyPlus and verify it responds to its version command.

        Returns:
            The validated executable path.

        Raises:
            EnergyPlusInstallationError: If discovery or version validation fails.
        """
        executable = self.locate_energyplus()
        self._logger.info("Validating EnergyPlus installation using %s", executable)
        try:
            completed = subprocess.run(
                [str(executable), "--version"],
                capture_output=True,
                text=True,
                check=False,
                timeout=min(self._timeout_seconds, 30),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EnergyPlusInstallationError(
                f"EnergyPlus was found at {executable} but could not be validated: {exc}"
            ) from exc

        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic output"
            raise EnergyPlusInstallationError(
                f"EnergyPlus version check failed with exit code "
                f"{completed.returncode}: {detail}"
            )
        self._logger.info("EnergyPlus installation validated: %s", completed.stdout.strip())
        return executable

    def run_simulation(
        self,
        idf_path: str | Path,
        weather_file: str | Path,
        output_directory: str | Path,
    ) -> SimulationResult:
        """Run one EnergyPlus simulation and return its captured result.

        Args:
            idf_path: Path to the EnergyPlus input data file.
            weather_file: Path to the EnergyPlus weather file.
            output_directory: Directory where EnergyPlus writes artifacts.

        Raises:
            EnergyPlusInputError: If input files or output directory are invalid.
            EnergyPlusSimulationError: If the process cannot start or exits nonzero.
            EnergyPlusSimulationTimeout: If execution exceeds the configured timeout.
        """
        if self._process and self._process.poll() is None:
            raise EnergyPlusSimulationError("A simulation is already running")

        idf = self._require_file(idf_path, "IDF input")
        weather = self._require_file(weather_file, "weather input")
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        simulation_idf = self._prepare_output_requests(idf, output)
        executable = self.validate_installation()
        command: Sequence[str] = [str(executable), "-w", str(weather), "-d", str(output), str(simulation_idf)]
        started = datetime.now(timezone.utc)
        self._logger.info("Starting EnergyPlus simulation: %s", " ".join(command))
        self._process = None
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=output,
            )
            self._status = SimulationStatus("running", pid=self._process.pid, started_at=started, output_directory=output)
            stdout, stderr = self._process.communicate(timeout=self._timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self._process.kill()
            stdout, stderr = self._process.communicate()
            finished = datetime.now(timezone.utc)
            self._status = SimulationStatus("timed_out", self._process.returncode, self._process.pid, stdout, stderr, started, finished, output)
            self._logger.exception("EnergyPlus simulation timed out after %s seconds", self._timeout_seconds)
            raise EnergyPlusSimulationTimeout(
                f"EnergyPlus simulation exceeded {self._timeout_seconds} seconds"
            ) from exc
        except OSError as exc:
            self._status = SimulationStatus("failed", started_at=started, finished_at=datetime.now(timezone.utc), output_directory=output)
            raise EnergyPlusSimulationError(f"Unable to start EnergyPlus: {exc}") from exc

        finished = datetime.now(timezone.utc)
        state: SimulationState = "completed" if self._process.returncode == 0 else "failed"
        self._status = SimulationStatus(state, self._process.returncode, self._process.pid, stdout, stderr, started, finished, output)
        self._logger.info("EnergyPlus finished with exit code %s", self._process.returncode)
        if stdout.strip():
            self._logger.debug("EnergyPlus stdout:\n%s", stdout.strip())
        if stderr.strip():
            self._logger.warning("EnergyPlus stderr:\n%s", stderr.strip())
        files = self.collect_output_files(output)
        building_state: BuildingState | None = None
        try:
            building_state = self._output_parser.parse(output, return_code=self._process.returncode)
        except (OSError, ValueError) as exc:
            # A simulation result remains useful even when a malformed output
            # artifact prevents parsing; the parser already logs recoverable
            # file/column issues internally.
            self._logger.exception("Unable to parse EnergyPlus outputs in %s", output)
        if self._process.returncode != 0:
            detail = stderr.strip() or stdout.strip() or "no diagnostic output"
            raise EnergyPlusSimulationError(
                f"EnergyPlus simulation failed with exit code {self._process.returncode}: {detail}"
            )
        return SimulationResult(
            run_id=output.name,
            output_directory=str(output),
            completed_at=finished,
            return_code=self._process.returncode,
            stdout=stdout,
            stderr=stderr,
            output_files=tuple(str(path) for path in files),
            building_state=building_state,
        )

    def _prepare_output_requests(self, idf: Path, output: Path) -> Path:
        """Run a copy of an IDF with parser-required outputs enabled.

        Several official EnergyPlus example models intentionally omit reporting
        requests. The parser cannot recover time-series energy or zone data
        from an ESO file alone, so the runner creates a local copy and adds the
        native CSV, meter, and zone-variable requests consumed by
        ``EnergyPlusOutputParser``. The caller's IDF is never modified.
        """
        content = idf.read_text(encoding="utf-8")
        has_output_requests = "! AI_RUNNER_OUTPUT_REQUESTS" in content
        prepared_content = content
        comfort_content, comfort_people_count = self._enable_fanger_comfort(prepared_content)
        if comfort_people_count:
            prepared_content = comfort_content
        if has_output_requests and not comfort_people_count:
            return idf
        if not has_output_requests and not re.search(r"^\s*OutputControl:Files\s*,", prepared_content, re.IGNORECASE | re.MULTILINE):
            additions = _PARSER_OUTPUT_REQUESTS
        else:
            additions = "\n".join(
                line for line in _PARSER_OUTPUT_REQUESTS.splitlines()
                if not line.startswith("OutputControl:Files") and "!-" not in line
            )
        if comfort_people_count and _COMFORT_MODEL_MARKER not in prepared_content:
            additions += "\n" + _COMFORT_SCHEDULES.strip()
        prepared = output / "__energyplus_runner_input.idf"
        prepared.write_text(prepared_content.rstrip() + "\n\n" + additions.strip() + "\n", encoding="utf-8")
        self._logger.info(
            "Prepared non-destructive EnergyPlus input copy with parser output requests and %d Fanger comfort zone(s): %s",
            comfort_people_count,
            prepared,
        )
        return prepared

    @staticmethod
    def _enable_fanger_comfort(content: str) -> tuple[str, int]:
        """Enable Fanger PMV/PPD reporting on copied ``People`` objects.

        EnergyPlus only emits Fanger comfort variables when a ``People``
        object names the Fanger model. This transformation is applied to the
        runner-owned IDF copy, never to the caller's original model.
        """
        if _COMFORT_MODEL_MARKER in content:
            return content, 0

        people_pattern = re.compile(r"(^\s*People\s*,.*?;)", re.IGNORECASE | re.MULTILINE | re.DOTALL)
        count = 0

        def add_model(match: re.Match[str]) -> str:
            nonlocal count
            block = match.group(1)
            if re.search(r"\bFanger\s*;", block, re.IGNORECASE):
                return block
            head, tail = block.rsplit(";", 1)
            count += 1
            return head + "," + tail + _FANGER_FIELDS

        transformed = people_pattern.sub(add_model, content)
        return transformed, count

    def check_simulation_status(self) -> SimulationStatus:
        """Return the latest simulation status, polling an active process first."""
        if self._process and self._process.poll() is not None and self._status.state == "running":
            return_code = self._process.returncode
            self._status = SimulationStatus(
                "completed" if return_code == 0 else "failed",
                return_code,
                self._process.pid,
                self._status.stdout,
                self._status.stderr,
                self._status.started_at,
                datetime.now(timezone.utc),
                self._status.output_directory,
            )
        self._logger.debug("EnergyPlus status: %s", self._status.state)
        return self._status

    def collect_output_files(self, output_directory: str | Path | None = None) -> list[Path]:
        """Return all regular files generated beneath an output directory.

        Files are returned in deterministic path order and are not parsed or
        filtered by extension. This lets later services decide which EnergyPlus
        reports they need while preserving every generated artifact.
        """
        directory = Path(output_directory) if output_directory else self._status.output_directory
        if directory is None or not directory.is_dir():
            raise EnergyPlusOutputError(f"EnergyPlus output directory does not exist: {directory}")
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        self._logger.info("Collected %d EnergyPlus output files from %s", len(files), directory)
        return files

    @staticmethod
    def _require_file(path: str | Path, label: str) -> Path:
        """Resolve and validate a required regular-file input."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise EnergyPlusInputError(f"{label} does not exist or is not a file: {resolved}")
        return resolved
