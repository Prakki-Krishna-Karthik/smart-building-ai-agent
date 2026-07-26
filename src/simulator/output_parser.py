"""Parser for standard EnergyPlus simulation output files.

EnergyPlus normally writes ``eplusout.csv``, ``eplusout.err``,
``eplusout.end``, and optionally ``eplusout.eso`` into the simulation output
directory. ``EnergyPlusOutputParser`` reads those files into immutable,
strongly typed dataclasses. CSV is the primary source for time-series values;
the text files provide run health, diagnostics, duration, and ESO availability.

The parser is intentionally tolerant: missing files, unknown columns, blank
values, and new EnergyPlus output variables are logged and represented as
empty/optional fields rather than causing unrelated dashboard or control-loop
work to fail. It does not perform AI reasoning or control optimization.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class EnergyState:
    """Aggregated electricity consumption in the units reported by EnergyPlus."""

    total_electricity_consumption: float | None = None
    hvac_electricity: float | None = None
    lighting_electricity: float | None = None
    equipment_electricity: float | None = None


@dataclass(frozen=True)
class ThermalState:
    """Latest available zone and outdoor thermal conditions."""

    zone_temperatures: dict[str, float] = field(default_factory=dict)
    outdoor_temperature: float | None = None
    zone_humidity: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ComfortState:
    """Latest available thermal comfort indicators by zone."""

    pmv: dict[str, float] = field(default_factory=dict)
    ppd: dict[str, float] = field(default_factory=dict)
    thermal_comfort_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildingState:
    """Complete structured snapshot extracted from one EnergyPlus run."""

    simulation_success: bool | None = None
    simulation_warnings: tuple[str, ...] = ()
    simulation_errors: tuple[str, ...] = ()
    simulation_duration_seconds: float | None = None
    energy: EnergyState = field(default_factory=EnergyState)
    thermal: ThermalState = field(default_factory=ThermalState)
    comfort: ComfortState = field(default_factory=ComfortState)
    zone_names: tuple[str, ...] = ()
    occupied_zones: tuple[str, ...] = ()
    hvac_operating_state: dict[str, str] = field(default_factory=dict)
    eso_available: bool = False
    eso_record_count: int = 0
    source_files: tuple[str, ...] = ()


class EnergyPlusOutputParser:
    """Read EnergyPlus output artifacts into a :class:`BuildingState`.

    Args:
        logger: Optional logger. If omitted, a module logger is used.

    The parser may be reused for many simulation directories. Each call to
    ``parse`` creates a new independent state object and never retains mutable
    references to a previous run.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Initialize an output parser."""
        self._logger = logger or logging.getLogger(__name__)

    def parse(self, output_directory: str | Path, return_code: int | None = None) -> BuildingState:
        """Parse all recognized files in an EnergyPlus output directory.

        Missing files and unsupported columns are handled gracefully and
        recorded through logging. The returned state always has valid typed
        defaults, making it safe to consume immediately after a run.

        Raises:
            FileNotFoundError: If ``output_directory`` does not exist.
            NotADirectoryError: If ``output_directory`` is not a directory.

        ``return_code`` should be supplied by ``EnergyPlusRunner`` when
        available. A zero code establishes successful process completion unless
        the diagnostics contain a genuine severe/fatal error; a non-zero code
        always marks the simulation unsuccessful. If no return code is known,
        the parser uses the explicit successful end marker and otherwise leaves
        success unknown rather than treating missing optional files as failure.
        """
        directory = Path(output_directory).expanduser().resolve()
        if not directory.exists():
            raise FileNotFoundError(f"EnergyPlus output directory not found: {directory}")
        if not directory.is_dir():
            raise NotADirectoryError(f"EnergyPlus output path is not a directory: {directory}")

        csv_path = directory / "eplusout.csv"
        eso_path = directory / "eplusout.eso"
        err_path = directory / "eplusout.err"
        end_path = directory / "eplusout.end"
        source_files = tuple(str(path) for path in (csv_path, eso_path, err_path, end_path) if path.is_file())

        rows, headers = self._read_csv(csv_path)
        warnings, errors = self._read_diagnostics(err_path)
        end_text = self._read_text(end_path)
        eso_available, eso_record_count = self._inspect_eso(eso_path)
        success = self._simulation_success(end_text, errors, return_code)
        duration = self._simulation_duration(end_text, rows, headers)
        energy = self._parse_energy(rows, headers)
        thermal = self._parse_thermal(rows, headers)
        comfort = self._parse_comfort(rows, headers)
        zone_names = self._zone_names(headers)
        occupied_zones = self._occupied_zones(rows, headers, zone_names)
        hvac_state = self._hvac_state(rows, headers)

        self._logger.info(
            "Parsed EnergyPlus output directory=%s csv_rows=%d warnings=%d errors=%d zones=%d",
            directory,
            len(rows),
            len(warnings),
            len(errors),
            len(zone_names),
        )
        return BuildingState(
            simulation_success=success,
            simulation_warnings=tuple(warnings),
            simulation_errors=tuple(errors),
            simulation_duration_seconds=duration,
            energy=energy,
            thermal=thermal,
            comfort=comfort,
            zone_names=tuple(sorted(zone_names)),
            occupied_zones=tuple(sorted(occupied_zones)),
            hvac_operating_state=hvac_state,
            eso_available=eso_available,
            eso_record_count=eso_record_count,
            source_files=source_files,
        )

    def _read_csv(self, path: Path) -> tuple[list[dict[str, str]], list[str]]:
        """Read a CSV while normalizing headers and tolerating malformed cells."""
        if not path.is_file():
            self._logger.warning("EnergyPlus CSV output is missing: %s", path)
            return [], []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                headers = [str(header).strip() for header in (reader.fieldnames or [])]
                rows = [
                    {str(key).strip(): (value or "").strip() for key, value in row.items() if key is not None}
                    for row in reader
                ]
        except (OSError, csv.Error) as exc:
            self._logger.error("Unable to parse EnergyPlus CSV %s: %s", path, exc)
            return [], []
        self._logger.debug("Read %d rows and %d columns from %s", len(rows), len(headers), path)
        return rows, headers

    def _read_diagnostics(self, path: Path) -> tuple[list[str], list[str]]:
        """Extract warning and error lines from ``eplusout.err``."""
        if not path.is_file():
            self._logger.warning("EnergyPlus error output is missing: %s", path)
            return [], []
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
        except OSError as exc:
            self._logger.error("Unable to read EnergyPlus diagnostics %s: %s", path, exc)
            return [], []
        warnings: list[str] = []
        errors: list[str] = []
        for line in lines:
            summary = re.search(
                r"(?P<warnings>\d+)\s+Warnings?\s*;\s*"
                r"(?P<severe>\d+)\s+Severe\s+Errors?",
                line,
                re.IGNORECASE,
            )
            if summary:
                if int(summary.group("warnings")) > 0:
                    warnings.append(line)
                if int(summary.group("severe")) > 0:
                    errors.append(line)
                continue
            if re.search(r"\*\*\s*Warning\b", line, re.IGNORECASE):
                warnings.append(line)
            elif re.search(r"\*\*\s*(?:Severe|Fatal)\b", line, re.IGNORECASE):
                errors.append(line)
        self._logger.debug("Read %d warnings and %d errors from %s", len(warnings), len(errors), path)
        return warnings, errors

    def _read_text(self, path: Path) -> str:
        """Read a text output file, returning an empty string when absent."""
        if not path.is_file():
            self._logger.warning("EnergyPlus end output is missing: %s", path)
            return ""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            self._logger.error("Unable to read EnergyPlus end file %s: %s", path, exc)
            return ""

    def _inspect_eso(self, path: Path) -> tuple[bool, int]:
        """Inspect optional ESO output and count non-empty data records."""
        if not path.is_file():
            self._logger.info("Optional EnergyPlus ESO output is not available: %s", path)
            return False, 0
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            self._logger.warning("Unable to inspect optional ESO file %s: %s", path, exc)
            return True, 0
        records = sum(1 for line in lines if line.strip() and not line.lstrip().startswith("!"))
        self._logger.debug("Inspected ESO file %s with %d records", path, records)
        return True, records

    @staticmethod
    def _simulation_success(end_text: str, errors: Sequence[str], return_code: int | None) -> bool | None:
        """Infer success from fatal diagnostics, process code, or end marker."""
        if errors:
            return False
        if return_code is not None:
            return return_code == 0
        if re.search(r"completed\s+successfully", end_text, re.IGNORECASE):
            return True
        return None

    def _simulation_duration(self, end_text: str, rows: list[dict[str, str]], headers: list[str]) -> float | None:
        """Extract elapsed process time, falling back to CSV timestamp span."""
        patterns = (
            r"elapsed\s+time\s*=\s*(\d+):(\d{1,2}):(\d{1,2}(?:\.\d+)?)",
            r"(?:simulation\s+duration|elapsed\s+time)\s*[:=]\s*([\d.]+)\s*seconds?",
        )
        for pattern in patterns:
            match = re.search(pattern, end_text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    hours, minutes, seconds = match.groups()
                    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                return float(match.group(1))
        date_column = self._find_column(headers, lambda name: name.lower() in {"date/time", "datetime", "timestamp"})
        if date_column and len(rows) >= 2:
            parsed = [self._parse_timestamp(row.get(date_column, "")) for row in rows]
            parsed = [value for value in parsed if value is not None]
            if len(parsed) >= 2:
                return max(0.0, (max(parsed) - min(parsed)).total_seconds())
        return None

    def _parse_energy(self, rows: list[dict[str, str]], headers: list[str]) -> EnergyState:
        """Aggregate recognized electricity columns without assuming units."""
        def total(predicate: object) -> float | None:
            columns = self._matching_columns(headers, predicate)
            values = [self._sum_column(rows, column) for column in columns]
            values = [value for value in values if value is not None]
            return sum(values) if values else None

        electricity = lambda name: "electricity" in name.lower() or "electric demand" in name.lower()
        return EnergyState(
            total_electricity_consumption=total(lambda name: electricity(name) and "facility" in name.lower()),
            hvac_electricity=total(lambda name: electricity(name) and any(word in name.lower() for word in ("hvac", "heating", "cooling"))),
            lighting_electricity=total(lambda name: electricity(name) and "lighting" in name.lower()),
            equipment_electricity=total(lambda name: electricity(name) and "equipment" in name.lower()),
        )

    def _parse_thermal(self, rows: list[dict[str, str]], headers: list[str]) -> ThermalState:
        """Extract latest zone temperature, outdoor temperature, and humidity."""
        zone_temperatures: dict[str, float] = {}
        zone_humidity: dict[str, float] = {}
        for column in headers:
            lower = column.lower()
            value = self._last_value(rows, column)
            if value is None:
                continue
            if "zone mean air temperature" in lower:
                zone_temperatures[self._zone_name(column)] = value
            elif "relative humidity" in lower and "zone" in lower:
                zone_humidity[self._zone_name(column)] = value
        outdoor_column = self._find_column(headers, lambda name: "outdoor air drybulb temperature" in name.lower() or "site outdoor air drybulb" in name.lower())
        return ThermalState(zone_temperatures, self._last_value(rows, outdoor_column) if outdoor_column else None, zone_humidity)

    def _parse_comfort(self, rows: list[dict[str, str]], headers: list[str]) -> ComfortState:
        """Extract PMV, PPD, and other recognized thermal comfort values."""
        pmv: dict[str, float] = {}
        ppd: dict[str, float] = {}
        metrics: dict[str, float] = {}
        for column in headers:
            lower = column.lower()
            value = self._last_value(rows, column)
            if value is None or "comfort" not in lower and "pmv" not in lower and "ppd" not in lower:
                continue
            metric_name = self._metric_name(column)
            metrics[metric_name] = value
            if "pmv" in lower:
                pmv[self._zone_name(column)] = value
            if "ppd" in lower:
                ppd[self._zone_name(column)] = value
        return ComfortState(pmv, ppd, metrics)

    def _zone_names(self, headers: Iterable[str]) -> set[str]:
        """Discover zone names from common zone-level output variables."""
        names = set()
        for column in headers:
            lower = column.lower()
            if any(token in lower for token in ("zone mean air temperature", "zone air relative humidity", "thermal comfort", "occupant", "people")):
                names.add(self._zone_name(column))
        return {name for name in names if name and name.lower() not in {"environment", "site"}}

    def _occupied_zones(self, rows: list[dict[str, str]], headers: list[str], zones: Iterable[str]) -> set[str]:
        """Infer occupied zones from occupant/people count columns."""
        occupied = set()
        for column in headers:
            lower = column.lower()
            if not any(token in lower for token in ("occupant", "people", "occupancy")):
                continue
            zone = self._zone_name(column)
            values = [self._to_float(row.get(column, "")) for row in rows]
            if any(value is not None and value > 0 for value in values):
                occupied.add(zone)
        return occupied.intersection(set(zones))

    def _hvac_state(self, rows: list[dict[str, str]], headers: list[str]) -> dict[str, str]:
        """Capture latest values from columns that explicitly describe HVAC state."""
        state = {}
        for column in headers:
            lower = column.lower()
            if "hvac" not in lower or not any(token in lower for token in ("state", "status", "operat")):
                continue
            value = self._last_text_value(rows, column)
            if value is not None:
                state[self._metric_name(column)] = value
        return state

    @staticmethod
    def _matching_columns(headers: Iterable[str], predicate: object) -> list[str]:
        return [header for header in headers if callable(predicate) and predicate(header)]

    @staticmethod
    def _find_column(headers: Iterable[str], predicate: object) -> str | None:
        return next((header for header in headers if callable(predicate) and predicate(header)), None)

    def _sum_column(self, rows: list[dict[str, str]], column: str) -> float | None:
        values = [self._to_float(row.get(column, "")) for row in rows]
        values = [value for value in values if value is not None]
        return sum(values) if values else None

    def _last_value(self, rows: list[dict[str, str]], column: str | None) -> float | None:
        if not column:
            return None
        for row in reversed(rows):
            value = self._to_float(row.get(column, ""))
            if value is not None:
                return value
        return None

    @staticmethod
    def _last_text_value(rows: list[dict[str, str]], column: str) -> str | None:
        for row in reversed(rows):
            value = row.get(column, "").strip()
            if value:
                return value
        return None

    @staticmethod
    def _to_float(value: str | None) -> float | None:
        if value is None or not value.strip():
            return None
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None

    @staticmethod
    def _zone_name(column: str) -> str:
        return column.split(":", 1)[0].strip() if ":" in column else column.strip()

    @staticmethod
    def _metric_name(column: str) -> str:
        return column.split(" [", 1)[0].strip()

    @staticmethod
    def _parse_timestamp(value: str) -> datetime | None:
        value = value.strip()
        for pattern in ("%m/%d %H:%M:%S", "%m/%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(value, pattern)
            except ValueError:
                continue
        return None
