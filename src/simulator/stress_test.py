"""Non-destructive inefficient-building test-mode preparation.

This module changes only a caller-provided copy of an IDF. It does not alter
EnergyPlus integration, optimization logic, or the original building model.
The transformations target named schedules used by the standard 5Zone example
and fail safely when a target schedule is absent.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil


@dataclass(frozen=True)
class StressTestChange:
    """One artificial inefficiency applied to the copied IDF."""

    schedule: str
    old_value: str
    new_value: str
    occurrences: int


_SCHEDULE_PATTERN = re.compile(
    r"(?P<object>(?:^|\n)\s*Schedule:Compact\s*,\s*"
    r"(?P<name>[^,;!]+)\s*,.*?;)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)
_UNTIL_VALUE_PATTERN = re.compile(
    r"(?P<prefix>Until:\s*[^,;]+,\s*)"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?P<suffix>\s*[,;])",
    re.IGNORECASE,
)


def prepare_stressed_idf(
    source_idf: str | Path,
    target_idf: str | Path,
    *,
    enabled: bool = True,
    logger: logging.Logger | None = None,
) -> tuple[Path, tuple[StressTestChange, ...]]:
    """Copy ``source_idf`` and apply bounded energy-wasting schedule changes.

    When ``enabled`` is false, the source is copied byte-for-byte and no
    artificial changes are made. The source is never written to.
    """
    log = logger or logging.getLogger(__name__)
    source = Path(source_idf).expanduser().resolve()
    target = Path(target_idf).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Stress-test source IDF does not exist: {source}")
    if source == target:
        raise ValueError("Stress-test target must be a copy, not the original IDF")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if not enabled:
        log.info("Stress test disabled; copied IDF unchanged: %s", target)
        return target, ()

    content = target.read_text(encoding="utf-8")
    changes: list[StressTestChange] = []
    targets = {
        # Keep a valid 0.5 C deadband while making occupied HVAC operation
        # inefficient: higher heating and lower cooling than the example.
        "htg-setp-sch": "23.0",
        "clg-setp-sch": "23.5",
        "lights-1": "1.0",
        "fanavailsched": "1.0",
    }

    def replace_schedule(match: re.Match[str]) -> str:
        name = match.group("name").strip()
        new_value = targets.get(name.casefold())
        if new_value is None:
            return match.group("object")
        object_text = match.group("object")
        count = 0
        old_values: list[str] = []

        def replace_value(value_match: re.Match[str]) -> str:
            nonlocal count
            old_value = value_match.group("value")
            if old_value == new_value:
                return value_match.group(0)
            count += 1
            old_values.append(old_value)
            return value_match.group("prefix") + new_value + value_match.group("suffix")

        updated = _UNTIL_VALUE_PATTERN.sub(replace_value, object_text)
        if count:
            changes.append(StressTestChange(name, ", ".join(old_values), new_value, count))
        return updated

    updated_content = _SCHEDULE_PATTERN.sub(replace_schedule, content)
    if not changes:
        raise ValueError("Stress test could not find any supported target schedules in the copied IDF")
    target.write_text(updated_content, encoding="utf-8")
    for change in changes:
        log.info(
            "Applied stress test: schedule=%s old_values=%s new_value=%s occurrences=%d",
            change.schedule, change.old_value, change.new_value, change.occurrences,
        )
    return target, tuple(changes)
