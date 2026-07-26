"""Central, environment-driven application configuration.

Configuration is deliberately kept separate from business logic so deployment
settings can change between a developer laptop, the hackathon demo, and a
future production environment. Secrets are loaded from environment variables
and are never committed to source control.
"""

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_environment_file(path: Path) -> None:
    """Load a simple ``.env`` file without requiring an optional dependency.

    ``python-dotenv`` remains supported when installed. The fallback handles
    the common ``KEY=value`` form, preserves Windows backslashes in paths, and
    never overwrites variables explicitly provided by the operating system.
    """
    if load_dotenv is not None:
        load_dotenv(path, override=False)
        return
    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[7:].lstrip()
        key, separator, value = stripped.partition("=")
        if not separator or not key.strip():
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_environment_file(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the simulator, LLM adapter, and UI."""

    project_root: Path = PROJECT_ROOT
    energyplus_executable: str = os.getenv("ENERGYPLUS_EXECUTABLE", "").strip() or "energyplus"
    energyplus_installation_path: Path | None = (
        Path(os.environ["ENERGYPLUS_INSTALLATION_PATH"])
        if os.getenv("ENERGYPLUS_INSTALLATION_PATH")
        else None
    )
    demo_idf_path: Path | None = Path(os.environ["DEMO_IDF_PATH"]).expanduser() if os.getenv("DEMO_IDF_PATH") else None
    demo_weather_file: Path | None = Path(os.environ["DEMO_WEATHER_FILE"]).expanduser() if os.getenv("DEMO_WEATHER_FILE") else None
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5")
    ollama_auto_pull: bool = os.getenv("OLLAMA_AUTO_PULL", "false").lower() in {"1", "true", "yes", "on"}
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
    ollama_max_retries: int = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
    ollama_retry_backoff_seconds: float = float(os.getenv("OLLAMA_RETRY_BACKOFF_SECONDS", "1.0"))
    ollama_context_max_chars: int = int(os.getenv("OLLAMA_CONTEXT_MAX_CHARS", "24000"))
    ollama_context_max_zones: int = int(os.getenv("OLLAMA_CONTEXT_MAX_ZONES", "40"))
    decision_min_temperature_c: float = float(os.getenv("DECISION_MIN_TEMPERATURE_C", "16.0"))
    decision_max_temperature_c: float = float(os.getenv("DECISION_MAX_TEMPERATURE_C", "30.0"))
    decision_min_fan_percent: float = float(os.getenv("DECISION_MIN_FAN_PERCENT", "0.0"))
    decision_max_fan_percent: float = float(os.getenv("DECISION_MAX_FAN_PERCENT", "100.0"))
    simulation_timeout_seconds: int = int(os.getenv("SIMULATION_TIMEOUT_SECONDS", "300"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    stress_test: bool = os.getenv("STRESS_TEST", "false").lower() in {"1", "true", "yes", "on"}

    @property
    def input_directory(self) -> Path:
        """Return the default EnergyPlus input directory."""
        return self.project_root / "data" / "input"

    @property
    def output_directory(self) -> Path:
        """Return the default simulation output directory."""
        return self.project_root / "data" / "output"

    @property
    def log_directory(self) -> Path:
        """Return the default application log directory."""
        return self.project_root / "data" / "logs"


settings = Settings()
