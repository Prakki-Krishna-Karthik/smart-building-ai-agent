"""Typed, resilient HTTP client for local Ollama model inference.

The client supports common hackathon models such as Qwen2.5, Llama3, and
Mistral while allowing any model tag accepted by the configured Ollama server.
It owns service health checks, local model discovery, optional model pulling,
chat transport, retries, timeout handling, and strict optimization-response
validation. Prompt content is loaded from ``src/llm/prompts`` and is never
embedded in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import socket
import time
from typing import Any, Literal, Mapping, Sequence
from urllib import error, request

from src.config.config import settings
from src.simulator.output_parser import BuildingState
from src.llm.exceptions import (
    OllamaConnectionError,
    OllamaHTTPError,
    OllamaModelError,
    OllamaResponseError,
)


MessageRole = Literal["system", "user", "assistant", "tool"]
SUPPORTED_MODEL_FAMILIES = ("qwen2.5", "llama3", "mistral")
PROMPT_DIRECTORY = Path(__file__).with_name("prompts")


@dataclass(frozen=True)
class ChatMessage:
    """A single Ollama chat message."""

    role: MessageRole
    content: str


@dataclass(frozen=True)
class ChatResponse:
    """Normalized response returned by Ollama's chat endpoint."""

    model: str
    content: str
    done: bool
    raw_response: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OllamaModel:
    """Metadata for a locally available Ollama model."""

    name: str
    model: str = ""
    size: int | None = None
    modified_at: str | None = None
    family: str | None = None


@dataclass(frozen=True)
class OllamaHealth:
    """Result of an Ollama service health check."""

    available: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class OptimizationAction:
    """One validated building-control recommendation from the LLM."""

    zone: str
    parameter: str
    current: float | int | str | None
    recommended: float | int | str | None
    priority: str
    expected_energy_change: str
    confidence_score: float | None = None
    estimated_energy_savings_pct: float | None = None
    llm_predicted_comfort_impact: str | None = None

    @property
    def estimated_comfort_impact(self) -> str | None:
        """Backward-compatible accessor for older callers and reports."""
        return self.llm_predicted_comfort_impact


@dataclass(frozen=True)
class BuildingOptimization:
    """Validated, typed optimization response returned to application code."""

    reasoning: str
    actions: tuple[OptimizationAction, ...]


@dataclass(frozen=True)
class AgentToolSelection:
    """One model-selected tool call or final optimization decision."""

    action: Literal["tool", "recommendation"]
    tool_name: str | None = None
    intent: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    recommendation: BuildingOptimization | None = None


class OllamaClient:
    """Communicate with a local Ollama server using its HTTP API.

    Args:
        model: Model tag, for example ``qwen2.5``, ``llama3``, or ``mistral``.
        base_url: Ollama server URL. Defaults to application configuration.
        auto_pull: Pull the requested model when it is not installed locally.
        timeout_seconds: Timeout for each HTTP request.
        max_retries: Number of retries after transient transport/server failures.
        logger: Optional logger for structured integration diagnostics.

    The constructor does not make a network request. Call ``health_check`` or
    a model/chat method when the service should be contacted.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        auto_pull: bool | None = None,
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
        retry_backoff_seconds: float | None = None,
        prompt_directory: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize a configured Ollama client without contacting the server."""
        self.model = (model or settings.ollama_model).strip()
        if not self.model:
            raise ValueError("Ollama model must not be empty")
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.auto_pull = settings.ollama_auto_pull if auto_pull is None else auto_pull
        self.timeout_seconds = timeout_seconds or settings.ollama_timeout_seconds
        self.max_retries = settings.ollama_max_retries if max_retries is None else max_retries
        self.retry_backoff_seconds = (
            settings.ollama_retry_backoff_seconds
            if retry_backoff_seconds is None
            else retry_backoff_seconds
        )
        self.prompt_directory = Path(prompt_directory) if prompt_directory else PROMPT_DIRECTORY
        self._logger = logger or logging.getLogger(__name__)

    def health_check(self) -> OllamaHealth:
        """Detect whether Ollama is running and return its reported version."""
        try:
            payload = self._request("GET", "/api/version")
        except OllamaError as exc:
            self._logger.warning("Ollama health check failed: %s", exc)
            return OllamaHealth(available=False, error=str(exc))
        version = payload.get("version")
        self._logger.info("Ollama is available%s", f" ({version})" if version else "")
        return OllamaHealth(available=True, version=str(version) if version else None)

    def list_models(self) -> tuple[OllamaModel, ...]:
        """Return all models currently available in the local Ollama registry."""
        payload = self._request("GET", "/api/tags")
        raw_models = payload.get("models", [])
        if not isinstance(raw_models, list):
            raise OllamaResponseError("Ollama /api/tags response has an invalid 'models' field")
        models: list[OllamaModel] = []
        for raw in raw_models:
            if not isinstance(raw, dict) or not raw.get("name"):
                self._logger.warning("Skipping malformed Ollama model entry: %r", raw)
                continue
            details = raw.get("details") if isinstance(raw.get("details"), dict) else {}
            models.append(
                OllamaModel(
                    name=str(raw["name"]),
                    model=str(raw.get("model", "")),
                    size=self._optional_int(raw.get("size")),
                    modified_at=str(raw["modified_at"]) if raw.get("modified_at") else None,
                    family=str(details["family"]) if details.get("family") else None,
                )
            )
        self._logger.info("Ollama reports %d locally available model(s)", len(models))
        return tuple(models)

    def ensure_model_available(self) -> None:
        """Verify the configured model, pulling it when auto-pull is enabled."""
        self._ensure_model_available()

    def is_model_available(self, models: tuple[OllamaModel, ...] | None = None) -> bool:
        """Return whether the configured model exists among local model tags."""
        available_models = models if models is not None else self.list_models()
        return any(self._model_matches(model) for model in available_models)

    def chat(self, messages: Sequence[ChatMessage | Mapping[str, str]]) -> ChatResponse:
        """Send a non-streaming chat request and return a typed response.

        ``messages`` accepts ``ChatMessage`` instances or mappings containing
        ``role`` and ``content``. The response content is intentionally kept as
        text here; ``optimize_building`` applies the stricter JSON schema.
        """
        self._ensure_model_available()
        normalized = self._normalize_messages(messages)
        payload = self._request(
            "POST",
            "/api/chat",
            {"model": self.model, "messages": normalized, "stream": False},
        )
        return self._chat_response(payload)

    def optimize_building(self, building_state: BuildingState) -> BuildingOptimization:
        """Request and validate structured optimization recommendations.

        The model receives a system prompt and a serialized ``BuildingState``
        through templates stored under ``src/llm/prompts``. Ollama is requested
        to return JSON only, then the response is validated field-by-field and
        converted to dataclasses. Invalid JSON or schema data raises
        ``OllamaResponseError`` after the configured retry policy is exhausted.
        """
        if not isinstance(building_state, BuildingState):
            raise TypeError("building_state must be a BuildingState instance")
        system_prompt = self._load_prompt("optimization_system.txt")
        user_template = self._load_prompt("optimization_user.txt")
        raw_state_json = json.dumps(asdict(building_state), indent=2, sort_keys=True, default=str)
        context = self._optimization_context(building_state)
        state_json = json.dumps(context, indent=2, sort_keys=True, default=str)
        user_prompt = user_template.replace("{building_state_json}", state_json)
        simplified_prompt = user_template.replace(
            "{building_state_json}",
            json.dumps(self._simplified_optimization_context(context), indent=2, sort_keys=True),
        )
        self._logger.info(
            "Optimization request diagnostics raw_state_chars=%d compact_state_chars=%d "
            "prompt_chars=%d estimated_tokens=%d zones=%d reduced=%s",
            len(raw_state_json),
            len(state_json),
            len(user_prompt) + len(system_prompt),
            self._estimate_tokens(system_prompt + user_prompt),
            len(context.get("thermal", {}).get("zone_temperatures", {})),
            bool(context.get("context_reduction", {}).get("reduced")),
        )
        self._ensure_model_available()
        last_error: OllamaResponseError | None = None
        for attempt in range(self.max_retries + 1):
            response_content = "<unavailable>"
            active_prompt = simplified_prompt if attempt == 1 else user_prompt
            try:
                payload = self._request(
                    "POST",
                    "/api/chat",
                    {
                        "model": self.model,
                        "messages": self._normalize_messages((
                            ChatMessage("system", system_prompt),
                            ChatMessage("user", active_prompt),
                        )),
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0},
                    },
                )
                response = self._chat_response(payload)
                response_content = response.content
                self._logger.info(
                    "Raw Ollama optimization response attempt=%d response_chars=%d simplified_prompt=%s: %s",
                    attempt + 1,
                    len(response_content),
                    attempt == 1,
                    response_content,
                )
                parsed = self._parse_optimization_json(response.content)
                self._logger.info("Optimization response accepted attempt=%d actions=%d", attempt + 1, len(parsed.actions))
                return parsed
            except OllamaResponseError as exc:
                last_error = exc
                self._logger.error(
                    "Ollama optimization response rejected attempt=%d simplified_prompt=%s reason=%s response_chars=%d raw_response=%s",
                    attempt + 1,
                    attempt == 1,
                    exc,
                    len(response_content),
                    response_content,
                )
                if attempt >= self.max_retries:
                    break
                delay = self.retry_backoff_seconds * (2**attempt)
                self._logger.warning("Invalid optimization response; retry %d/%d in %.2fs: %s", attempt + 1, self.max_retries, delay, exc)
                time.sleep(delay)
        raise last_error or OllamaResponseError("Unable to validate Ollama optimization response")

    def select_tool(
        self,
        tool_specifications: Sequence[Mapping[str, Any]],
        context: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] = (),
    ) -> AgentToolSelection:
        """Ask the model to select the next tool or return final recommendations.

        The response contract is intentionally separate from ``optimize_building``::

            {"action":"tool", "tool":"parse_outputs", "intent":"inspect latest state"}
            {"action":"recommendation", "reasoning":"...", "actions":[...]}

        Tool execution remains outside this transport class. The agent layer
        dispatches the selected tool and supplies its typed result on the next
        call, which prevents the model from receiving implicit filesystem or
        process permissions.
        """
        prompt_path = self.prompt_directory / "tool_selection_user.txt"
        try:
            template = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OllamaResponseError(f"Unable to load prompt template {prompt_path}: {exc}") from exc
        if not template:
            raise OllamaResponseError(f"Prompt template is empty: {prompt_path}")
        system = self._load_prompt("tool_selection_system.txt")
        reduced_context = self._reduce_agent_context(context)
        allowed_tools = {str(spec.get("name", "")).strip() for spec in tool_specifications if spec.get("name")}
        tool_catalog_json = json.dumps(list(tool_specifications), indent=2, default=str)
        history_json = json.dumps(list(history)[-4:], indent=2, default=str)
        user = template.replace("{tool_catalog}", tool_catalog_json)
        user = user.replace("{context_json}", json.dumps(reduced_context, indent=2, default=str))
        user = user.replace("{history_json}", history_json)
        simplified_user = template.replace("{tool_catalog}", tool_catalog_json)
        simplified_user = simplified_user.replace(
            "{context_json}",
            json.dumps(self._simplified_optimization_context(reduced_context), indent=2, default=str),
        )
        simplified_user = simplified_user.replace("{history_json}", history_json)
        self._logger.info(
            "Tool-selection diagnostics full_prompt_chars=%d simplified_prompt_chars=%d "
            "estimated_tokens=%d state_reduced=%s history_items=%d",
            len(system) + len(user),
            len(system) + len(simplified_user),
            self._estimate_tokens(system + user),
            bool(reduced_context.get("context_reduction", {}).get("reduced")),
            len(history),
        )
        last_error: OllamaResponseError | None = None
        retry_limit = min(self.max_retries, 2)
        for attempt in range(retry_limit + 1):
            response_content = "<unavailable>"
            try:
                retry_suffix = "" if attempt == 0 else (
                    "\n\nYour previous response was invalid. Correct it now. Return exactly one JSON object "
                    "with action=tool and tool+intent, or action=recommendation with reasoning+actions. "
                    "If final_decision_required is true, you MUST use action=recommendation and include "
                    "an actions array; never return selection, tool_name, or planning fields. "
                    "Do not include Markdown or explanatory text."
                )
                active_user = simplified_user if attempt == 1 else user
                messages = self._normalize_messages((
                    ChatMessage("system", system),
                    ChatMessage("user", active_user + retry_suffix),
                ))
                payload = self._request(
                    "POST",
                    "/api/chat",
                    {
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": 0},
                    },
                )
                response = self._chat_response(payload)
                response_content = response.content
                self._logger.info(
                    "Raw Ollama tool-selection response attempt=%d response_chars=%d simplified_prompt=%s: %s",
                    attempt + 1,
                    len(response_content),
                    attempt == 1,
                    response_content,
                )
                raw = self._extract_json_object(response.content)
                if not isinstance(raw, dict) or raw.get("action") not in {"tool", "recommendation"}:
                    raise OllamaResponseError("Tool-selection JSON must contain action=tool or action=recommendation")
                reasoning = raw.get("reasoning", "")
                if not isinstance(reasoning, str):
                    raise OllamaResponseError("Tool-selection reasoning must be a string")
                if raw["action"] == "tool":
                    tool_name = raw.get("tool")
                    intent = raw.get("intent", "")
                    if "arguments" in raw:
                        raise OllamaResponseError("Tool selection must not contain filesystem or execution arguments")
                    if not isinstance(tool_name, str) or not tool_name.strip() or not isinstance(intent, str) or not intent.strip():
                        raise OllamaResponseError("Tool action requires a string tool and high-level intent")
                    if tool_name.strip() not in allowed_tools:
                        raise OllamaResponseError(
                            f"Unknown tool '{tool_name.strip()}'; choose only from registered tools: "
                            f"{', '.join(sorted(allowed_tools))}"
                        )
                    return AgentToolSelection("tool", tool_name.strip(), intent=intent.strip(), reasoning=reasoning)
                recommendation = self._parse_optimization_json(json.dumps({"reasoning": reasoning, "actions": raw.get("actions", [])}))
                return AgentToolSelection("recommendation", reasoning=reasoning, recommendation=recommendation)
            except json.JSONDecodeError as exc:
                last_error = OllamaResponseError(f"Tool-selection response was not valid JSON: {exc}")
            except OllamaResponseError as exc:
                last_error = exc
                self._logger.error(
                    "Ollama tool-selection response rejected attempt=%d: %s; raw_response=%s",
                    attempt + 1,
                    exc,
                    response_content,
                )
            if attempt < retry_limit:
                delay = self.retry_backoff_seconds * (2**attempt)
                self._logger.warning(
                    "Invalid tool-selection response; retry %d/%d in %.2fs: %s",
                    attempt + 1,
                    retry_limit,
                    delay,
                    last_error,
                )
                time.sleep(delay)
        raise last_error or OllamaResponseError("Unable to validate Ollama tool-selection response")

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        """Parse a JSON object from plain or fenced model output."""
        text = content.strip()
        if not text:
            raise OllamaResponseError("Tool-selection response was empty")
        candidates = [text]
        if "```" in text:
            candidates.append(text.replace("```json", "").replace("```", "").strip())
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise OllamaResponseError("Tool-selection response did not contain a JSON object")

    def _ensure_model_available(self) -> None:
        """Verify the configured model and optionally pull it once."""
        models = self.list_models()
        if any(self._model_matches(model) for model in models):
            return
        if not self.auto_pull:
            raise OllamaModelError(
                f"Ollama model '{self.model}' is not installed. "
                "Set OLLAMA_AUTO_PULL=true to pull it automatically."
            )
        self._logger.info("Ollama model %s is missing; pulling it", self.model)
        self._request("POST", "/api/pull", {"model": self.model, "stream": False})
        pulled_models = self.list_models()
        if not any(self._model_matches(model) for model in pulled_models):
            raise OllamaModelError(f"Ollama pull completed but model '{self.model}' is still unavailable")

    def _model_matches(self, model: OllamaModel) -> bool:
        """Match exact tags and the common implicit ``:latest`` tag."""
        configured = self.model.casefold()
        for candidate in (model.name, model.model):
            normalized = candidate.casefold()
            if normalized == configured or normalized == f"{configured}:latest":
                return True
        return False

    def _request(self, method: str, path: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Perform a JSON HTTP request with bounded exponential retries."""
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        url = f"{self.base_url}{path}"
        for attempt in range(self.max_retries + 1):
            try:
                req = request.Request(url, data=body, headers=headers, method=method)
                with request.urlopen(req, timeout=self.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw) if raw else {}
                if not isinstance(parsed, dict):
                    raise OllamaResponseError(f"Ollama returned a non-object JSON response from {path}")
                return parsed
            except error.HTTPError as exc:
                detail = self._read_http_error(exc)
                transient = exc.code == 429 or exc.code >= 500
                if not transient or attempt >= self.max_retries:
                    raise OllamaHTTPError(f"Ollama HTTP {exc.code} for {path}: {detail}") from exc
                self._retry(attempt, path, f"HTTP {exc.code}")
            except (error.URLError, TimeoutError, socket.timeout) as exc:
                if attempt >= self.max_retries:
                    raise OllamaConnectionError(f"Unable to reach Ollama at {self.base_url}: {exc}") from exc
                self._retry(attempt, path, str(exc))
            except json.JSONDecodeError as exc:
                raise OllamaResponseError(f"Ollama returned invalid JSON from {path}: {exc}") from exc
        raise OllamaConnectionError(f"Request to Ollama failed: {path}")

    def _retry(self, attempt: int, path: str, reason: str) -> None:
        """Log and sleep before a transient request retry."""
        delay = self.retry_backoff_seconds * (2**attempt)
        self._logger.warning("Ollama request %s failed (%s); retry %d/%d in %.2fs", path, reason, attempt + 1, self.max_retries, delay)
        time.sleep(delay)

    @staticmethod
    def _read_http_error(exc: error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", errors="replace")[:500]
        except OSError:
            return str(exc)

    @staticmethod
    def _normalize_messages(messages: Sequence[ChatMessage | Mapping[str, str]]) -> list[dict[str, str]]:
        normalized = []
        for message in messages:
            if isinstance(message, ChatMessage):
                role, content = message.role, message.content
            elif isinstance(message, Mapping):
                role, content = message.get("role"), message.get("content")
            else:
                raise TypeError("Each chat message must be ChatMessage or a role/content mapping")
            if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
                raise ValueError("Each chat message requires a valid role and string content")
            normalized.append({"role": role, "content": content})
        if not normalized:
            raise ValueError("At least one chat message is required")
        return normalized

    def _chat_response(self, payload: Mapping[str, Any]) -> ChatResponse:
        message = payload.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise OllamaResponseError("Ollama chat response is missing message.content")
        return ChatResponse(
            model=str(payload.get("model", self.model)),
            content=message["content"],
            done=bool(payload.get("done", True)),
            raw_response=payload,
        )

    def _parse_optimization_json(self, content: str) -> BuildingOptimization:
        """Validate the required optimization JSON shape and create dataclasses."""
        try:
            raw = json.loads(self._repair_json_text(content))
        except json.JSONDecodeError as exc:
            raise OllamaResponseError(f"Optimization response was not valid JSON: {exc}") from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("reasoning"), str) or not isinstance(raw.get("actions"), list):
            raise OllamaResponseError("Optimization JSON must contain string 'reasoning' and array 'actions'")
        actions = []
        required = ("zone", "parameter", "current", "recommended", "priority", "expected_energy_change")
        for index, item in enumerate(raw["actions"]):
            if not isinstance(item, dict) or any(key not in item for key in required):
                raise OllamaResponseError(f"Optimization action {index} is missing required fields")
            if not all(isinstance(item[key], str) for key in ("zone", "parameter", "priority", "expected_energy_change")):
                raise OllamaResponseError(f"Optimization action {index} has invalid text fields")
            actions.append(OptimizationAction(**{key: item[key] for key in required}))
        return BuildingOptimization(reasoning=raw["reasoning"], actions=tuple(actions))

    def _optimization_context(self, building_state: BuildingState) -> dict[str, Any]:
        """Build a compact optimization context without changing BuildingState."""
        occupied = list(building_state.occupied_zones)
        temperatures = dict(building_state.thermal.zone_temperatures)
        humidity = dict(building_state.thermal.zone_humidity)
        pmv = dict(building_state.comfort.pmv)
        ppd = dict(building_state.comfort.ppd)
        names = list(dict.fromkeys(occupied or building_state.zone_names))
        max_zones = max(1, settings.ollama_context_max_zones)
        if len(names) > max_zones:
            names = sorted(names, key=lambda zone: abs(temperatures.get(zone, 22.0) - 22.0), reverse=True)[:max_zones]
        selected = set(names)
        context: dict[str, Any] = {
            "simulation_success": building_state.simulation_success,
            "simulation_warnings": list(building_state.simulation_warnings)[-5:],
            "simulation_errors": list(building_state.simulation_errors)[-5:],
            "energy": {"total_electricity_consumption": building_state.energy.total_electricity_consumption, "hvac_electricity": building_state.energy.hvac_electricity},
            "thermal": {
                "zone_temperatures": {k: v for k, v in temperatures.items() if k in selected},
                "zone_humidity": {k: v for k, v in humidity.items() if k in selected},
                "outdoor_temperature": building_state.thermal.outdoor_temperature,
            },
            "comfort": {
                "pmv": {k: v for k, v in pmv.items() if k.split(" PEOPLE ", 1)[0] in selected or k in selected},
                "ppd": {k: v for k, v in ppd.items() if k.split(" PEOPLE ", 1)[0] in selected or k in selected},
            },
            "zone_names": names,
            "occupied_zones": [zone for zone in occupied if zone in selected],
            "hvac_operating_state": {k: v for k, v in building_state.hvac_operating_state.items() if k in selected},
        }
        serialized = json.dumps(context, separators=(",", ":"), default=str)
        if len(serialized) > settings.ollama_context_max_chars:
            context = self._simplified_optimization_context(context)
            context["context_reduction"] = {"reduced": True, "reason": "context exceeded configured character limit"}
        else:
            context["context_reduction"] = {"reduced": len(names) < len(occupied or building_state.zone_names), "reason": "zone cap" if len(names) < len(occupied or building_state.zone_names) else "none"}
        return context

    def _reduce_agent_context(self, context: Mapping[str, Any]) -> dict[str, Any]:
        """Reduce an agent-loop context while preserving control-flow signals."""
        reduced = dict(context)
        state = context.get("building_state")
        if isinstance(state, Mapping):
            thermal = state.get("thermal") if isinstance(state.get("thermal"), Mapping) else {}
            energy = state.get("energy") if isinstance(state.get("energy"), Mapping) else {}
            comfort = state.get("comfort") if isinstance(state.get("comfort"), Mapping) else {}
            occupied = list(state.get("occupied_zones", ()))
            temperatures = thermal.get("zone_temperatures", {}) if isinstance(thermal.get("zone_temperatures"), Mapping) else {}
            humidity = thermal.get("zone_humidity", {}) if isinstance(thermal.get("zone_humidity"), Mapping) else {}
            max_zones = max(1, settings.ollama_context_max_zones)
            names = list(dict.fromkeys(occupied or state.get("zone_names", ())))
            if len(names) > max_zones:
                names = sorted(names, key=lambda zone: abs(float(temperatures.get(zone, 22.0)) - 22.0), reverse=True)[:max_zones]
            selected = set(names)
            reduced["building_state"] = {
                "simulation_success": state.get("simulation_success"),
                "simulation_warnings": list(state.get("simulation_warnings", ()))[:5],
                "simulation_errors": list(state.get("simulation_errors", ()))[:5],
                "energy": {"total_electricity_consumption": energy.get("total_electricity_consumption"), "hvac_electricity": energy.get("hvac_electricity")},
                "thermal": {
                    "zone_temperatures": {k: v for k, v in temperatures.items() if k in selected},
                    "zone_humidity": {k: v for k, v in humidity.items() if k in selected},
                    "outdoor_temperature": thermal.get("outdoor_temperature"),
                },
                "comfort": {
                    "pmv": {k: v for k, v in (comfort.get("pmv", {}) if isinstance(comfort.get("pmv"), Mapping) else {}).items() if k.split(" PEOPLE ", 1)[0] in selected or k in selected},
                    "ppd": {k: v for k, v in (comfort.get("ppd", {}) if isinstance(comfort.get("ppd"), Mapping) else {}).items() if k.split(" PEOPLE ", 1)[0] in selected or k in selected},
                },
                "zone_names": names,
                "occupied_zones": [zone for zone in occupied if zone in selected],
                "hvac_operating_state": {},
                "source_files": (),
            }
            original_size = len(json.dumps(state, default=str))
            reduced_size = len(json.dumps(reduced["building_state"], default=str))
            reduced["context_reduction"] = {"reduced": reduced_size < original_size, "original_chars": original_size, "reduced_chars": reduced_size}
        return reduced

    @staticmethod
    def _simplified_optimization_context(context: Mapping[str, Any]) -> dict[str, Any]:
        """Retain only fields needed for a safe setpoint recommendation."""
        thermal = context.get("thermal", {}) if isinstance(context.get("thermal"), Mapping) else {}
        energy = context.get("energy", {}) if isinstance(context.get("energy"), Mapping) else {}
        return {
            "simulation_success": context.get("simulation_success"),
            "energy": {"total_electricity_consumption": energy.get("total_electricity_consumption"), "hvac_electricity": energy.get("hvac_electricity")},
            "zone_names": context.get("zone_names", []),
            "occupied_zones": context.get("occupied_zones", []),
            "zone_temperatures": thermal.get("zone_temperatures", {}),
            "zone_humidity": thermal.get("zone_humidity", {}),
            "outdoor_temperature": thermal.get("outdoor_temperature"),
            "pmv": context.get("comfort", {}).get("pmv", {}) if isinstance(context.get("comfort"), Mapping) else {},
            "ppd": context.get("comfort", {}).get("ppd", {}) if isinstance(context.get("comfort"), Mapping) else {},
        }

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate tokens conservatively for diagnostics without a tokenizer."""
        return max(1, (len(text) + 3) // 4)

    @staticmethod
    def _repair_json_text(content: str) -> str:
        """Apply only semantics-preserving JSON cleanup before schema validation."""
        text = content.strip()
        if "```" in text:
            text = text.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
        # Trailing commas are unambiguous; missing commas are intentionally
        # not guessed because that could change semantic content.
        import re
        return re.sub(r",\s*([}\]])", r"\1", text)

    def _load_prompt(self, filename: str) -> str:
        """Load a required prompt template from the configured prompt directory."""
        path = self.prompt_directory / filename
        try:
            content = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise OllamaResponseError(f"Unable to load prompt template {path}: {exc}") from exc
        if not content:
            raise OllamaResponseError(f"Prompt template is empty: {path}")
        return content

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None


# Backward-compatible name for the original scaffold boundary.
OllamaDecisionClient = OllamaClient
