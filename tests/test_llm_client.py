"""Unit tests for the Ollama client using mocked HTTP responses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.llm.ollama_client import (
    BuildingOptimization,
    ChatMessage,
    OllamaClient,
)
from src.llm.exceptions import OllamaModelError, OllamaResponseError
from src.simulator.output_parser import BuildingState, ComfortState, EnergyState, ThermalState


class FakeHTTPResponse:
    """Minimal context-manager response compatible with urllib usage."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_health_models_and_chat_use_typed_responses() -> None:
    """Basic Ollama endpoints should map to dataclasses, not dictionaries."""
    responses = iter(
        [
            {"version": "0.9.0"},
            {"models": [{"name": "qwen2.5", "size": 123, "details": {"family": "qwen2"}}]},
            # chat() verifies model availability before posting /api/chat.
            {"models": [{"name": "qwen2.5", "size": 123, "details": {"family": "qwen2"}}]},
            {"message": {"role": "assistant", "content": "hello"}, "model": "qwen2.5", "done": True},
        ]
    )
    with patch("src.llm.ollama_client.request.urlopen", side_effect=lambda *_args, **_kwargs: FakeHTTPResponse(next(responses))):
        client = OllamaClient(model="qwen2.5", max_retries=0)
        health = client.health_check()
        models = client.list_models()
        chat = client.chat([ChatMessage("user", "hello")])

    assert health.available is True
    assert health.version == "0.9.0"
    assert models[0].name == "qwen2.5"
    assert chat.content == "hello"


def test_chat_rejects_missing_model() -> None:
    """Chat calls should fail clearly when the configured model is unavailable."""
    with patch("src.llm.ollama_client.request.urlopen", return_value=FakeHTTPResponse({"models": []})):
        with pytest.raises(OllamaModelError):
            OllamaClient(model="mistral", max_retries=0).chat([{"role": "user", "content": "hello"}])


def test_optimize_building_validates_and_types_json_response() -> None:
    """Optimization JSON should be converted into typed action dataclasses."""
    responses = iter(
        [
            {"models": [{"name": "qwen2.5"}]},
            {
                "model": "qwen2.5",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "reasoning": "The occupied zone is warmer than the comfort target.",
                            "actions": [
                                {
                                    "zone": "Office A",
                                    "parameter": "Cooling Setpoint",
                                    "current": 24,
                                    "recommended": 22,
                                    "priority": "high",
                                    "expected_energy_change": "-8%",
                                }
                            ],
                        }
                    ),
                },
            },
        ]
    )
    state = BuildingState(zone_names=("Office A",))
    with patch("src.llm.ollama_client.request.urlopen", side_effect=lambda *_args, **_kwargs: FakeHTTPResponse(next(responses))):
        result = OllamaClient(model="qwen2.5", max_retries=0).optimize_building(state)

    assert isinstance(result, BuildingOptimization)
    assert result.actions[0].zone == "Office A"
    assert result.actions[0].recommended == 22


def test_optimize_building_rejects_non_json_response(tmp_path: Path) -> None:
    """Malformed model output should raise a typed response exception."""
    responses = iter(
        [
            {"models": [{"name": "qwen2.5"}]},
            {"message": {"content": "not json"}},
        ]
    )
    with patch("src.llm.ollama_client.request.urlopen", side_effect=lambda *_args, **_kwargs: FakeHTTPResponse(next(responses))):
        with pytest.raises(OllamaResponseError):
            OllamaClient(model="qwen2.5", max_retries=0).optimize_building(BuildingState())


def test_optimize_building_repairs_fenced_trailing_comma_json() -> None:
    """Safe formatting cleanup occurs before the unchanged schema validation."""
    responses = iter(
        [
            {"models": [{"name": "qwen2.5"}]},
            {"message": {"content": '```json\n{"reasoning":"safe","actions":[{"zone":"Office A","parameter":"Cooling Setpoint","current":24,"recommended":24.5,"priority":"Medium","expected_energy_change":"-1%",},],}\n```'}},
        ]
    )
    with patch("src.llm.ollama_client.request.urlopen", side_effect=lambda *_args, **_kwargs: FakeHTTPResponse(next(responses))):
        result = OllamaClient(model="qwen2.5", max_retries=0).optimize_building(BuildingState())
    assert result.actions[0].zone == "Office A"


def test_optimization_context_reduces_large_zone_state() -> None:
    """Large states are reduced to optimization-relevant context fields."""
    zones = tuple(f"Zone {index}" for index in range(500))
    state = BuildingState(
        energy=EnergyState(total_electricity_consumption=1000, hvac_electricity=500),
        thermal=ThermalState(
            zone_temperatures={zone: 18.0 + index / 100 for index, zone in enumerate(zones)},
            zone_humidity={zone: 40.0 for zone in zones},
        ),
        comfort=ComfortState(
            pmv={f"{zone} PEOPLE 1": -1.0 for zone in zones},
            ppd={f"{zone} PEOPLE 1": 30.0 for zone in zones},
        ),
        zone_names=zones,
        occupied_zones=zones,
    )
    context = OllamaClient(model="qwen2.5")._optimization_context(state)
    assert len(context["zone_names"]) <= 40
    assert "source_files" not in context
    assert "context_reduction" in context
