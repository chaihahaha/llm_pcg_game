"""Configuration loading with defaults + deep merge + dotted access."""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

DEFAULTS: Dict[str, Any] = {
    "llm": {
        "base_url": "http://127.0.0.1:8080",
        "model": "qwen3.8_27b/Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "api_key": "sk-local",
        "context_limit": 65536,
        "max_tokens_out": 768,
        "temperature": 0.85,
        "timeout": 300,
        "enable_thinking": False,
        "mock": False,
        "auto_fallback_mock": True,
        "retries": 2,
        "cache": True,
        "cache_path": "data/llm_cache.db",
    },
    "world": {
        "seed": 20260923,
        "region_size": 512,
        "zone_size": 128,
        "chunk_size": 16,
        "view_size": 16,
        "max_nations": 8,
        "max_npcs_per_chunk": 3,
        "start_hour": 8,
        "start_x": 72,
        "start_y": 72,
    },
    "evolution": {
        "enabled": True,
        "tick_hours": 1,
        "schedule": {"chunk": 6, "zone": 24, "region": 72, "world": 240},
        "max_llm_calls_per_wait": 4,
        "max_chunk_scopes_per_advance": 2,
        "catchup_steps": 3,
        "max_catchup_calls": 8,
        "distance_scaling": True,
        "neighbor_radius": 6,
        "max_neighbors": 12,
        "max_local_events": 8,
        "max_higher_events": 6,
    },
    "context": {
        "max_prompt_tokens": 24000,
        "max_dialogue_turns": 6,
        "max_event_digest": 12,
        "npc_memory_keep": 24,
    },
    "game": {
        "db_path": "data/world.db",
        "autosave": True,
        "player_hp": 30,
        "player_atk": 5,
        "player_def": 2,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def find_default_config() -> str | None:
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(here, "configs", "default.json")
    return candidate if os.path.isfile(candidate) else None


def load_config(path: str | None = None, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load config: DEFAULTS <- json file <- overrides."""
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        path = find_default_config()
    if path and os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, json.load(fh))
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    return cfg


def cfg_get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def cfg_set(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = cfg
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
