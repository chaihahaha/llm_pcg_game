"""Runtime rules and monkey patches.

This is where the model gets to change the *program*, not just the world:

* **rules** are named knobs the engine reads live (speech length, move cost,
  which terrain is passable, evolution bias...).  Changing one changes
  behaviour on the very next action.
* **hooks** are sandboxed lambdas injected at named extension points
  (``can_enter``, ``on_enter``, ``damage``, ``speech``, ``move_cost``,
  ``evolve_bias``).  A hook returning ``None`` means "no opinion, use default".

Both are persisted in the ``patches`` table and re-compiled when the save is
loaded, so a world-changing spell outlives the session.  A patch that fails to
validate or compile is stored **disabled** and reported instead of crashing the
game — the model must never be able to brick a save.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from .sandbox import SandboxError, compile_hook, exec_patch, validate_code

HOOKS = ("can_enter", "on_enter", "damage", "speech", "move_cost", "evolve_bias", "sim_npc")

DEFAULTS: Dict[str, Any] = {
    "npc_speech_max_chars": 0,      # 0 = unlimited
    "move_cost_hours": 1,
    "damage_multiplier": 1.0,
    "heal_multiplier": 1.0,
    "extra_passable_terrain": [],   # e.g. ["forest"]
    "extra_solid_terrain": [],
    "forbidden_words": [],
    "evolution_bias": "",
    "action_cost_multiplier": 1.0,
    # --- deterministic substrate knobs: the LLM may rewrite any of these -----
    "hunger_rate": 4.0,             # need gained per 24h
    "fatigue_rate": 6.0,
    "anger_rate": 0.5,
    "hostility_cool_hours": 168,   # how long a grudge keeps an NPC hostile
    "morale_rate": 3.0,             # morale lost per 24h
    "eat_at": 65.0,                 # hunger at which an NPC forages
    "starve_at": 90.0,              # hunger at which HP starts dropping
    "starve_hp_loss": 1.0,          # HP per 6h while starving
    "anger_attack_at": 70.0,        # anger at which an adjacent NPC turns hostile
    "mood_override_hours": 12,      # how long an LLM-written mood outranks the formula
    "fire_burnout_hours": 48.0,
    "food_spoil_hours": 72.0,
    "relation_decay": 1.0,          # points per week toward neutral
}

_RULE_DOC = {
    "npc_speech_max_chars": "int，NPC 一句话最多多少字（0=不限）",
    "move_cost_hours": "int，玩家每走一步消耗的小时数（>=0）",
    "damage_multiplier": "float，所有伤害倍率（>=0）",
    "heal_multiplier": "float，所有治疗倍率（>=0）",
    "extra_passable_terrain": "list[str]，额外可通行的地形名（如 forest）",
    "extra_solid_terrain": "list[str]，额外不可通行的地形名",
    "forbidden_words": "list[str]，NPC 对话中禁止出现的词，出现则被替换为***",
    "evolution_bias": "str，附加到所有演化提示词里的世界走向指令",
    "action_cost_multiplier": "float，自由动作耗时倍率",
    "hunger_rate": "float 0-50，NPC 每 24h 增长的饥饿",
    "fatigue_rate": "float 0-50，每 24h 增长的疲惫",
    "morale_rate": "float 0-50，每 24h 流失的士气",
    "eat_at": "float 0-100，饥饿到多少会去找吃的",
    "starve_at": "float 0-100，饥饿到多少开始掉血",
    "starve_hp_loss": "float 0-10，挨饿时每 6h 掉多少血",
    "anger_attack_at": "float 0-100，愤怒到多少会动手",
    "mood_override_hours": "int 0-240，LLM 指定的情绪压过公式多久",
    "fire_burnout_hours": "float 0-1000，火多久烧尽",
    "food_spoil_hours": "float 0-1000，食物多久腐败",
    "relation_decay": "float 0-5，关系每周向中立回落多少",
}

_RULE_BOUNDS = {
    "npc_speech_max_chars": (0, 2000),
    "move_cost_hours": (0, 48),
    "damage_multiplier": (0.0, 20.0),
    "heal_multiplier": (0.0, 20.0),
    "action_cost_multiplier": (0.0, 50.0),
    "hunger_rate": (0.0, 50.0), "fatigue_rate": (0.0, 50.0), "morale_rate": (0.0, 50.0),
    "anger_rate": (0.0, 50.0), "hostility_cool_hours": (0, 8760),
    "eat_at": (0.0, 100.0), "starve_at": (0.0, 100.0), "anger_attack_at": (0.0, 100.0),
    "starve_hp_loss": (0.0, 10.0),
    "mood_override_hours": (0, 240),
    "fire_burnout_hours": (0.0, 1000.0), "food_spoil_hours": (0.0, 1000.0),
    "relation_decay": (0.0, 5.0),
}


# Models rarely reproduce our exact identifiers ("hunger_rate" came back as
# "hunger_growth_multiplier").  Rejecting a plausible synonym silently would be
# the worst outcome: the world simply would not change and nobody would know.
_RULE_ALIASES: Dict[str, str] = {
    "hunger_growth_multiplier": "hunger_rate", "hunger_growth": "hunger_rate",
    "hunger_multiplier": "hunger_rate", "hunger_increase_rate": "hunger_rate",
    "starvation_rate": "hunger_rate", "hunger_speed": "hunger_rate",
    "fatigue_growth_multiplier": "fatigue_rate", "fatigue_growth": "fatigue_rate",
    "fatigue_multiplier": "fatigue_rate", "exhaustion_rate": "fatigue_rate",
    "morale_decay_rate": "morale_rate", "morale_loss_rate": "morale_rate",
    "morale_growth": "morale_rate",
    "anger_decay_rate": "anger_rate", "anger_growth": "anger_rate",
    "eat_threshold": "eat_at", "forage_at": "eat_at", "hunger_eat_threshold": "eat_at",
    "starvation_threshold": "starve_at", "starve_threshold": "starve_at",
    "starve_damage": "starve_hp_loss", "starvation_damage": "starve_hp_loss",
    "anger_attack_threshold": "anger_attack_at", "violence_threshold": "anger_attack_at",
    "speech_limit": "npc_speech_max_chars", "npc_speech_limit": "npc_speech_max_chars",
    "max_speech_chars": "npc_speech_max_chars", "speech_max_length": "npc_speech_max_chars",
    "movement_cost_hours": "move_cost_hours", "move_cost": "move_cost_hours",
    "damage_multiplier_global": "damage_multiplier", "global_damage_multiplier": "damage_multiplier",
    "fire_duration_hours": "fire_burnout_hours", "fire_lifetime": "fire_burnout_hours",
    "food_spoil_time": "food_spoil_hours", "spoil_hours": "food_spoil_hours",
    "relationship_decay": "relation_decay", "relation_decay_rate": "relation_decay",
}

# vocabulary so "threshold" can find "at", "growth" can find "rate", ...
_RULE_SYNONYMS = {
    "rate": {"rate", "growth", "speed", "multiplier", "perday", "perhour", "coef", "coefficient"},
    "at": {"at", "threshold", "limit", "point", "level", "when"},
    "hours": {"hours", "hour", "duration", "time", "lifetime"},
    "loss": {"loss", "damage", "drain", "cost"},
    "multiplier": {"multiplier", "factor", "scale", "ratio"},
}


def resolve_rule_key(key: str) -> Optional[str]:
    """Map a model-invented rule name onto a real one, or None."""
    if not key:
        return None
    raw = str(key).strip().strip("'\"")
    norm = raw.lower().replace("-", "_").replace(" ", "_")
    if norm in DEFAULTS:
        return norm
    if norm in _RULE_ALIASES:
        return _RULE_ALIASES[norm]
    tokens = [t for t in norm.split("_") if t]
    if not tokens:
        return None
    best, best_score = None, 0.0
    for canonical in DEFAULTS:
        want = canonical.split("_")
        hits = 0.0
        for w in want:
            pool = _RULE_SYNONYMS.get(w, {w})
            if any(t in pool or t == w for t in tokens):
                hits += 1.0
        score = hits / len(want)
        if score > best_score:
            best, best_score = canonical, score
    return best if best_score >= 0.75 else None


class RuntimeRules:
    def __init__(self, store, world_id: int, logger=None):
        self.store = store
        self.world_id = world_id
        self.logger = logger
        self.rules: Dict[str, Any] = dict(DEFAULTS)
        self.hooks: Dict[str, Callable] = {}
        self.sources: Dict[str, dict] = {}   # target -> patch row
        self.api: Dict[str, Any] = {}
        self.code_errors: List[str] = []
        self.patch_failures: List[str] = []
        self.load()

    def bind_api(self, api: Dict[str, Any]) -> None:
        """Hand the patch layer the surface it may monkey-patch.

        ``api`` exposes set_rule / add_hook / register_action / log plus small
        read-only snapshots.  Code patches are re-run on every load, so whatever
        they install comes back with the save.
        """
        self.api = dict(api)

        def _set(key, value, reason="", tick=None):
            # a rejected rule inside a patch must be visible, never silent
            ok, msg = self.set_rule(str(key), value, reason=str(reason),
                                    tick=self._tick() if tick is None else int(tick),
                                    source="code")
            if not ok:
                self.patch_failures.append(msg)
            return ok

        self.api["set_rule"] = _set
        self.api["add_hook"] = lambda hook, fn, reason="": self.add_hook_callable(
            str(hook), fn, reason=str(reason))
        self.api["rules"] = self.rules

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        self.rules = dict(DEFAULTS)
        self.hooks = {}
        self.sources = {}
        self.code_errors = []
        code_rows = []
        for row in self.store.list_patches(self.world_id, enabled_only=True):
            if row["kind"] == "rule":
                value = _loads(row["value_json"], None)
                ok, _ = self.validate_rule(row["target"], value)
                if ok:
                    self.rules[row["target"]] = value
                    self.sources[row["target"]] = row
                elif self.logger:
                    self.logger.warning("ignoring out-of-range rule %s=%r", row["target"], value)
            elif row["kind"] == "hook":
                try:
                    self.hooks[row["target"]] = compile_hook(row["expr"])
                    self.sources[row["target"]] = row
                except SandboxError as exc:
                    # a broken patch must never take the save down with it
                    self.store.disable_patch(row["id"])
                    self.store.commit()
                    if self.logger:
                        self.logger.error("disabled broken hook %s: %s", row["target"], exc)
            elif row["kind"] == "code":
                code_rows.append(row)
        for row in code_rows:
            self._run_code(row, reapply=True)

    def _tick(self) -> int:
        try:
            return int(self.store.get_meta("tick", 0) or 0)
        except Exception:  # noqa: BLE001
            return 0

    def _run_code(self, row: dict, reapply: bool = False) -> Tuple[bool, str]:
        source = row.get("expr") or ""
        if not self.api:
            return True, ""   # nothing bound yet; will run on bind
        self.patch_failures = []
        try:
            exec_patch(source, {"api": self.api})
        except SandboxError as exc:
            self.store.disable_patch(int(row["id"]))
            self.store.commit()
            msg = f"代码补丁被沙箱拒绝并已停用：{exc}"
            self.code_errors.append(msg)
            if self.logger:
                self.logger.error(msg)
            return False, msg
        except Exception as exc:  # noqa: BLE001 - a patch bug must not kill the game
            msg = f"代码补丁运行出错：{exc}"
            self.code_errors.append(msg)
            if self.logger:
                self.logger.error(msg)
            return False, msg
        if self.logger:
            self.logger.info("code patch #%s applied%s", row["id"], " (reload)" if reapply else "")
        return True, ""

    def run_code_patches(self) -> None:
        for row in self.store.list_patches(self.world_id, enabled_only=True):
            if row["kind"] == "code":
                self._run_code(row, reapply=True)

    def add_code_patch(self, source: str, reason: str = "", tick: int = 0,
                       source_tag: str = "llm") -> Tuple[bool, str]:
        err = validate_code(source, extra_names={"api"})
        if err:
            return False, f"代码补丁被沙箱拒绝：{err}"
        self.store.upsert_patch(self.world_id, "code", "code", expr=source.strip(),
                                reason=reason[:200], source=source_tag, tick=tick)
        self.store.commit()
        row = self.store.get_patch(self.world_id, "code", "code")
        ok, msg = self._run_code(row or {}, reapply=True)
        if not ok:
            return False, msg
        if self.patch_failures:
            return True, ("代码补丁已生效，但其中有条目被拒绝："
                          + "；".join(self.patch_failures[:3]))
        return True, "代码补丁已生效"

    def add_hook_callable(self, hook: str, fn: Callable, reason: str = "") -> bool:
        """In-memory hook registration (used by code patches; re-created on load)."""
        if hook not in HOOKS or not callable(fn):
            return False
        self.hooks[hook] = fn
        return True

    # --------------------------------------------------------------- queries
    def get(self, key: str, default: Any = None) -> Any:
        return self.rules.get(key, DEFAULTS.get(key, default))

    def call(self, hook: str, ctx: Dict[str, Any], default: Any = None) -> Any:
        fn = self.hooks.get(hook)
        if fn is None:
            return default
        try:
            out = fn(ctx)
        except Exception as exc:  # noqa: BLE001 - a bad hook must not crash play
            if self.logger:
                self.logger.warning("hook %s raised: %s", hook, exc)
            return default
        return default if out is None else out

    def is_terrain_passable(self, terrain: str, base_passable: bool) -> bool:
        if terrain in (self.get("extra_passable_terrain") or []):
            return True
        if terrain in (self.get("extra_solid_terrain") or []):
            return False
        return base_passable

    def sanitize_speech(self, text: str) -> str:
        text = text or ""
        limit = int(self.get("npc_speech_max_chars") or 0)
        if limit > 0 and len(text) > limit:
            text = text[:limit]
        for word in (self.get("forbidden_words") or []):
            if word and word in text:
                text = text.replace(word, "***")
        return text

    # ---------------------------------------------------------------- writes
    @staticmethod
    def validate_rule(key: str, value: Any) -> Tuple[bool, str]:
        if key not in DEFAULTS:
            return False, f"未知规则 {key}（可用：{', '.join(sorted(DEFAULTS))}）"
        bounds = _RULE_BOUNDS.get(key)
        if bounds:
            try:
                num = float(value)
            except (TypeError, ValueError):
                return False, f"{key} 需要数值"
            lo, hi = bounds
            if not (lo <= num <= hi):
                return False, f"{key} 需在 [{lo},{hi}] 内"
            value = int(num) if isinstance(DEFAULTS[key], int) else num
        if key in ("extra_passable_terrain", "extra_solid_terrain", "forbidden_words"):
            if not isinstance(value, list):
                return False, f"{key} 需要字符串数组"
            value = [str(v)[:24] for v in value][:12]
        if key in ("evolution_bias",) and not isinstance(value, str):
            return False, "evolution_bias 需要字符串"
        return True, ""

    def set_rule(self, key: str, value: Any, reason: str = "", tick: int = 0,
                 source: str = "llm") -> Tuple[bool, str]:
        canonical = resolve_rule_key(key)
        if canonical is None:
            msg = f"未知规则 {key}（可用：{', '.join(sorted(DEFAULTS))}）"
            self.patch_failures.append(msg)
            return False, msg
        if canonical != str(key).strip():
            if self.logger:
                self.logger.info("rule %r resolved to %r", key, canonical)
        key = canonical
        ok, msg = self.validate_rule(key, value)
        if not ok:
            self.patch_failures.append(msg)
            return False, msg
        if key in _RULE_BOUNDS:
            lo, hi = _RULE_BOUNDS[key]
            num = float(value)
            value = int(num) if isinstance(DEFAULTS[key], int) else num
        if key in ("extra_passable_terrain", "extra_solid_terrain", "forbidden_words"):
            value = [str(v)[:24] for v in value][:12]
        self.store.upsert_patch(self.world_id, "rule", key, value=value, reason=reason[:200],
                                source=source, tick=tick)
        self.store.commit()
        self.rules[key] = value
        self.sources[key] = self.store.get_patch(self.world_id, "rule", key) or {}
        return True, f"{key} = {value!r}"

    def add_hook(self, hook: str, expr: str, reason: str = "", tick: int = 0,
                 source: str = "llm") -> Tuple[bool, str]:
        if hook not in HOOKS:
            return False, f"未知钩子 {hook}（可用：{', '.join(HOOKS)}）"
        try:
            fn = compile_hook(expr)
        except SandboxError as exc:
            return False, f"钩子被沙箱拒绝：{exc}"
        self.store.upsert_patch(self.world_id, "hook", hook, expr=expr.strip(), reason=reason[:200],
                                source=source, tick=tick)
        self.store.commit()
        self.hooks[hook] = fn
        self.sources[hook] = self.store.get_patch(self.world_id, "hook", hook) or {}
        return True, f"钩子 {hook} 已生效"

    def remove(self, target: str) -> bool:
        row = self.sources.get(target) or self.store.get_patch(self.world_id, "code", target)
        if not row:
            return False
        self.store.delete_patch(int(row["id"]))
        self.store.commit()
        self.load()
        return True

    def code_patches(self) -> List[dict]:
        return [r for r in self.store.list_patches(self.world_id, enabled_only=True)
                if r["kind"] == "code"]

    # --------------------------------------------------------------- digests
    def rules_digest(self) -> str:
        diffs = {k: v for k, v in self.rules.items() if v != DEFAULTS.get(k)}
        if not diffs:
            return "（全部为默认值）"
        return "；".join(f"{k}={v!r}" for k, v in diffs.items())

    def describe(self) -> List[str]:
        out = []
        for key, value in self.rules.items():
            mark = " (改)" if value != DEFAULTS.get(key) else ""
            out.append(f"规则 {key} = {value!r}{mark}")
        for hook in self.hooks:
            row = self.sources.get(hook) or {}
            out.append(f"钩子 {hook} = {row.get('expr', '')}   ← {row.get('reason', '')}")
        for row in self.code_patches():
            out.append(f"代码补丁 #{row['id']} ← {row.get('reason', '')}\n"
                       + "\n".join("    " + ln for ln in (row.get("expr") or "").splitlines()))
        for err in self.code_errors:
            out.append(f"⚠ {err}")
        return out

    @staticmethod
    def doc() -> str:
        return "；".join(f"{k}: {v}" for k, v in _RULE_DOC.items())

    @staticmethod
    def hooks_doc() -> str:
        return (
            "can_enter(ctx{terrain,kind,has_object,x,y})->True/False/None；"
            "on_enter(ctx{terrain,x,y})->效果字典或None；"
            "damage(ctx{attacker,defender,base})->int或None；"
            "speech(ctx{npc,reply,mood})->str或None；"
            "move_cost(ctx{terrain})->int或None；"
            "evolve_bias(ctx{lod,name})->str或None。返回 None 表示沿用默认行为。"
        )


def _loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default
