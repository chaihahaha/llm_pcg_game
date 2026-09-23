"""Interpreter for LLM-authored action programs.

The model never gets to run arbitrary Python against world state.  Instead it
composes a small, whitelisted effect language, and *this* module is the only
thing that touches the database.  Every effect is bounded (coordinates must be
next to the player, terrain comes from the fixed vocabulary, numbers are
clamped), so a hallucinated or over-ambitious action degrades into "nothing
happened" instead of corrupting the world.

Effects that change the *program* rather than the world (``rule``, ``hook``,
``learn``) are delegated to :mod:`pcg.rules` and the action registry.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .terrain import is_solid, normalize, normalize_kind

MAX_REACH = 3          # Chebyshev distance from the player for world edits
MAX_EFFECTS = 12

_OBJECT_SOLID = {"rock": 1, "ruin": 1, "building": 1, "tree": 1, "ore": 1, "altar": 1}


class EffectContext:
    """Everything an effect may touch, in one place."""

    def __init__(self, store, world_id: int, wm, rules, player, tick: int,
                 narrator=None, actions=None, logger=None):
        self.store = store
        self.world_id = world_id
        self.wm = wm
        self.rules = rules
        self.player = player
        self.tick = tick
        self.narrator = narrator
        self.actions = actions
        self.logger = logger

    # -- helpers
    def resolve_xy(self, eff: Dict[str, Any]) -> Optional[tuple]:
        if self.player is None:
            return None
        px, py = self.player.x, self.player.y
        try:
            if eff.get("x") is not None or eff.get("y") is not None:
                x = int(eff.get("x", px))
                y = int(eff.get("y", py))
            else:
                x = px + int(eff.get("dx", 0) or 0)
                y = py + int(eff.get("dy", 0) or 0)
        except (TypeError, ValueError):
            return None
        if max(abs(x - px), abs(y - py)) > MAX_REACH:
            return None
        return x, y


def apply_program(ctx: EffectContext, program: Dict[str, Any]) -> List[str]:
    """Apply an action program; return human-readable log lines."""
    log: List[str] = []
    effects = program.get("effects") if isinstance(program, dict) else None
    if not isinstance(effects, list):
        return log
    for eff in effects[:MAX_EFFECTS]:
        if not isinstance(eff, dict):
            continue
        try:
            line = _apply_one(ctx, eff)
        except Exception as exc:  # noqa: BLE001 - one bad effect must not kill the turn
            line = None
            if ctx.logger:
                ctx.logger.warning("effect %s failed: %s", eff.get("op"), exc)
        if line:
            log.append(line)
    return log


def _apply_one(ctx: EffectContext, eff: Dict[str, Any]) -> Optional[str]:
    op = str(eff.get("op") or "").strip()
    store, wid = ctx.store, ctx.world_id

    if op == "none":
        return None

    if op == "set_tile":
        xy = ctx.resolve_xy(eff)
        if not xy:
            return None
        x, y = xy
        terrain = normalize(str(eff.get("terrain", "")))
        desc = str(eff.get("desc", ""))[:200]
        store.update_tile_fields(wid, x, y, terrain=terrain, desc=desc or None, updated_tick=ctx.tick)
        name = str(eff.get("name", ""))[:30]
        if name:
            store.update_tile_fields(wid, x, y, name=name)
        store.commit()
        return f"（{x},{y}）变为{terrain}" + (f"：{desc}" if desc else "")

    if op == "create_object":
        xy = ctx.resolve_xy(eff)
        if not xy:
            return None
        x, y = xy
        kind = normalize_kind(str(eff.get("kind", "object")))
        name = str(eff.get("name", ""))[:30] or "无名之物"
        oid = store.add_object(wid, x, y, kind, name, str(eff.get("desc", ""))[:200],
                               solid=int(eff.get("solid", _OBJECT_SOLID.get(kind, 0)) or 0),
                               tick=ctx.tick)
        store.commit()
        return f"出现「{name}」(#{oid})"

    if op == "destroy_object":
        try:
            oid = int(eff.get("id"))
        except (TypeError, ValueError):
            return None
        obj = store.get_object(oid)
        if not obj or not obj.get("alive", 1):
            return None
        store.destroy_object(oid, tick=ctx.tick, desc=str(eff.get("desc", ""))[:200])
        store.commit()
        return f"「{obj['name']}」被摧毁"

    if op == "create_npc":
        xy = ctx.resolve_xy(eff)
        if not xy:
            return None
        x, y = xy
        name = str(eff.get("name", ""))[:20]
        if not name or store.find_npc_by_name(wid, name):
            return None
        store.add_npc(wid, x, y, name, race=str(eff.get("race", "")),
                      role=str(eff.get("role", "")), personality=str(eff.get("personality", ""))[:60],
                      appearance=str(eff.get("appearance", ""))[:80], tick=ctx.tick)
        store.commit()
        return f"{name} 出现在（{x},{y}）"

    if op == "move_player":
        xy = ctx.resolve_xy(eff)
        if not xy:
            return None
        x, y = xy
        store.update_player(wid, x=x, y=y, updated_tick=ctx.tick)
        if ctx.player is not None:
            ctx.player.refresh()
        ctx.wm.mark_explored(x, y)
        store.commit()
        return f"你移动到（{x},{y}）"

    if op in ("damage_player", "heal_player", "damage_npc", "heal_npc"):
        return _vital(ctx, op, eff)

    if op == "grant":
        item = str(eff.get("item", "")).strip()[:30]
        if not item or ctx.player is None:
            return None
        try:
            qty = max(1, min(20, int(eff.get("qty", 1) or 1)))
        except (TypeError, ValueError):
            qty = 1
        inv = dict((ctx.player.data.get("state") or {}).get("inventory") or {})
        inv[item] = int(inv.get(item, 0)) + qty
        state = dict(ctx.player.data.get("state") or {})
        state["inventory"] = inv
        store.update_player(wid, state=state)
        ctx.player.refresh()
        store.commit()
        return f"获得「{item}」x{qty}"

    if op == "set_weather":
        if ctx.player is None:
            return None
        chunk = ctx.wm.ensure_node(3, ctx.player.x, ctx.player.y)
        data = dict(chunk.get("data") or {})
        data["weather"] = str(eff.get("text", ""))[:20]
        store.update_node(chunk["id"], data=data, tick=ctx.tick)
        store.commit()
        return f"天气：{data['weather']}"

    if op == "event":
        text = str(eff.get("text", "")).strip()[:200]
        if not text:
            return None
        xy = ctx.resolve_xy(eff) or (0, 0)
        store.add_event(wid, ctx.tick, 3, None, xy[0], xy[1],
                        str(eff.get("kind", "action"))[:20], text)
        store.commit()
        return f"※ {text}"

    if op == "memory":
        name = str(eff.get("npc", "")).strip()
        npc = store.find_npc_by_name(wid, name) if name else None
        text = str(eff.get("text", "")).strip()[:200]
        if not npc or not text:
            return None
        store.add_memory(wid, npc["id"], ctx.tick, str(eff.get("kind", "fact"))[:16], text,
                         about=str(eff.get("about", ""))[:32])
        store.prune_memories(npc["id"], keep=24)
        store.commit()
        return None

    if op == "relation":
        a_kind = str(eff.get("a_kind", "npc")).lower()
        b_kind = str(eff.get("b_kind", "npc")).lower()
        a_name, b_name = str(eff.get("a_name", ""))[:32], str(eff.get("b_name", ""))[:32]
        if not _exists(ctx, a_kind, a_name) or not _exists(ctx, b_kind, b_name):
            return None
        store.upsert_relation(wid, a_kind, a_name, b_kind, b_name,
                              kind=str(eff.get("kind", "中立"))[:16],
                              value=int(eff.get("value", 0) or 0),
                              note=str(eff.get("note", ""))[:160], tick=ctx.tick)
        store.commit()
        return f"关系变化：{a_name} -{eff.get('kind')}-> {b_name}"

    if op == "rule":
        ok, msg = ctx.rules.set_rule(str(eff.get("key", "")), eff.get("value"),
                                     reason=str(eff.get("reason", "")), tick=ctx.tick,
                                     source="player")
        return f"※ 世界规则改变：{msg}" if ok else (f"（规则被拒绝：{msg}）" if msg else None)

    if op == "hook":
        ok, msg = ctx.rules.add_hook(str(eff.get("hook", "")), str(eff.get("expr", "")),
                                     reason=str(eff.get("reason", "")), tick=ctx.tick,
                                     source="player")
        return f"※ 世界法则被改写：{msg}" if ok else f"（钩子被拒绝：{msg}）"

    if op == "remove_patch":
        target = str(eff.get("target", ""))
        return f"※ 世界法则复原：{target}" if ctx.rules.remove(target) else None

    if op == "learn" and ctx.actions is not None:
        ok, msg = ctx.actions.learn(eff, tick=ctx.tick)
        return f"※ 学会新动作「{msg}」" if ok else None

    if ctx.logger:
        ctx.logger.info("unknown effect op: %s", op)
    return None


def _vital(ctx: EffectContext, op: str, eff: Dict[str, Any]) -> Optional[str]:
    try:
        amount = int(eff.get("amount", 1) or 1)
    except (TypeError, ValueError):
        amount = 1
    amount = max(1, min(50, amount))
    store, wid = ctx.store, ctx.world_id
    if op == "damage_player" and ctx.player is not None:
        amount = max(1, int(round(amount * float(ctx.rules.get("damage_multiplier", 1.0)))))
        ctx.player.damage(amount)
        store.commit()
        return f"你受到 {amount} 点伤害（HP {ctx.player.hp}）"
    if op == "heal_player" and ctx.player is not None:
        amount = max(1, int(round(amount * float(ctx.rules.get("heal_multiplier", 1.0)))))
        ctx.player.heal(amount)
        store.commit()
        return f"你恢复了 {amount} 点生命（HP {ctx.player.hp}）"
    name = str(eff.get("npc", "")).strip() or str(eff.get("name", "")).strip()
    npc = store.find_npc_by_name(wid, name) if name else None
    if not npc or not npc.get("alive", 1):
        return None
    if op == "damage_npc":
        hp = max(0, int(npc["hp"]) - amount)
        store.update_npc(npc["id"], hp=hp, alive=1 if hp > 0 else 0, hostile=1, updated_tick=ctx.tick)
        store.commit()
        return f"{npc['name']} 受到 {amount} 点伤害" + ("，倒下了" if hp == 0 else "")
    hp = min(int(npc["hp_max"]), int(npc["hp"]) + amount)
    store.update_npc(npc["id"], hp=hp, updated_tick=ctx.tick)
    store.commit()
    return f"{npc['name']} 恢复了 {amount} 点生命"


def _exists(ctx: EffectContext, kind: str, name: str) -> bool:
    if kind == "nation":
        return ctx.store.get_nation(ctx.world_id, name) is not None
    return ctx.store.find_npc_by_name(ctx.world_id, name) is not None


def terrain_passable(rules, terrain: str) -> bool:
    return rules.is_terrain_passable(terrain, not is_solid(terrain))
