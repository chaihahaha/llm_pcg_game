"""The deterministic substrate: cheap, always-on micro-simulation.

Division of labour
------------------
The LLM is bad value for "is this NPC hungry" and "how long until it calms
down": it is slow (30-45 s per call), non-deterministic and imprecise.  Those
are *numbers*, and numbers belong to code.

So the world's trivial continuity is a plain simulation:

* closed-form integration — a 200-hour jump is **one** O(NPC) pass, not 200
  passes over every NPC,
* thresholds produce *discrete* consequences (eat, collapse, attack, die),
  with events timestamped at the interpolated moment they were crossed,
* it writes to the database like everything else, so it becomes history.

Then the LLM sits on top of it and does what it is actually good at:

1. **initialises** the substrate (an NPC's temperament, what it wants),
2. **reads** it as a numeric digest and turns it into meaning/narrative,
3. **perturbs** it at the per-NPC level through the ``sim_npc`` hook,
4. **rewrites** it wholesale through rules/hooks/code patches.

The result is a world that keeps living with zero LLM calls, and gets its
fantasy colour from the model rather than its arithmetic.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .rng import hash_unit

# needs live in npcs.state_json["sim"] so no schema migration is needed
FIELDS = ("hunger", "fatigue", "anger", "morale")

DEFAULT_SIM: Dict[str, float] = {
    "hunger": 20.0, "fatigue": 20.0, "anger": 0.0, "morale": 55.0,
    "meal_tick": 0, "mood_override": "", "mood_override_tick": -10 ** 9,
    "goal": "", "condition": "", "starving_since": -1, "warned": [],
}

EMPTY_VALUES = {"", "无", "none", "null", "n/a", "-", "未知"}

FOOD_KINDS = {"food", "crop", "meat", "ration", "plant"}
FIRE_KINDS = {"fire", "campfire", "flame", "brazier"}

MOOD_BY_STATE = (
    ("虚弱", lambda s, hp, mx: mx and hp / mx <= 0.3),
    ("愤怒", lambda s, hp, mx: s["anger"] >= 60),
    ("焦躁", lambda s, hp, mx: s["hunger"] >= 70),
    ("疲惫", lambda s, hp, mx: s["fatigue"] >= 80),
    ("消沉", lambda s, hp, mx: s["morale"] <= 25),
    ("愉悦", lambda s, hp, mx: s["morale"] >= 75),
)


def sim_state(npc: Dict[str, Any]) -> Dict[str, Any]:
    """This NPC's needs — seeded per individual the first time we see it.

    Starting everyone at the same numbers makes a crowd march in lockstep, so a
    brand new NPC gets a deterministic temperament derived from its id.  The
    LLM can still override any of it, and once stored the value is authoritative.
    """
    stored = (npc.get("state") or {}).get("sim")
    if stored:
        state = dict(DEFAULT_SIM)
        state.update(stored)
        return state
    key = npc.get("id") or npc.get("name") or 0
    return {
        **DEFAULT_SIM,
        "hunger": round(4.0 + hash_unit("sim", key, "hunger") * 55.0, 1),
        "fatigue": round(4.0 + hash_unit("sim", key, "fatigue") * 55.0, 1),
        "anger": round(hash_unit("sim", key, "anger") * 30.0, 1),
        "morale": round(30.0 + hash_unit("sim", key, "morale") * 55.0, 1),
    }


def derive_mood(state: Dict[str, Any], hp: int, hp_max: int, tick: int,
                override_hours: int = 12) -> str:
    """Mood is *computed* from needs, so it is consistent and free."""
    if state.get("mood_override") and tick - int(state.get("mood_override_tick", -10 ** 9)) \
            < override_hours:
        return str(state["mood_override"])
    for label, test in MOOD_BY_STATE:
        if test(state, hp, hp_max):
            return label
    return "平静"


class Simulator:
    """Advances needs, thresholds, objects and relations without the LLM."""

    MAX_EVENTS = 6

    def __init__(self, store, wm, rules, logger=None):
        self.store = store
        self.wm = wm
        self.rules = rules
        self.logger = logger
        self.world_id = wm.world_id

    # ------------------------------------------------------------------ entry
    def step(self, tick: int, hours: int, px: int, py: int) -> List[dict]:
        if hours <= 0:
            return []
        events: List[dict] = []
        dt = float(hours)
        for npc in self.store.list_npcs(self.world_id):
            try:
                events.extend(self._step_npc(npc, tick, dt, px, py))
            except Exception as exc:  # noqa: BLE001 - the sim must never break a turn
                if self.logger:
                    self.logger.warning("sim failed for npc %s: %s", npc.get("name"), exc)
        self._step_objects(tick, hours)
        self._decay_relations(tick, hours)
        self.store.commit()
        return events[: self.MAX_EVENTS]

    # ------------------------------------------------------------------- npcs
    def _step_npc(self, npc: dict, tick: int, dt: float, px: int, py: int) -> List[dict]:
        out: List[dict] = []
        state = sim_state(npc)
        before = {k: float(state[k]) for k in FIELDS}
        rates = {k: float(self.rules.get(f"{k}_rate", 0.0) or 0.0) for k in FIELDS}

        state["hunger"] = min(100.0, before["hunger"] + rates["hunger"] * dt / 24.0)
        state["fatigue"] = min(100.0, before["fatigue"] + rates["fatigue"] * dt / 24.0)
        state["anger"] = max(0.0, before["anger"] - rates["anger"] * dt / 24.0)
        state["morale"] = max(0.0, min(100.0, before["morale"] - rates["morale"] * dt / 24.0))

        # --- the LLM's fantasy intervention point, per NPC, per step ----------
        hooked = self.rules.call("sim_npc", {
            "name": npc["name"], "role": npc.get("role", ""), "race": npc.get("race", ""),
            "hunger": state["hunger"], "fatigue": state["fatigue"],
            "anger": state["anger"], "morale": state["morale"],
            "hp": npc["hp"], "hp_max": npc["hp_max"], "tick": tick,
        }, default=None)
        if isinstance(hooked, dict):
            for key in FIELDS:
                if isinstance(hooked.get(key), (int, float)):
                    state[key] = max(0.0, min(100.0, float(hooked[key])))
            mood_override = str(hooked.get("mood", "")).strip()[:16]
            if mood_override and mood_override.lower() not in EMPTY_VALUES:
                state["mood_override"] = mood_override
                state["mood_override_tick"] = tick
            if hooked.get("note"):
                out.append(self._event(tick, npc, "sim", str(hooked["note"])[:160]))
            if hooked.get("goal"):
                state["goal"] = str(hooked["goal"])[:40]

        hp = int(npc["hp"])
        patch: Dict[str, Any] = {}

        # --- starvation: HP loss only for the part of the window spent starving
        starve_at = float(self.rules.get("starve_at", 90.0))
        if state["hunger"] >= starve_at and dt > 0:
            span = max(1e-6, state["hunger"] - before["hunger"])
            frac = 1.0 if before["hunger"] >= starve_at else (state["hunger"] - starve_at) / span
            loss = int(float(self.rules.get("starve_hp_loss", 1.0)) * (dt * frac) / 6.0)
            if loss > 0:
                hp = max(0, hp - loss)
                patch["hp"] = hp
                if hp == 0 and npc["alive"]:
                    patch["alive"] = 0
                    patch["status"] = "死亡"
                    out.append(self._event(tick, npc, "death",
                                           f"{npc['name']}在饥困中倒下（第{tick // 24 + 1}天）"))
        elif before["hunger"] >= starve_at and state["hunger"] < starve_at:
            state["starving_since"] = -1
        elif state["hunger"] >= starve_at and state["starving_since"] < 0:
            state["starving_since"] = tick

        # --- eat: deterministic foraging for the nearest food -----------------
        if state["hunger"] >= float(self.rules.get("eat_at", 65.0)) and npc.get("alive", 1):
            ate = self._try_eat(npc, state, tick)
            if ate:
                out.append(ate)

        # --- violence: anger + a distrusted neighbour -------------------------
        if (npc.get("alive", 1) and state["anger"] >= float(self.rules.get("anger_attack_at", 70.0))
                and (abs(npc["x"] - px) + abs(npc["y"] - py)) <= 1):
            if not npc.get("hostile"):
                patch["hostile"] = 1
                state["condition"] = "暴怒"
                out.append(self._event(tick, npc, "conflict",
                                       f"{npc['name']}的怒气再也压不住，向附近的人扑了上去"))

        # --- collapse / resolve ----------------------------------------------
        if npc.get("alive", 1) and state["fatigue"] >= 95.0 and state["condition"] != "昏睡":
            state["condition"] = "昏睡"
            out.append(self._event(tick, npc, "sim", f"{npc['name']}撑不住了，就地昏睡过去"))
        elif state["condition"] == "昏睡" and state["fatigue"] < 60.0 and state["hunger"] < 60.0:
            state["condition"] = ""

        if (npc.get("alive", 1) and state["morale"] <= 15.0
                and "动摇" not in state.get("warned", [])):
            state.setdefault("warned", []).append("动摇")
            out.append(self._event(tick, npc, "sim", f"{npc['name']}的信念开始动摇"))

        # --- write back -------------------------------------------------------
        mood = derive_mood(state, hp, int(npc["hp_max"]), tick,
                           int(self.rules.get("mood_override_hours", 12)))
        if mood != npc.get("mood") or patch:
            patch["mood"] = mood
            patch["updated_tick"] = tick
        if state.get("goal"):
            state["condition"] = state.get("condition", "")
        new_state = dict(npc.get("state") or {})
        new_state["sim"] = state
        patch["state"] = new_state
        if patch:
            self.store.update_npc(npc["id"], **patch)
        return out

    def _try_eat(self, npc: dict, state: Dict[str, Any], tick: int) -> Optional[dict]:
        """Walk one step toward the nearest food and consume it."""
        foods = [o for o in self.store.objects_near(self.world_id, npc["x"], npc["y"], 4, limit=12)
                 if o.get("kind") in FOOD_KINDS]
        if not foods:
            return None
        target = min(foods, key=lambda o: abs(o["x"] - npc["x"]) + abs(o["y"] - npc["y"]))
        dx = (target["x"] > npc["x"]) - (target["x"] < npc["x"])
        dy = (target["y"] > npc["y"]) - (target["y"] < npc["y"])
        self.store.update_npc(npc["id"], x=npc["x"] + dx, y=npc["y"] + dy, updated_tick=tick)
        if abs(target["x"] - npc["x"]) + abs(target["y"] - npc["y"]) <= 1:
            state["hunger"] = max(0.0, state["hunger"] - 60.0)
            state["morale"] = min(100.0, state["morale"] + 10.0)
            state["meal_tick"] = tick
            self.store.destroy_object(target["id"], tick=tick, desc="被吃剩下的残渣。")
            return self._event(tick, npc, "sim", f"{npc['name']}吃掉了{target['name']}")
        return None

    # ---------------------------------------------------------------- objects
    def _step_objects(self, tick: int, hours: int) -> None:
        burnout = float(self.rules.get("fire_burnout_hours", 48.0))
        spoil = float(self.rules.get("food_spoil_hours", 72.0))
        for obj in self.store.list_objects(self.world_id, limit=400):
            # NB: created_tick may legitimately be 0 — `or tick` would silently
            # turn "created at the start of the world" into "created just now"
            created = obj.get("created_tick")
            age = tick - (tick if created is None else int(created))
            if obj.get("kind") in FIRE_KINDS and age >= burnout:
                self.store.destroy_object(obj["id"], tick=tick, desc="只剩一摊冷透的灰烬。")
            elif obj.get("kind") in FOOD_KINDS and age >= spoil:
                self.store.update_object(obj["id"], kind="rot", desc="已经腐败变质，散发恶臭。",
                                         updated_tick=tick)

    def _decay_relations(self, tick: int, hours: int) -> None:
        """Feelings fade: closed-form, one step per whole week since last touch."""
        per_week = float(self.rules.get("relation_decay", 1.0))
        if per_week <= 0:
            return
        week = 24 * 7
        for rel in self.store.list_relations(self.world_id, limit=200):
            value = int(rel["value"])
            if value == 0:
                continue
            last = int(rel.get("updated_tick") or 0)
            weeks = (tick - last) // week
            if weeks < 1:
                continue
            step = 1 if value > 0 else -1
            new_value = value - step * min(abs(value), int(per_week * weeks))
            self.store.conn.execute(
                "UPDATE relations SET value=?, updated_tick=? WHERE id=?",
                (new_value, tick, rel["id"]))

    # ----------------------------------------------------------------- helper
    def _event(self, tick: int, npc: dict, kind: str, text: str) -> dict:
        eid = self.store.add_event(self.world_id, tick, 3, npc.get("node_id"),
                                   npc["x"], npc["y"], kind, text[:200])
        return {"id": eid, "tick": tick, "lod": 3, "kind": kind, "summary": text,
                "x": npc["x"], "y": npc["y"], "node_id": npc.get("node_id"), "sim": True}


def sim_digest(npc: dict, tick: int) -> str:
    """Compact numeric view the LLM narrates from."""
    s = sim_state(npc)
    return (f"饥饿{s['hunger']:.0f} 疲惫{s['fatigue']:.0f} 愤怒{s['anger']:.0f} "
            f"士气{s['morale']:.0f}" + (f" 处境{s['condition']}" if s.get("condition") else ""))
