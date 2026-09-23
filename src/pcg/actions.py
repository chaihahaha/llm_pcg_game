"""Free-form actions: "我想向下挖进入地底" / "砍这棵树".

The player types whatever they want; the model turns it into an *action
program* — a small list of whitelisted effects, optionally plus a runtime patch
when the action implies changing how the game works.  Programs that look
re-useful are stored as named actions, so the same verb works forever after
without another round-trip.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import prompts
from .config import cfg_get
from .effects import EffectContext, apply_program
from .rng import hash_int
from .terrain import default_desc, symbol_of


class ActionResolver:
    def __init__(self, store, llm, cfg: Dict[str, Any], wm, rules, narrator=None, logger=None):
        self.store = store
        self.llm = llm
        self.cfg = cfg
        self.wm = wm
        self.rules = rules
        self.narrator = narrator
        self.logger = logger

    # ------------------------------------------------------------------ view
    def scene(self, player) -> str:
        wid = self.wm.world_id
        tile = self.wm.tile_at(player.x, player.y)
        lines = [
            f"你站在 ({player.x},{player.y})：{tile.get('name') or default_desc(tile['terrain'])}"
            f"（{tile['terrain']}）{(tile.get('desc') or '')[:50]}",
        ]
        objs = self.store.objects_at(wid, player.x, player.y)
        if objs:
            lines.append("脚下物体：" + "；".join(f"{o['name']}({o['kind']})" for o in objs))
        npcs = self.store.npcs_at(wid, player.x, player.y)
        if npcs:
            lines.append("脚下人物：" + "；".join(f"{n['name']}({n['role']})" for n in npcs))

        around = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
            t = self.store.get_tile(wid, player.x + dx, player.y + dy)
            if t:
                tag = f"{dx:+d},{dy:+d}:{symbol_of(t['terrain'])}"
                o = self.store.objects_at(wid, player.x + dx, player.y + dy)
                n = self.store.npcs_at(wid, player.x + dx, player.y + dy)
                if o:
                    tag += f"[{o[0]['name']}]"
                if n:
                    tag += f"[{n[0]['name']}]"
                around.append(tag)
        lines.append("相邻：" + " ".join(around))

        objs2 = self.store.objects_near(wid, player.x, player.y, 4, limit=8)
        if objs2:
            lines.append("附近物体：" + "；".join(
                f"#{o['id']}{o['name']}({o['kind']})[{o['x']},{o['y']}]" for o in objs2))
        npcs2 = self.store.npcs_near(wid, player.x, player.y, 5, limit=6)
        if npcs2:
            lines.append("附近人物：" + "；".join(
                f"{n['name']}({n['role']},{n['mood']})[{n['x']},{n['y']}]" for n in npcs2))
        events = self.store.events_for_player(wid, player.x, player.y, 32, limit=5)
        if events:
            lines.append("近来此地：" + "；".join(e["summary"][:50] for e in events))
        return "\n".join(lines)

    def inventory(self, player) -> str:
        inv = ((player.data.get("state") or {}).get("inventory") or {})
        return "、".join(f"{k}x{v}" for k, v in inv.items()) if inv else "（空）"

    def learned_digest(self) -> str:
        rows = self.store.list_actions(self.wm.world_id)
        if not rows:
            return ""
        return "；".join(f"{a['name']}（{a['title']}）" for a in rows[:12])

    # --------------------------------------------------------------- resolve
    def resolve(self, player, text: str) -> Dict[str, Any]:
        seed = hash_int("action", self.wm.world_id, player.x, player.y, text,
                        mod=2 ** 31)
        msg = prompts.action_task(
            seed, text,
            f"{player.data['name']} Lv{player.data['level']} HP{player.hp}/"
            f"{player.data['hp_max']} 金币{player.data['gold']}",
            self.scene(player), self.inventory(player),
            self.wm.world_bible(),
            self.rules.rules_digest(), self.rules.hooks_doc(), prompts.EFFECT_OPS_DOC,
            prompts.RULE_DOC, self.learned_digest(),
            int(cfg_get(self.cfg, "actions.max_hours", 24)),
        )
        msgs = prompts.build(self.wm.world_bible(), msg)
        data = self.llm.json(msgs, task="action", max_tokens=1100, temperature=0.6, default={})
        if not isinstance(data, dict):
            data = {}
        data.setdefault("feasible", False)
        data.setdefault("narrative", "你尝试了一下，但没有成功。")
        return data

    def apply(self, player, program: Dict[str, Any]) -> List[str]:
        ctx = EffectContext(self.store, self.wm.world_id, self.wm, self.rules, player,
                            tick=int(self.store.get_meta("tick", 0) or 0),
                            narrator=self.narrator, actions=self, logger=self.logger)
        return apply_program(ctx, program)

    def program_cost(self, program: Dict[str, Any]) -> int:
        try:
            hours = int(program.get("cost_hours", 0) or 0)
        except (TypeError, ValueError):
            hours = 0
        cap = int(cfg_get(self.cfg, "actions.max_hours", 24))
        hours = max(0, min(cap, hours))
        return int(round(hours * float(self.rules.get("action_cost_multiplier", 1.0))))

    # ----------------------------------------------------------- learned actions
    def learn(self, spec: Dict[str, Any], tick: int = 0) -> Tuple[bool, str]:
        name = str(spec.get("name") or spec.get("action") or "").strip()[:24]
        if not name:
            return False, ""
        title = str(spec.get("title") or name)[:40]
        description = str(spec.get("description") or spec.get("narrative") or "")[:200]
        payload = {
            "effects": spec.get("effects") or [],
            "cost_hours": spec.get("cost_hours", 0),
            "narrative": str(spec.get("narrative") or "")[:200],
        }
        if not payload["effects"]:
            return False, ""
        self.store.upsert_action(self.wm.world_id, name, title, description, payload, tick=tick)
        self.store.commit()
        return True, title or name

    def try_learned(self, text: str, player) -> Optional[Tuple[dict, dict]]:
        """Match the player's phrasing against stored actions.

        Returns ``(program, action_row)`` so the caller can replay it instantly.
        """
        needle = text.strip().lower()
        if not needle:
            return None
        for action in self.store.list_actions(self.wm.world_id):
            keys = [action["name"].lower(), action["title"].lower()]
            if any(k and (needle == k or needle.startswith(k) or k in needle) for k in keys):
                spec = action.get("spec") or {}
                program = {
                    "feasible": True,
                    "narrative": spec.get("narrative") or f"你再次{action['title']}。",
                    "cost_hours": spec.get("cost_hours", 0),
                    "effects": spec.get("effects") or [],
                    "action": action["name"],
                    "replayed": True,
                }
                return program, action
        return None
