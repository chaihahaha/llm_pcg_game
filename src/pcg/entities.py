"""Player, movement and combat — deliberately plain Python.

The LLM is *not* in the loop for these; only combat flavour text goes to it.
This keeps latency and cost low while the model focuses on world content.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .db import Store
from .rng import hash_int, hash_unit
from .terrain import is_solid

_DIRS = {
    "n": (0, -1), "north": (0, -1), "北": (0, -1), "w": (0, -1), "up": (0, -1),
    "s": (0, 1), "south": (0, 1), "南": (0, 1), "down": (0, 1),
    "a": (-1, 0), "west": (-1, 0), "西": (-1, 0), "left": (-1, 0),
    "d": (1, 0), "east": (1, 0), "东": (1, 0), "right": (1, 0),
    "ne": (1, -1), "nw": (-1, -1), "se": (1, 1), "sw": (-1, 1),
}


class Player:
    def __init__(self, store: Store, world_id: int, rules=None):
        self.store = store
        self.world_id = world_id
        self.rules = rules
        row = store.get_player(world_id)
        if row is None:
            raise RuntimeError("player row missing")
        self.data = row

    # -- convenience accessors
    @property
    def x(self) -> int:
        return int(self.data["x"])

    @property
    def y(self) -> int:
        return int(self.data["y"])

    @property
    def hp(self) -> int:
        return int(self.data["hp"])

    @property
    def alive(self) -> bool:
        return self.hp > 0

    def refresh(self) -> None:
        self.data = self.store.get_player(self.world_id) or self.data

    def _write(self, **fields) -> None:
        self.store.update_player(self.world_id, **fields)
        self.refresh()

    # -- actions
    def move_to(self, x: int, y: int) -> Tuple[bool, str]:
        if not self.alive:
            return False, "你已经倒下了。"
        ok, reason = can_enter(self.store, self.world_id, x, y, self.rules)
        if not ok:
            return False, reason
        self._write(x=x, y=y)
        return True, ""

    def move(self, direction: str) -> Tuple[bool, str]:
        delta = _DIRS.get(direction.lower())
        if delta is None:
            return False, f"未知方向：{direction}"
        return self.move_to(self.x + delta[0], self.y + delta[1])

    def damage(self, amount: int) -> None:
        self._write(hp=max(0, self.hp - max(0, int(amount))))

    def heal(self, amount: int) -> None:
        self._write(hp=min(int(self.data["hp_max"]), self.hp + max(0, int(amount))))

    @property
    def inventory(self) -> Dict[str, int]:
        return dict((self.data.get("state") or {}).get("inventory") or {})

    def gain_xp(self, amount: int) -> List[str]:
        msgs: List[str] = []
        xp = int(self.data["xp"]) + int(amount)
        level = int(self.data["level"])
        while xp >= level * 20:
            xp -= level * 20
            level += 1
            self.store.conn.execute(
                "UPDATE player SET hp_max=hp_max+5, atk=atk+1, hp=hp_max, level=?, xp=? WHERE world_id=?",
                (level, xp, self.world_id),
            )
            msgs.append(f"你升到了 {level} 级！")
        self.store.update_player(self.world_id, xp=xp, level=level)
        self.refresh()
        return msgs


# --------------------------------------------------------------------- queries

def can_enter(store: Store, world_id: int, x: int, y: int, rules=None) -> Tuple[bool, str]:
    """May the player step onto (x, y)?

    ``rules`` (a :class:`pcg.rules.RuntimeRules`) lets the world's own laws —
    and any patch the model installed — rewrite the answer.
    """
    tile = store.get_tile(world_id, x, y)
    objs = store.objects_at(world_id, x, y) if tile is not None else []

    if tile is not None:
        blocked_reason = None
        if is_solid(tile["terrain"]):
            blocked_reason = f"{tile.get('name') or tile['terrain']} 挡住了去路。"
        if blocked_reason is None:
            for obj in objs:
                if obj.get("solid"):
                    blocked_reason = f"{obj['name']} 挡住了去路。"
                    break
        if rules is not None:
            passable = rules.is_terrain_passable(tile["terrain"], not is_solid(tile["terrain"]))
            hook = rules.call("can_enter", {
                "terrain": tile["terrain"], "x": x, "y": y, "base_blocked": blocked_reason is not None,
                "has_object": bool(objs), "name": tile.get("name") or "",
            }, default=None)
            if hook is True:
                blocked_reason = None          # the world's laws now allow it
            elif hook is False:
                return False, f"某种力量阻止你进入{tile['terrain']}。"
            elif hook is None and passable and blocked_reason and blocked_reason.startswith(
                    tile.get("name") or tile["terrain"]):
                blocked_reason = None          # terrain was made passable by a rule
        if blocked_reason:
            return False, blocked_reason

    npcs = store.npcs_at(world_id, x, y)
    if npcs:
        names = "、".join(n["name"] for n in npcs[:2])
        return False, f"{names} 在那里，先交谈或攻击（talk / attack）。"
    return True, ""


def find_npc(store: Store, world_id: int, name: str, x: int, y: int, radius: int = 2) -> Optional[dict]:
    if name:
        npc = store.find_npc_by_name(world_id, name)
        if npc:
            return npc
    here = store.npcs_at(world_id, x, y)
    if here:
        return here[0]
    near = store.npcs_near(world_id, x, y, radius)
    if near:
        return near[0]
    return None


# ---------------------------------------------------------------------- combat

def _roll(seed_key, lo: int, hi: int) -> int:
    return lo + hash_int(*seed_key, mod=(hi - lo + 1))


def attack(store: Store, world_id: int, player: Player, npc: dict, tick: int,
           rules=None) -> Dict[str, Any]:
    """One player attack + one retaliation.  Pure Python, deterministic."""
    if not npc or not npc.get("alive", 1):
        return {"ok": False, "log": ["目标已经倒下。"]}

    seed_key = ("atk", world_id, npc["id"], tick, player.hp)
    log: List[str] = []
    dmg = max(1, int(player.data["atk"]) + _roll(seed_key + ("p",), 0, 3) - int(npc["def"]))
    if rules is not None:
        base = max(1, int(round(dmg * float(rules.get("damage_multiplier", 1.0)))))
        hooked = rules.call("damage", {
            "attacker": player.data.get("name", "你"), "defender": npc["name"],
            "base": base, "npc_role": npc.get("role", ""), "npc_race": npc.get("race", ""),
        }, default=None)
        try:
            dmg = max(1, int(hooked)) if hooked is not None else base
        except (TypeError, ValueError):
            dmg = base
    npc_hp = int(npc["hp"]) - dmg
    log.append(f"你击中 {npc['name']}，造成 {dmg} 点伤害。")
    killed = npc_hp <= 0
    if killed:
        store.update_npc(npc["id"], hp=0, alive=0, updated_tick=tick)
        log.append(f"{npc['name']} 倒下了。")
        xp = 5 + hash_int("xp", npc["id"], mod=8)
        log.extend(player.gain_xp(xp))
        gold = hash_int("gold", npc["id"], mod=12)
        if gold:
            store.update_player(world_id, gold=int(player.data["gold"]) + gold)
            player.refresh()
            log.append(f"你搜到了 {gold} 枚硬币。")
        store.add_event(world_id, tick, 3, npc.get("node_id"), npc["x"], npc["y"], "combat",
                        f"玩家击败了{npc['name']}（{npc.get('role','')}）")
    else:
        store.update_npc(npc["id"], hp=npc_hp, hostile=1, updated_tick=tick)
        if hash_unit(seed_key + ("ret",)) < 0.75:
            back = max(1, int(npc["atk"]) + _roll(seed_key + ("n",), 0, 2) - int(player.data["def"]))
            player.damage(back)
            log.append(f"{npc['name']} 反击，你受到 {back} 点伤害。（HP {player.hp}/{player.data['hp_max']}）")
        else:
            log.append(f"{npc['name']} 没能打中你。")
    store.commit()
    return {"ok": True, "killed": killed, "log": log,
            "result": "击杀" if killed else f"造成{dmg}伤害"}


def npc_turn(store: Store, world_id: int, npc: dict, player_x: int, player_y: int, tick: int) -> List[str]:
    """Extremely small NPC AI: hostile NPCs adjacent to the player attack once."""
    log: List[str] = []
    if not npc.get("alive", 1) or not npc.get("hostile"):
        return log
    dist = abs(npc["x"] - player_x) + abs(npc["y"] - player_y)
    if dist <= 1:
        player = store.get_player(world_id)
        if player and player["hp"] > 0:
            dmg = max(1, int(npc["atk"]) - int(player["def"]))
            store.update_player(world_id, hp=max(0, int(player["hp"]) - dmg))
            log.append(f"{npc['name']} 扑了上来，你受到 {dmg} 点伤害。")
    store.commit()
    return log
