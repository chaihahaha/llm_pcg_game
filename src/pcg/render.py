"""Terminal rendering: the 16x16 local view and per-tile inspection text.

No GUI — just text.  "Clicking a tile" is `inspect <x> <y>`.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .terrain import TERRAIN, symbol_of

OBJ_SYM = {
    "rock": "o", "ruin": "+", "plant": "v", "tree": "T", "arcane": "%",
    "building": "b", "item": "i", "ore": "*", "water": "~", "track": ".",
    "altar": "&", "corpse": "x", "creature": "c", "object": "o", "feature": "o",
}

RUBBLE_SYM = "x"  # destroyed objects stay on the map as ruins


def _npc_char(npc: dict) -> str:
    name = str(npc.get("name") or "?")
    return name[0] if name else "?"


def render_view(wm, cx: int, cy: int, size: int = 16, player=None) -> str:
    store = wm.store
    world_id = wm.world_id
    x0, y0 = cx - size // 2, cy - size // 2
    tiles = wm.view_tiles(cx, cy, size)

    grid: Dict[Tuple[int, int], str] = {}
    for (tx, ty), tile in tiles.items():
        grid[(tx, ty)] = symbol_of(tile["terrain"])

    objs = store.objects_near(world_id, cx, cy, size // 2 + 1, limit=200, alive_only=False)
    for o in objs:
        # ruins stay visible: destruction is a change of state, not erasure
        grid[(o["x"], o["y"])] = (OBJ_SYM.get(o.get("kind", "object"), "o")
                                  if o.get("alive", 1) else RUBBLE_SYM)

    npcs = store.npcs_near(world_id, cx, cy, size // 2 + 1, limit=100)
    for n in npcs:
        grid[(n["x"], n["y"])] = _npc_char(n)

    if player is not None and player.alive:
        grid[(player.x, player.y)] = "@"

    header1 = "    " + "".join(str((x0 + i) // 10 % 10) for i in range(size))
    header2 = "    " + "".join(str((x0 + i) % 10) for i in range(size))
    lines = [header1, header2]
    for j in range(size):
        row = "".join(grid.get((x0 + i, y0 + j), "?") for i in range(size))
        marker = " <-你" if player is not None and y0 + j == player.y else ""
        lines.append(f"{y0 + j:4d}{row}{marker}")

    lines.append("")
    lines.append("符号：" + " ".join(f"{v[0]}={k}" for k, v in TERRAIN.items() if k in
                                     ("water", "grass", "tall_grass", "forest", "hill",
                                      "mountain", "ruins", "farm", "road", "arcane")))
    lines.append("      o=物体 +=遗迹 %=魔力 b=建筑 v=植物 @=你 汉字=人物")
    if npcs:
        near = sorted(npcs, key=lambda n: abs(n["x"] - cx) + abs(n["y"] - cy))[:6]
        lines.append("附近人物：" + "，".join(
            f"{n['name']}({n['role']},{n['mood']})[{n['x']},{n['y']}]" for n in near))
    return "\n".join(lines)


def render_tile_info(wm, x: int, y: int, describe: bool = True) -> str:
    store = wm.store
    world_id = wm.world_id
    tile = wm.ensure_tile_desc(x, y) if describe else wm.tile_at(x, y)
    chunk = wm.ensure_node(3, x, y)
    zone = wm.ensure_node(2, x, y)
    region = wm.ensure_node(1, x, y)

    lines = [
        f"=== 格子 ({x}, {y}) ===",
        f"地形：{tile.get('name') or tile['terrain']}（{tile['terrain']}）",
        f"归属：{region.get('name','')} › {zone.get('name','')} › {chunk.get('name','')}",
        f"描述：{tile.get('desc') or '（尚未细看）'}",
    ]
    detail = (tile.get("data") or {}).get("detail")
    if detail:
        lines.append(f"细节：{detail}")

    objects = store.objects_at(world_id, x, y)
    if objects:
        lines.append("物体：")
        for o in objects:
            hp = f"，耐久 {o['hp']}/{o['hp_max']}" if o.get("hp_max") else ""
            lines.append(f"  · {o['name']}（{o['kind']}{hp}）：{o['desc'] or '没有更多说明。'}")
    debris = [o for o in store.objects_at(world_id, x, y, alive_only=False) if not o.get("alive", 1)]
    if debris:
        lines.append("残迹：")
        for o in debris:
            when = (o.get("state") or {}).get("destroyed_tick")
            lines.append(f"  · {o['name']}（已毁于第{int(when)//24+1}天）：{o['desc']}")

    npcs = store.npcs_at(world_id, x, y)
    if npcs:
        lines.append("人物：")
        for n in npcs:
            lines.append(f"  · {n['name']}（{n['race']} {n['role']}，{n['mood']}，"
                         f"HP {n['hp']}/{n['hp_max']}）：{n['personality'] or '看不出情绪。'}")

    events = store.events_for_player(world_id, x, y, 1, limit=3)
    if events:
        lines.append("此处发生过：" + "；".join(e["summary"][:50] for e in events))
    return "\n".join(lines)


def render_status(player, tick: int, weather: str = "", quest: Optional[dict] = None,
                  llm_stats: Optional[dict] = None) -> str:
    day, hour = int(tick) // 24 + 1, int(tick) % 24
    bits = [f"[第{day}天 {hour:02d}:00]",
            f"HP {player.hp}/{player.data['hp_max']}",
            f"Lv{player.data['level']}",
            f"ATK {player.data['atk']}",
            f"金币 {player.data['gold']}",
            f"位置 ({player.x},{player.y})"]
    if weather:
        bits.append(f"天气 {weather}")
    if quest:
        bits.append(f"任务「{quest['title']}」")
    if llm_stats:
        bits.append(f"LLM 实调{llm_stats.get('calls',0)}/缓存{llm_stats.get('cache_hits',0)}")
    return " | ".join(bits)


def render_help() -> str:
    return (
        "指令：\n"
        "  look / l                 刷新周围 16x16 视图\n"
        "  move <w|a|s|d|n|s|e|nw|...>   向一个方向移动一格\n"
        "  goto <x> <y>             覆写到指定坐标（一次一格，可重复）\n"
        "  inspect [x y]            查看格子说明（“点击”格子，缺省为脚下）\n"
        "  talk [名字] [说的话]      与相邻人物对话（LLM 实时生成）\n"
        "  attack [名字]            攻击相邻目标（Python 战斗逻辑）\n"
        "  wait [小时]              原地等待，推进世界演化\n"
        "  story                    查看当前任务\n"
        "  newstory                 让说书人立刻生成新任务\n"
        "  journal [n]              查看最近事件\n"
        "  world                    查看世界圣经（国家/魔法/科技）\n"
        "  stats                    查看数据库与 LLM 统计\n"
        "  auto <n>                 自动探索 n 步\n"
        "  help / quit"
    )
