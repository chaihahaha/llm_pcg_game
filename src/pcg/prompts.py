"""Prompt construction.

Rule 1: the system prompt is byte-identical for every call -> llama.cpp prompt
cache is reused across the whole session.
Rule 2: every prompt carries a machine hint ``[[TASK:x]]`` and ``[[SEED:n]]``
so the MockBackend can stand in for the real model.
Rule 3: prompts never contain raw world state, only bounded digests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .terrain import legend_text

SYSTEM = (
    "你是一个开放世界沙盒生成与推演引擎，服务于一个文字 roguelike 游戏。"
    "世界的历史演化、国家、民族、魔法、科技、矿产、农业、天气、地点描述与人物对话都由你决定。\n"
    "硬性要求：\n"
    "1) 只输出一个 JSON 对象，不要输出 markdown 代码块、不要解释、不要多余文字。\n"
    "2) 所有自然语言文本使用简体中文；描述要具体、感官化、有细节，避免空洞套话。\n"
    "3) 保持与前文设定的世界观、地理、人物、历史一致；不要引入与既有设定矛盾的元素。\n"
    "4) 名字要独特且符合当地文化语感，避免直接使用现实国家或真实人名。\n"
    "5) 数值必须在你被要求的范围内，字段名必须与要求的 schema 完全一致。\n"
    "6) 若某项信息未知，给出合理且克制的推断，不要写“未知”“无法确定”。\n"
)


def build(world_bible: str, task_text: str) -> List[dict]:
    """Assemble the message list.  Prefix order is stable for prompt caching."""
    messages: List[dict] = [{"role": "system", "content": SYSTEM}]
    if world_bible:
        messages.append({"role": "system", "content": world_bible})
    messages.append({"role": "user", "content": task_text})
    return messages


def world_bible(world: Dict[str, Any], nations: Sequence[Dict[str, Any]], extra: str = "") -> str:
    """Compact, stable digest of the top LOD.  Bounded to ~320 tokens."""
    lines = [
        "[世界圣经]",
        f"世界名：{world.get('name','')}｜纪元：{world.get('era','')}",
        f"魔法体系：{str(world.get('magic_system',''))[:80]}",
        f"科技基线：{str(world.get('tech_baseline',''))[:80]}",
        f"世界概要：{str(world.get('summary',''))[:160]}",
    ]
    if nations:
        ns = "；".join(
            f"{n.get('name','')}({n.get('race','')},{n.get('gov','')},科技{n.get('tech','')},"
            f"魔法{n.get('magic','')})" for n in nations[:6]
        )
        lines.append(f"主要国家：{ns}")
    if extra:
        lines.append(extra)
    return "\n".join(lines)


# ---------------------------------------------------------------- generation

def world_task(seed: int, hint: str = "") -> str:
    return (
        f"[[TASK:world]][[SEED:{seed}]]\n"
        "请从零创造一个原创奇幻世界。"
        + (f"\n额外要求：{hint}" if hint else "")
        + "\n输出 JSON："
        '{"name":"世界名(2-4字)","era":"当前纪元名","cosmology":"宇宙观/世界起源(30-60字)",'
        '"magic_system":"魔法体系与代价(40-80字)","tech_baseline":"主流科技水平(20-40字)",'
        '"summary":"世界现状(60-120字)","nations":[{"name":"国名","race":"主体民族",'
        '"gov":"政体","tech":"科技标签(2-4字)","magic":"魔法亲和(2-6字)",'
        '"resources":["矿产或农产品"],"summary":"国家概要(20-40字)","capital_x":整数,"capital_y":整数}]}\n'
        "nations 数量 3-5 个，capital 坐标范围 0-600。"
    )


def region_task(seed: int, world_name: str, rx: int, ry: int, size: int, hint: str = "") -> str:
    return (
        f"[[TASK:region]][[SEED:{seed}]]\n"
        f"在世界「{world_name}」中生成一块区域（左上角坐标 {rx},{ry}，边长 {size} 格）。"
        + (f"\n已知：{hint}" if hint else "")
        + "\n输出 JSON："
        '{"name":"区域名","climate":"气候(6-12字)","biome_mix":[["地形名",权重],...],'
        '"cultures":["当地文化特征"],"nations_present":["在此有势力的国家名或自治势力"],'
        '"features":[{"name":"显著地点","kind":"地貌类型","desc":"描述(20-40字)"}],'
        '"hazards":["危险"],"summary":"区域概要(50-100字)"}\n'
        f"biome_mix 权重之和为 1，地形名只能取：{legend_text()}。features 2-4 个。"
    )


def zone_task(seed: int, region: Dict[str, Any], zx: int, zy: int, size: int) -> str:
    feats = "；".join(f"{f.get('name','')}({f.get('desc','')})"
                      for f in (region.get("data", {}).get("features") or [])[:3])
    return (
        f"[[TASK:zone]][[SEED:{seed}]]\n"
        f"父级区域「{region.get('name','')}」，气候 {region.get('data',{}).get('climate','')}，"
        f"概要：{str(region.get('summary',''))[:120]}\n"
        f"区域内已知显著地点：{feats or '无'}\n"
        f"请细化其中一块子区域（左上角 {zx},{zy}，边长 {size} 格）。输出 JSON："
        '{"name":"地点名","biome_mix":[["地形名",权重],...],'
        '"features":[{"name":"地物名","kind":"ruins|water|rock|arcane|building","desc":"描述(20-40字)"}],'
        '"hazards":["危险"],"npc_seeds":[{"name":"人名","race":"民族","role":"身份","personality":"性格(10-20字)"}],'
        '"summary":"子区域概要(40-80字)"}\n'
        f"地形名只能取：{legend_text()}。features 2-4 个，npc_seeds 1-3 个。"
    )


def chunk_task(seed: int, zone: Dict[str, Any], cx: int, cy: int, size: int) -> str:
    feats = "；".join(f"{f.get('name','')}" for f in (zone.get("data", {}).get("features") or [])[:3])
    return (
        f"[[TASK:chunk]][[BIOME:{_dominant(zone)}]][[SEED:{seed}]]\n"
        f"父级子区域「{zone.get('name','')}」概要：{str(zone.get('summary',''))[:120]}\n"
        f"附近地名：{feats or '无'}\n"
        f"生成其中一块 {size}x{size} 的地块（左上角世界坐标 {cx},{cy}）。输出 JSON："
        '{"summary":"地块概要(30-60字)","weather":"当前天气(4-10字)",'
        f'"rows":["..TT^^......~~..","...共{size}个字符串,每个{size}字符..."],'
        '"features":[{"x":0,"y":0,"kind":"rock|ruin|plant|arcane|water|building","name":"名称","desc":"描述(15-35字)"}],'
        '"npcs":[{"x":0,"y":0,"name":"人名","race":"民族","role":"身份","personality":"性格"}]}\n'
        f"rows 必须是恰好 {size} 个字符串，每个字符串恰好 {size} 个字符，且只能使用下列字符：\n"
        f"{legend_text()}\n"
        "重要：字符必须紧密相连，不要空格、逗号或任何分隔符，不要自己发明字符。示例（16 字符）：..TT^^......~~..\n"
        "地形要与父区域一致（以主导地形为主，水/林/丘陵点缀）。"
        f"features 2-4 个，npcs 0-2 个，坐标必须是 0-{size-1} 的整数。"
    )


def tile_task(seed: int, tile: Dict[str, Any], zone_name: str, chunk_summary: str,
              neighbors: str, objects: str) -> str:
    return (
        f"[[TASK:tile]][[TERRAIN:{tile.get('terrain','grass')}]][[SEED:{seed}]]\n"
        f"所在子区域：{zone_name}｜地块概要：{str(chunk_summary)[:100]}\n"
        f"相邻地形：{neighbors or '未知'}\n"
        f"格上物体：{objects or '无'}\n"
        f"请为这一格（世界坐标 {tile.get('x')},{tile.get('y')}）写具体描述。输出 JSON："
        '{"name":"地形短名(2-6字)","desc":"感官化描述(30-60字)","detail":"再细看会注意到的细节(20-50字)"}'
    )


# ---------------------------------------------------------------- evolution

def evolve_task(lod_name: str, seed: int, scope_desc: str, local_digest: str,
                higher_digest: str, neighbor_digest: str, hour: int, day: int,
                allow_tiles: bool) -> str:
    tile_rule = (
        "允许 type=tile 修改单格地形（谨慎使用，最多 1 处）。"
        if allow_tiles else "不要修改单格地形。"
    )
    return (
        f"[[TASK:evolve_{lod_name}]][[SEED:{seed}]]\n"
        f"当前时间：第 {day} 天 第 {hour} 时。\n"
        f"【本层对象】{scope_desc}\n"
        f"【本层近期历史】{local_digest or '无'}\n"
        f"【更高层级(LOD 更高)的演化结果】{higher_digest or '无'}\n"
        f"【相邻对象现状】{neighbor_digest or '无'}\n"
        "请推演本层下一个时间步的变化。必须与更高层级的历史走向自洽（例如高层的战争/灾荒/魔力潮汐"
        "应当在这里留下痕迹）。变化要小而具体，避免每步都发生剧变。输出 JSON："
        '{"summary":"更新后的本层概要(30-80字)","weather":"天气(可为空字符串)",'
        '"events":[{"kind":"politics|economy|magic|weather|wildlife|conflict|discovery","text":"事件描述(20-50字)"}],'
        '"changes":[{"type":"new_object","x":整数,"y":整数,"kind":"物体类型","name":"名称","desc":"描述"},'
        '{"type":"new_npc","x":整数,"y":整数,"name":"人名","race":"民族","role":"身份","personality":"性格"},'
        '{"type":"npc","name":"已有NPC名","hp_delta":整数,"move":[dx,dy],"mood":"情绪","note":"发生了什么"},'
        '{"type":"object","id":整数,"destroyed":true},'
        '{"type":"tile","x":整数,"y":整数,"terrain":"地形名","desc":"变化描述"}]}\n'
        f"events 0-3 条，changes 0-4 条（可以只有 events）。{tile_rule}"
        "新增物体的坐标必须落在本层的坐标范围内。"
    )


def dialogue_task(seed: int, npc: Dict[str, Any], world_brief: str, place: str,
                  local_events: str, history: str, player_line: str, quest: str) -> str:
    return (
        f"[[TASK:dialogue]][[NPC:{npc.get('name','')}]][[SEED:{seed}]]\n"
        f"你扮演 NPC「{npc.get('name','')}」，{npc.get('race','')}，{npc.get('role','')}，"
        f"性格：{npc.get('personality','')}，当前情绪：{npc.get('mood','平静')}。\n"
        f"所处位置：{place}。世界背景：{world_brief}\n"
        f"当地近期传闻：{local_events or '无'}\n"
        f"你们之前的对话：\n{history or '（初次见面）'}\n"
        f"玩家说：{player_line}\n"
        f"玩家当前任务：{quest or '暂无'}\n"
        "用 NPC 的口吻回答（1-3 句，符合身份、性格与当地见闻，可以夹带线索或提出请求）。输出 JSON："
        '{"reply":"NPC说的话","mood":"回答后NPC的情绪","action":"none|quest|trade|attack|info",'
        '"action_data":{"quest_title":"当 action=quest 时给出","quest_summary":"任务概要"}}'
    )


def story_task(seed: int, player_brief: str, world_brief: str, recent: str,
               current: str) -> str:
    return (
        f"[[TASK:story]][[SEED:{seed}]]\n"
        f"{player_brief}\n世界背景：{world_brief}\n"
        f"最近发生的事：{recent or '无'}\n"
        f"上一个任务：{current or '无'}\n"
        "请设计接下来推动故事的一条任务线。要与世界大势或当地传闻挂钩，规模可完成（数分钟到十几分钟）。输出 JSON："
        '{"title":"任务名(4-10字)","summary":"任务背景(40-80字)","objective":"玩家要做什么(20-40字)",'
        '"stakes":"失败后果(15-30字)","hint":"去哪找线索(15-30字)"}'
    )


def flavor_task(seed: int, attacker: str, defender: str, result: str, place: str) -> str:
    return (
        f"[[TASK:flavor]][[SEED:{seed}]]\n"
        f"{place} 中，{attacker} 对 {defender} 发起攻击，结果：{result}。\n"
        "用一句 15-30 字的战报描写，只写画面，不要数值。输出 JSON：{\"text\":\"...\"}"
    )


# ---------------------------------------------------------------- helpers

def _dominant(node: Dict[str, Any]) -> str:
    from .terrain import normalize

    mix = (node.get("data") or {}).get("biome_mix") or []
    best, best_w = "grass", -1.0
    for entry in mix:
        try:
            name, weight = entry[0], float(entry[1])
        except (IndexError, TypeError, ValueError):
            continue
        if weight > best_w:
            best, best_w = normalize(str(name)), weight
    return best


def digest_nations(nations: Sequence[Dict[str, Any]]) -> str:
    return "｜".join(f"{n['name']}({n.get('tech','')}/{n.get('magic','')})" for n in nations[:6])


def digest_events(events: Sequence[Dict[str, Any]], limit: int = 8) -> str:
    return "；".join(str(e.get("summary", ""))[:60] for e in events[:limit])


def digest_objects(objects: Sequence[Dict[str, Any]], limit: int = 12) -> str:
    return "；".join(f"[{o.get('x')},{o.get('y')}]{o.get('name','')}({o.get('kind','')})"
                     for o in objects[:limit])


def digest_npcs(npcs: Sequence[Dict[str, Any]], limit: int = 10) -> str:
    return "；".join(f"[{n.get('x')},{n.get('y')}]{n.get('name','')}({n.get('role','')},"
                     f"{n.get('mood','')},HP{n.get('hp')})" for n in npcs[:limit])


def digest_tiles(tiles: Sequence[Dict[str, Any]], limit: int = 24) -> str:
    return "；".join(f"[{t.get('x')},{t.get('y')}]{t.get('terrain','')}" for t in tiles[:limit])
