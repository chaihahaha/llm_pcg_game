"""Prompt construction.

Rule 1: the system prompt is byte-identical for every call -> llama.cpp prompt
cache is reused across the whole session.
Rule 2: every prompt carries a machine hint ``[[TASK:x]]`` and ``[[SEED:n]]``
so the MockBackend can stand in for the real model.
Rule 3: prompts never contain raw world state, only bounded digests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

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
        '"resources":["矿产或农产品"],"summary":"国家概要(20-40字)","capital_x":整数,"capital_y":整数}],'
        '"relations":[{"a":"国名A","b":"国名B","kind":"敌对|盟友|贸易|冷战|朝贡|世仇",'
        '"value":-5到5的整数,"note":"关系由来(15-30字)"}]}\n'
        "nations 数量 3-5 个，capital 坐标范围 0-600。"
        "relations 给出 3-6 条国家之间的关系边，a/b 必须是上面 nations 里的国名。"
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
    maxc = size - 1
    return (
        f"[[TASK:chunk]][[BIOME:{_dominant(zone)}]][[SEED:{seed}]]\n"
        f"父级子区域「{zone.get('name','')}」概要：{str(zone.get('summary',''))[:120]}\n"
        f"附近地名：{feats or '无'}\n"
        f"生成其中一块 {size}x{size} 的地块（左上角世界坐标 {cx},{cy}）。"
        f"用 {size}x{size} 的网格（x 向右 0-{maxc}，y 向下 0-{maxc}）描述地貌。输出 JSON："
        '{"summary":"地块概要(30-60字)","weather":"当前天气(4-10字)",'
        '"patches":[{"terrain":"地形名","x":0,"y":0,"w":8,"h":16},'
        '{"terrain":"地形名","x":8,"y":0,"w":8,"h":9}],'
        '"features":[{"x":0,"y":0,"kind":"rock|ruin|plant|arcane|water|building","name":"名称","desc":"描述(15-35字)"}],'
        '"npcs":[{"x":0,"y":0,"name":"人名","race":"民族","role":"身份","personality":"性格",'
        '"appearance":"稳定可复用的外貌特征(15-30字)"}],'
        '"relations":[{"a":"人名","b":"另一个人名或国名","kind":"盟友|敌对|亲属|债主|师徒|雇主|同乡",'
        '"value":-5到5,"note":"关系由来(10-25字)"}]}\n'
        f"patches 为 3-6 个矩形地块，x/y 是左上角坐标(0-{maxc})，w/h 至少 2，"
        "矩形可重叠（后面的覆盖前面的），未覆盖处会按主导地形自动填充。"
        f"地形名只能取：{legend_text()}\n"
        "地形要与父区域一致（以主导地形为主，水/林/丘陵点缀），避免全图单一地形。"
        f"features 2-4 个，npcs 0-2 个，坐标必须是 0-{maxc} 的整数。"
        "relations 可为空数组；若有 npcs 且彼此相识（同乡、雇主、仇家等），给 1-3 条。"
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
                allow_tiles: bool, elapsed_hours: int = 0, period_hours: int = 0,
                bbox: Tuple[int, int, int, int] | None = None,
                scope_history: str = "") -> str:
    tile_rule = (
        "允许 type=tile 修改单格地形（谨慎使用，最多 1 处）。"
        if allow_tiles else "不要修改单格地形。"
    )
    leap = ""
    if period_hours and elapsed_hours > period_hours * 3 // 2:
        skipped = max(1, elapsed_hours // max(1, period_hours))
        leap = (
            f"⚠ 本层上一次演化到现在已经过去 {elapsed_hours} 小时（约 {skipped} 个周期），"
            "玩家不在场。请一次性推演这段时间的**累积**变化：可以有多次兴衰、"
            "渐进式的恶化或恢复、人口的迁移、关系的变化；不要只描写一瞬间的小事。"
            "summary 要概括这段时期的整体走向。\n"
        )
    scope_line = ""
    if bbox:
        sx, sy, sw, sh = bbox
        scope_line = (f"[[SCOPE:{sx},{sy},{sw},{sh}]]\n"
                      f"本层在世界坐标中的范围：x ∈ [{sx}, {sx + sw - 1}]，"
                      f"y ∈ [{sy}, {sy + sh - 1}]。\n")
    return (
        f"[[TASK:evolve_{lod_name}]][[SEED:{seed}]]\n"
        f"{scope_line}"
        f"当前时间：第 {day} 天 第 {hour} 时。\n"
        f"{leap}"
        f"【本层对象】{scope_desc}\n"
        f"【本层往事（你此前概括的长期走向，必须承接，不得当作没发生）】{scope_history or '（还没有）'}\n"
        f"【本层近期历史】{local_digest or '无'}\n"
        f"【更高层级(LOD 更高)的演化结果】{higher_digest or '无'}\n"
        f"【相邻对象现状】{neighbor_digest or '无'}\n"
        "请推演本层下一个时间步的变化。必须与更高层级的历史走向自洽（例如高层的战争/灾荒/魔力潮汐"
        "应当在这里留下痕迹）。变化要小而具体，避免每步都发生剧变。输出 JSON："
        '{"summary":"更新后的本层概要(30-80字)","history":"更新后的本层往事摘要(60-150字，'
        '承接已有往事、保留尚未解决的事与已离场/已解散的组织，只做滚动合并)",'
        '"weather":"天气(可为空字符串)",'
        '"events":[{"kind":"politics|economy|magic|weather|wildlife|conflict|discovery","text":"事件描述(20-50字)"}],'
        '"changes":[{"type":"new_object","x":整数,"y":整数,"kind":"物体类型","name":"名称","desc":"描述"},'
        '{"type":"new_npc","x":整数,"y":整数,"name":"人名","race":"民族","role":"身份","personality":"性格"},'
        '{"type":"npc","name":"已有NPC名","hp_delta":整数,"move":[dx,dy],"mood":"情绪",'
        '"status":"当前处境（如 昏迷/重伤/潜逃中/被俘/正常，无变化则省略）","note":"发生了什么"},'
        '{"type":"object","id":整数,"destroyed":true},'
        '{"type":"tile","x":整数,"y":整数,"terrain":"地形名","desc":"变化描述"},'
        '{"type":"relation","a_kind":"nation|npc","a_name":"名字","b_kind":"nation|npc","b_name":"名字",'
        '"kind":"盟友|敌对|贸易|债主|亲属|师徒|世仇","value":-5到5,"note":"关系变化原因(15-30字)"},'
        '{"type":"memory","name":"NPC名","kind":"fact|promise|grudge|debt|goal","text":"该NPC此后会记住的事"}]}\n'
        f"events 0-3 条，changes 0-4 条（可以只有 events）。{tile_rule}"
        "★ 所有 x/y 一律使用【世界坐标】（与上面给出的范围、以及相邻对象的坐标同一坐标系），"
        "不要使用 0-15 之类的地块内局部坐标；越界的改动会被丢弃。"
        "relation / memory 里出现的名字必须是【本层对象】或【相邻对象现状】里已有的名字，不要发明新名字。"
        "关系变化要克制：没有明确事件支撑就不要改动关系。"
        "★ NPC 的 name/race/role/appearance 是身份，**任何情况下都不得更改**；"
        "填入 memory 的内容必须与该 NPC 的身份与既往经历一致（不要一会儿是矿商一会儿是情报贩子）。"
        "★ 已经撤离、死亡、解散、被摧毁的人或组织，不得在后续事件里当作仍在此地正常活动；"
        "若要重新登场，必须先用 new_npc/new_object 说明其如何回来。"
        "★ 描述新物体时不得与其所在坐标的地形/既有物体矛盾（不要把一个东西说成在别处）。"
    )


def dialogue_task(seed: int, npc: Dict[str, Any], world_brief: str, place: str,
                  local_events: str, history: str, player_line: str, quest: str,
                  memories: str = "", relations: str = "", npc_history: str = "") -> str:
    return (
        f"[[TASK:dialogue]][[NPC:{npc.get('name','')}]][[SEED:{seed}]]\n"
        f"你扮演 NPC「{npc.get('name','')}」，{npc.get('race','')}，{npc.get('role','')}，"
        f"性格：{npc.get('personality','')}，当前情绪：{npc.get('mood','平静')}"
        + (f"，当前处境：{npc.get('status')}" if npc.get("status") else "")
        + "。\n"
        + (f"你的固定外貌（不得改口、不得添加未记录的身体特征）：{npc.get('appearance')}\n"
           if npc.get("appearance") else "")
        + (f"你最近亲历过的事（与这些保持一致）：{npc_history}\n" if npc_history else "")
        + f"所处位置：{place}。世界背景：{world_brief}\n"
        f"当地近期传闻：{local_events or '无'}\n"
        f"你记得的事（这是你的长期记忆，必须与之一致，不得遗忘或否认）：{memories or '（暂无）'}\n"
        f"你的人际关系（必须保持一致，可流露态度）：{relations or '（暂无）'}\n"
        f"你们之前的对话：\n{history or '（初次见面）'}\n"
        f"玩家说：{player_line}\n"
        f"玩家当前任务：{quest or '暂无'}\n"
        "用 NPC 的口吻回答（1-3 句，符合身份、性格与当地见闻，可以夹带线索或提出请求）。"
        "绝不能与上述记忆、亲历之事或先前对话矛盾；不要编造未记录的外貌特征，"
        "不要给出与记录不符的人数/伤亡数字；若玩家问到你不知道的事，就承认不知道。输出 JSON："
        '{"reply":"NPC说的话","mood":"回答后NPC的情绪","action":"none|quest|trade|attack|info",'
        '"action_data":{"quest_title":"当 action=quest 时给出","quest_summary":"任务概要"},'
        '"memories":[{"kind":"fact|promise|grudge|debt|goal","about":"涉及的人或地","text":"本次对话后你会记住的新事"}],'
        '"relation_changes":[{"kind":"盟友|敌对|债主|亲属|师徒","value":-5到5,"note":"为什么"}]}'
        "memories 只在本次对话真的产生了新信息时才给（0-2 条）。"
    )


def memory_consolidate_task(seed: int, npc: Dict[str, Any], memories: str) -> str:
    return (
        f"[[TASK:memory]][[NPC:{npc.get('name','')}]][[SEED:{seed}]]\n"
        f"NPC「{npc.get('name','')}」（{npc.get('race','')}，{npc.get('role','')}）"
        f"积累了很多记忆，需要压缩成更少的条目以免遗忘。\n"
        f"现有记忆：\n{memories}\n"
        "把这些记忆合并、去重、抽象成 5-8 条长期记忆：保留对身份、立场、恩怨、承诺、"
        "重大见闻至关重要的内容，丢弃无关细节，但**不得丢掉任何冲突、承诺、债务或人名**。输出 JSON："
        '{"memories":[{"kind":"fact|promise|grudge|debt|goal|history","about":"涉及的人或地","text":"记忆"}]}'
    )


def story_task(seed: int, player_brief: str, world_brief: str, recent: str,
               current: str, saga: str = "", world_events: str = "") -> str:
    return (
        f"[[TASK:story]][[SEED:{seed}]]\n"
        f"{player_brief}\n世界背景：{world_brief}\n"
        f"最近发生的事：{recent or '无'}\n"
        f"世界/区域层面的动向：{world_events or '无'}\n"
        f"玩家已经经历过的事件线（必须与之一致，可以承接或收束，不得当作没发生过）：\n{saga or '（还没有）'}\n"
        f"上一个任务：{current or '无'}\n"
        "请设计接下来推动故事的一条任务线。要与世界大势或当地传闻挂钩，规模可完成（数分钟到十几分钟）。"
        "如果上一条线索未完成，应当推进它而不是凭空另起炉灶。输出 JSON："
        '{"title":"任务名(4-10字)","summary":"任务背景(40-80字)","objective":"玩家要做什么(20-40字)",'
        '"stakes":"失败后果(15-30字)","hint":"去哪找线索(15-30字)"}'
    )


def audit_task(seed: int, material: str) -> str:
    return (
        f"[[TASK:audit]][[SEED:{seed}]]\n"
        "你是这个开放世界的一致性审计员。下面是同一个世界里的一份档案，包含："
        "玩家离开某地前后的物体与人物状态、按时间排列的事件流水、NPC 的长期记忆、"
        "人物之间的关系边、以及历次任务线。\n"
        "请找出其中**真正的问题**，只报告有证据的矛盾：\n"
        "A. 逻辑矛盾（同一实体前后状态冲突、已经死亡/被毁的东西又出现、位置瞬移、"
        "数值越界、时间倒流）\n"
        "B. 失忆（NPC 或任务线遗忘了先前已确立的事实、承诺、恩怨、人物）\n"
        "C. 与历史冲突（新描述与既有事件/设定/身份不一致）\n"
        "D. 世界停滞（某地在很长时间里毫无变化，却又与上层大事并存）\n"
        "没有把握的就不要报。输出 JSON："
        '{"contradictions":[{"category":"A|B|C|D","where":"涉及对象/地点","issue":"问题(30-60字)",'
        '"evidence":"档案中的依据(30-60字)","severity":"high|medium|low"}],'
        '"verdict":"整体一致性评价(40-80字)"}\n'
        "档案：\n" + material
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
