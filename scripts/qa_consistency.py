#!/usr/bin/env python3
"""Long-run consistency QA for the LLM-PCG world.

The scenario the project must survive:

    explore area A  ->  walk far away to B  ->  a long time passes  ->  come
    back to A  ->  is A still the same place, evolved coherently?

It checks structural invariants (identity, monotonic life/death, bounded
movement, no duplicates, history preserved) plus "did it actually evolve".
Semantic checks (does the prose contradict itself?) are reported as material
for review, since only a human can judge those.

    python3 scripts/qa_consistency.py --mock
    python3 scripts/qa_consistency.py --distance 240 --hours 400 -v
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
from typing import List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pcg.config import load_config  # noqa: E402
from pcg.game import Game  # noqa: E402

A_START = (72, 72)


class Snapshot:
    def __init__(self, game: Game, cx: int, cy: int, radius: int = 24):
        self.cx, self.cy, self.radius = cx, cy, radius
        self.tick = game.tick()
        store, wid = game.store, game.world_id
        self.tiles = {(t["x"], t["y"]): t for t in
                      store.list_tiles(wid, cx - radius, cy - radius, cx + radius, cy + radius)}
        self.objects = {o["id"]: o for o in store.objects_near(wid, cx, cy, radius, limit=500,
                                                       alive_only=False)}
        self.npcs = {n["id"]: n for n in
                     store.npcs_near(wid, cx, cy, radius, alive_only=False, limit=500)}
        self.nodes = {}
        for lod in (1, 2, 3):
            node = store.get_node(wid, lod, cx, cy)
            if node is None:  # fall back to the node that actually contains the point
                for cand in store.nodes_in_rect(wid, lod, cx - 600, cy - 600, cx + 600, cy + 600):
                    if cand["x"] <= cx < cand["x"] + cand["w"] and cand["y"] <= cy < cand["y"] + cand["h"]:
                        node = cand
                        break
            self.nodes[lod] = node
        self.events = store.events_for_player(wid, cx, cy, radius, limit=200)


def snapshot(game: Game, cx: int, cy: int, radius: int) -> Snapshot:
    return Snapshot(game, cx, cy, radius)


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.0f}%" if b else "n/a"


_STATUS_PATH = ""


def say(msg: str) -> None:
    """Print AND append to the status file.

    A long QA run is monitored from another shell; buffered stdout has bitten
    this project before, so progress is mirrored to a file we can tail.
    """
    print(msg, flush=True)
    if _STATUS_PATH:
        try:
            with open(_STATUS_PATH, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")
        except OSError:
            pass


def check_identity(before: Snapshot, after: Snapshot, problems: List[str],
                   notes: List[str]) -> None:
    gone = [oid for oid in before.objects if oid not in after.objects]
    if gone:
        names = ", ".join(f"#{i} {before.objects[i]['name']}" for i in gone[:5])
        problems.append(f"A 区有 {len(gone)} 个物体行被物理删除（历史丢失）：{names}")
    else:
        notes.append(f"物体身份保持：{len(before.objects)} 个 id 全部保留")

    destroyed = [oid for oid, o in before.objects.items()
                 if o.get("alive") and oid in after.objects and not after.objects[oid].get("alive")]
    if destroyed:
        kept = all(after.objects[i]["desc"] for i in destroyed)
        notes.append(f"离开期间被摧毁 {len(destroyed)} 个物体"
                     + ("（均保留残迹文本，历史可追溯）" if kept else "（部分残迹描述为空）"))
        if not kept:
            problems.append("被摧毁的物体没有留下任何残迹描述")

    moved = []
    for oid, o in before.objects.items():
        n = after.objects.get(oid)
        if n and (n["x"], n["y"], n["kind"]) != (o["x"], o["y"], o["kind"]):
            moved.append((o, n))
    if moved:
        problems.append(f"{len(moved)} 个物体位置/类型被改变（应只改状态，不应挪动物体）："
                        + ", ".join(f"#{a['id']}{a['name']}->({b['x']},{b['y']})" for a, b in moved[:4]))
    else:
        notes.append("物体坐标与类型稳定")

    renamed = [o for oid, o in before.objects.items()
               if oid in after.objects and after.objects[oid]["name"] != o["name"] and o["name"]]
    if renamed:
        problems.append(f"{len(renamed)} 个物体被改名（历史名不应丢失）")


def check_lifecycle(before: Snapshot, after: Snapshot, problems: List[str], notes: List[str]) -> None:
    revived = []
    for nid, a in before.npcs.items():
        b = after.npcs.get(nid)
        if not b:
            problems.append(f"NPC #{nid} {a['name']} 被删除（应保留为死亡记录）")
            continue
        if not a["alive"] and b["alive"]:
            revived.append(b["name"])
        if b["hp"] > b["hp_max"]:
            problems.append(f"NPC {b['name']} HP {b['hp']} 超过上限 {b['hp_max']}")
        if b["hp"] < 0:
            problems.append(f"NPC {b['name']} HP 为负 {b['hp']}")
        if b["alive"] and b["hp"] == 0:
            problems.append(f"NPC {b['name']} 存活但 HP=0")
    if revived:
        problems.append(f"死者复活：{revived}")
    if not revived and before.npcs:
        notes.append("生死状态单调（死者未复活）")

    names = [n["name"] for n in after.npcs.values()]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        problems.append(f"A 区出现重名 NPC：{sorted(dupes)}")

    for nid, a in before.npcs.items():
        b = after.npcs.get(nid)
        if not b or not b["alive"]:
            continue
        elapsed_hours = max(0, after.tick - before.tick)
        allowed = max(6, elapsed_hours)          # <=1 tile per elapsed hour
        dist = abs(b["x"] - a["x"]) + abs(b["y"] - a["y"])
        if dist > allowed:
            problems.append(f"NPC {b['name']} 在 {elapsed_hours} 小时内位移 {dist} 格"
                            f"（上限 {allowed}），疑似瞬移"
                            f"（({a['x']},{a['y']})->({b['x']},{b['y']})）")


def check_evolution(before: Snapshot, after: Snapshot, problems: List[str], notes: List[str]) -> None:
    obj_changed = sum(1 for oid, o in before.objects.items()
                      if oid in after.objects and after.objects[oid]["updated_tick"] > o["updated_tick"])
    npc_changed = sum(1 for nid, n in before.npcs.items()
                      if nid in after.npcs and after.npcs[nid]["updated_tick"] > n["updated_tick"])
    node_advanced = {lod: (before.nodes.get(lod) is not None and after.nodes.get(lod) is not None
                           and after.nodes[lod]["updated_tick"] > before.nodes[lod]["updated_tick"])
                     for lod in (1, 2, 3)}
    new_events = [e for e in after.events if e["tick"] > before.tick]
    new_objects = [oid for oid, o in after.objects.items()
                       if oid not in before.objects and o.get("alive")]

    notes.append(f"离开期间：物体更新 {obj_changed}/{len(before.objects)}，"
                 f"NPC 更新 {npc_changed}/{len(before.npcs)}，新增物体 {len(new_objects)}，"
                 f"A 区新事件 {len(new_events)}")
    notes.append("各层节点是否推进：" + ", ".join(
        f"lod{lod}={'是' if v else '否'}" for lod, v in node_advanced.items()))

    if obj_changed == 0 and npc_changed == 0 and not new_events and not new_objects:
        problems.append("A 区在玩家离开期间完全没有演化（世界被冻结）")
    for lod in (2, 3):
        if not node_advanced[lod]:
            problems.append(f"A 区 lod{lod} 节点 updated_tick 未推进（该层未参与演化）")
    if not new_events:
        problems.append("A 区没有任何本地事件记录（演化结果未落库到该坐标）")


def check_history_preserved(before: Snapshot, after: Snapshot,
                            problems: List[str], notes: List[str]) -> None:
    """A described tile must keep its prose; a returned explorer should recognise it."""
    kept = 0
    lost = []
    for key, t in before.tiles.items():
        if not t["desc"]:
            continue
        a = after.tiles.get(key)
        if a is None:
            lost.append(key)
        elif not a["desc"]:
            lost.append(key)
        else:
            kept += 1
    if lost:
        problems.append(f"{len(lost)} 个已描述格子的描述丢失：{lost[:4]}")
    if kept:
        notes.append(f"格物描述保留：{kept} 个已描述格子的文本未丢失")


def check_relationships(game: Game, notes: List[str], problems: List[str]) -> None:
    store, wid = game.store, game.world_id
    rels = store.list_relations(wid)
    nations = store.list_nations(wid)
    nation_rel = sum(1 for r in rels if r["a_kind"] == "nation")
    npc_rel = sum(1 for r in rels if r["a_kind"] == "npc")
    notes.append(f"关系图：{len(rels)} 条边（国家-国家 {nation_rel}，含 NPC {npc_rel}）")
    if nations and nation_rel == 0:
        problems.append("国家之间没有任何关系边（外交关系图缺失）")
    if npc_rel == 0:
        problems.append("NPC 之间/与势力没有任何关系边")


def check_memory(game: Game, notes: List[str], problems: List[str]) -> None:
    store, wid = game.store, game.world_id
    with_mem = 0
    for npc in store.npcs_near(wid, A_START[0], A_START[1], 40, limit=100):
        if store.npc_memories(npc["id"], limit=1):
            with_mem += 1
    notes.append(f"NPC 长期记忆：{with_mem} 个 NPC 有记忆条目")
    if store.count_npcs(wid) and with_mem == 0:
        problems.append("没有任何 NPC 拥有长期记忆（对话后不记事，必然失忆）")


def check_story(game: Game, notes: List[str], problems: List[str]) -> None:
    store, wid = game.store, game.world_id
    hist = store.story_history(wid, limit=50)
    notes.append(f"任务线历史：{len(hist)} 条，当前：" +
                 (store.current_story(wid) or {}).get("title", "无"))
    if len(hist) >= 2:
        # a new arc must be able to see earlier arcs; we surface the material
        notes.append("任务标题序列：" + " -> ".join(h["title"] for h in reversed(hist)))


def audit_with_llm(game: Game, a_before: "Snapshot", a_after: "Snapshot",
                   notes: List[str], problems: List[str]) -> dict:
    """Ask the model to review the archive for contradictions the assertions miss.

    Structural checks cannot tell that an NPC "forgot" a promise; only reading
    the record can.  This is the semantic half of the QA.
    """
    from pcg import prompts

    store, wid = game.store, game.world_id
    parts: List[str] = []

    if a_before is not None:
        parts.append("## 离开 A 区前的物体")
        for o in a_before.objects.values():
            parts.append(f"- #{o['id']} {o['name']}({o['kind']}) @({o['x']},{o['y']}) "
                         f"HP{o['hp']}：{(o['desc'] or '')[:60]}")
    if a_after is not None:
        parts.append("\n## 返回 A 区后的物体")
        for o in a_after.objects.values():
            parts.append(f"- #{o['id']} {o['name']}({o['kind']}) @({o['x']},{o['y']}) "
                         f"HP{o['hp']} tick{o['updated_tick']}：{(o['desc'] or '')[:60]}")
    parts.append("\n## A 区/世界事件流水（按时间）")
    for e in sorted(store.recent_events(wid, limit=30), key=lambda x: x["tick"]):
        lod = {0: "世界", 1: "区域", 2: "子区域", 3: "地块"}.get(e["lod"], "?")
        parts.append(f"- 第{e['tick'] // 24 + 1}天[{lod}] {e['summary'][:80]}")
    parts.append("\n## NPC 长期记忆")
    for npc in store.npcs_near(wid, *A_START, 60, limit=12):
        mem = store.npc_memories(npc["id"], limit=8)
        if mem:
            parts.append(f"- {npc['name']}（{npc['role']}，{npc['mood']}）")
            parts.extend(f"    · [{m['kind']}] {m['text'][:70]}" for m in mem)
    parts.append("\n## 关系图")
    for r in store.list_relations(wid, limit=30):
        parts.append(f"- {r['a_name']} --{r['kind']}({r['value']})--> {r['b_name']}"
                     f"：{(r['note'] or '')[:40]}")
    parts.append("\n## 任务线（旧 -> 新）")
    for s in reversed(store.story_history(wid, limit=12)):
        parts.append(f"- [{s['state']}] {s['title']}：{(s['summary'] or '')[:70]}")
    parts.append("\n## 对话摘录（部分）")
    for d in store.conn.execute(
            "SELECT npc_id, role, tick, content FROM dialogue WHERE world_id=? ORDER BY id LIMIT 30",
            (wid,)):
        parts.append(f"- 第{d['tick'] // 24 + 1}天 {d['role']}: {d['content'][:70]}")

    material = "\n".join(parts)
    msgs = prompts.build(game.wm.world_bible(), prompts.audit_task(1234, material))
    data = game.llm.json(msgs, task="audit", max_tokens=1200, temperature=0.2, default={})
    verdict = str(data.get("verdict") or "")
    findings = [c for c in (data.get("contradictions") or []) if isinstance(c, dict)]
    say(f"\n## LLM 一致性审计：{verdict}")
    for c in findings[:10]:
        sev = c.get("severity", "?")
        say(f"  [{c.get('category', '?')}/{sev}] {c.get('where', '')}：{c.get('issue', '')}")
        say(f"        依据：{c.get('evidence', '')}")
    if not findings:
        say("  （未发现矛盾）")
    high = [c for c in findings if c.get("severity") == "high"]
    if high:
        problems.append(f"LLM 审计发现 {len(high)} 处高危矛盾")
    elif findings:
        notes.append(f"LLM 审计发现 {len(findings)} 处低/中危问题（见报告）")
    return {"verdict": verdict, "contradictions": findings}


def run(args) -> int:
    global _STATUS_PATH
    _STATUS_PATH = args.status
    cfg = load_config(None, {
        "llm": {"mock": bool(args.mock), "cache_path": args.cache},
        "game": {"db_path": args.db},
    })
    if os.path.exists(args.db):
        os.remove(args.db)
    game = Game(cfg, verbose=args.verbose)
    game.auto_story = True
    if not args.mock and not game.llm.ping():
        say("✗ 无法连接本地大模型，请确认服务在运行；否则本次长跑毫无意义。")
        game.close()
        return 2

    problems: List[str] = []
    notes: List[str] = []
    try:
        world = game.new_world(seed=args.seed, hint=args.hint)
        say(f"世界：{world['name']}（{world['era']}）  后端={game.llm.backend_name}")
        if game.llm.degraded:
            say("⚠ 后端已降级为 Mock —— 本次结果不具备真实性")

        say(f"[1] 在 A{A_START} 附近探索…")
        _quiet(game, "auto 6")
        _quiet(game, "auto 6")
        _quiet(game, "newstory")
        a_before = snapshot(game, *A_START, args.radius)

        # meet and talk to an NPC if one is around, so memory has something to store
        npcs = game.store.npcs_near(game.world_id, *A_START, 30, limit=1)
        if npcs:
            _quiet(game, f"talk {npcs[0]['name']} 你好，你在这里做什么？")
            _quiet(game, "talk 我们以前见过吗？")

        far = (A_START[0] + args.distance, A_START[1] + args.distance)
        say(f"[2] 远行到 {far} …")
        game.teleport(*far)
        _quiet(game, "newstory")
        npcs_far = game.store.npcs_near(game.world_id, *far, 30, limit=2)
        for n in npcs_far:
            _quiet(game, f"talk {n['name']} 这边有什么新鲜事？")
        say(f"[3] 在远处等待 {args.hours} 小时…")
        _quiet(game, f"wait {args.hours}")
        far_snap = snapshot(game, *far, args.radius)

        say("[4] 返回 A …")
        game.teleport(*A_START)
        for _ in range(args.settle):
            _quiet(game, "wait 6")
        a_after = snapshot(game, *A_START, args.radius)

        check_identity(a_before, a_after, problems, notes)
        check_lifecycle(a_before, a_after, problems, notes)
        check_evolution(a_before, a_after, problems, notes)
        check_history_preserved(a_before, a_after, problems, notes)
        check_relationships(game, notes, problems)
        check_memory(game, notes, problems)
        check_story(game, notes, problems)

        audit_result = None
        if args.audit:
            audit_result = audit_with_llm(game, a_before, a_after, notes, problems)
        if game.llm.degraded and not args.mock:
            problems.append("LLM 中途降级为 Mock 后端：后半段内容为虚构，结论不可用")
        if args.report:
            _write_report(args.report, game, world, a_before, far_snap, a_after, notes, problems,
                          audit_result)
    finally:
        stats = game.llm.stats
        game.close()

    say("\n--- 观察 ---")
    for n in notes:
        say(" · " + n)
    say("\n--- 问题 ---")
    if problems:
        for p in problems:
            say(" ✗ " + p)
    else:
        say(" 无")
    say(f"\nLLM：{stats}")
    say(f"检查项：{len(notes)} 观察 / {len(problems)} 问题")
    return 1 if problems else 0


def _quiet(game: Game, line: str) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        game.execute(line)
    return buf.getvalue()


def _write_report(path: str, game: Game, world: dict, a_before: Snapshot,
                  far_snap: Snapshot, a_after: Snapshot,
                  notes: List[str], problems: List[str],
                  audit: dict | None = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    store, wid = game.store, game.world_id
    lines: List[str] = []
    w = lines.append
    w(f"# 一致性长跑报告：{world['name']}（{world['era']}）\n")
    w(f"- 后端：{game.llm.backend_name}")
    w(f"- A 区：{A_START}，离开前 tick={a_before.tick}，返回后 tick={a_after.tick}")
    w(f"- LLM 统计：`{game.llm.stats}`\n")
    w("## 观察")
    w("\n".join(f"- {n}" for n in notes))
    w("\n## 问题")
    w("\n".join(f"- ☠ {p}" for p in problems) if problems else "- 无")

    if audit:
        w("\n## LLM 一致性审计")
        w(f"评价：{audit.get('verdict', '')}")
        for c in audit.get("contradictions") or []:
            w(f"- [{c.get('category', '?')}/{c.get('severity', '?')}] {c.get('where', '')}："
              f"{c.get('issue', '')}  依据：{c.get('evidence', '')}")

    w("\n## A 区物体：离开前 -> 返回后")
    w("| id | 名称 | 类别 | 位置 | 描述 | HP |")
    w("| --- | --- | --- | --- | --- | --- |")
    for oid, o in a_before.objects.items():
        b = a_after.objects.get(oid)
        after_cell = (f"{b['name']} / ({b['x']},{b['y']}) / {b['hp']}" if b else "**已删除**")
        w(f"| {oid} | {o['name']} | {o['kind']} | ({o['x']},{o['y']}) | "
          f"{(o['desc'] or '')[:46]} | {o['hp']} -> {after_cell} |")

    w("\n## A 区 NPC")
    w("| id | 名字 | 身份 | 心情 | HP | 位置 |")
    w("| --- | --- | --- | --- | --- | --- |")
    for nid, n in a_after.npcs.items():
        before = a_before.npcs.get(nid)
        w(f"| {nid} | {n['name']} | {n['role']} | {before['mood'] if before else '-'} -> {n['mood']} | "
          f"{n['hp']}/{n['hp_max']} | ({n['x']},{n['y']}) |")

    w("\n## 关系图")
    for r in store.list_relations(wid):
        w(f"- {r['a_kind']}:{r['a_name']} --{r['kind']}({r['value']})--> "
          f"{r['b_kind']}:{r['b_name']}  {(r['note'] or '')[:60]}")

    w("\n## NPC 长期记忆")
    for npc in store.npcs_near(wid, *A_START, 40, limit=30):
        mem = store.npc_memories(npc["id"], limit=8)
        if mem:
            w(f"- **{npc['name']}**（{npc['role']}）")
            for m in mem:
                w(f"  - [第{m['tick']//24+1}天] ({m['kind']}) {m['text']}")

    w("\n## 任务线")
    for s in reversed(store.story_history(wid, limit=20)):
        w(f"- [{s['state']}] {s['title']}：{(s['summary'] or '')[:80]}")

    w("\n## 事件（A 区 / 世界）")
    for e in sorted(store.recent_events(wid, limit=40), key=lambda x: x["tick"]):
        lod = {0: "世界", 1: "区域", 2: "子区域", 3: "地块"}.get(e["lod"], "?")
        w(f"- [第{e['tick']//24+1}天·{lod}] {e['summary'][:110]}")

    w("\n## 远行地 B 的状态")
    for nid, n in far_snap.npcs.items():
        w(f"- NPC {n['name']}（{n['role']}）HP {n['hp']} 位置 ({n['x']},{n['y']})")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"报告已写入 {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="用确定性 Mock（秒级，查结构性缺陷）")
    ap.add_argument("--db", default=os.path.join(ROOT, "data", "qa.db"))
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "qa_cache.db"))
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--hint", default="")
    ap.add_argument("--distance", type=int, default=240, help="A 到 B 的曼哈顿距离")
    ap.add_argument("--hours", type=int, default=400, help="在远处等待的游戏小时")
    ap.add_argument("--settle", type=int, default=1, help="返回 A 后再等待几次（每次 6 小时）")
    ap.add_argument("--radius", type=int, default=24)
    ap.add_argument("--report", default="")
    ap.add_argument("--status", default=os.path.join(ROOT, "logs", "qa_status.txt"))
    ap.add_argument("--audit", action="store_true", help="跑完后让 LLM 审计档案中的矛盾")
    ap.add_argument("--audit-only", action="store_true",
                    help="不跑场景，只对已有 --db 做一次 LLM 审计")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    if args.audit_only:
        args.audit = True
        return audit_only(args)
    return run(args)


def audit_only(args) -> int:
    cfg = load_config(None, {
        "llm": {"mock": bool(args.mock), "cache_path": args.cache},
        "game": {"db_path": args.db},
    })
    game = Game(cfg, verbose=args.verbose)
    problems: List[str] = []
    notes: List[str] = []
    try:
        if not args.mock and not game.llm.ping():
            say("✗ 无法连接本地大模型。")
            return 2
        world = game.load_world()
        say(f"审计世界：{world['name']}（{world['era']}）  后端={game.llm.backend_name}")
        result = audit_with_llm(game, None, None, notes, problems)
        if args.report:
            os.makedirs(os.path.dirname(os.path.abspath(args.report)) or ".", exist_ok=True)
            lines = [f"# LLM 一致性审计：{world['name']}（{world['era']}）\n",
                     f"- 后端：{game.llm.backend_name}",
                     f"- LLM 统计：`{game.llm.stats}`\n",
                     f"## 评价\n{result.get('verdict', '')}\n", "## 发现"]
            for c in result.get("contradictions") or []:
                lines.append(f"- [{c.get('category', '?')}/{c.get('severity', '?')}] "
                             f"{c.get('where', '')}：{c.get('issue', '')}  依据：{c.get('evidence', '')}")
            if not result.get("contradictions"):
                lines.append("- 未发现矛盾")
            with open(args.report, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            say(f"报告已写入 {args.report}")
    finally:
        game.close()
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
