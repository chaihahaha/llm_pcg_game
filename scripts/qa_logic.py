#!/usr/bin/env python3
"""Logic-hole probe: hunt for narrative/state contradictions, not crashes.

The trigger for this script was a real report: the player stabbed 巴洛, then
`talk`ed to him, and he answered as if nothing had happened.  Each check below
asserts that some *causal link* actually exists between two subsystems —
combat↔dialogue, substrate↔dialogue, death↔interaction, hostility↔LLM cost.

    python3 scripts/qa_logic.py            # mock backend, instant
"""
from __future__ import annotations

import contextlib
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from pcg import entities  # noqa: E402
from pcg.config import load_config  # noqa: E402
from pcg.game import Game  # noqa: E402
from pcg.sim import sim_state  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"{'✓' if ok else '✗'} {name}" + (f"   — {detail}" if detail else ""))


def fresh(name: str) -> str:
    p = os.path.join(ROOT, "data", name)
    for suf in ("", "-wal", "-shm"):
        os.remove(p + suf) if os.path.exists(p + suf) else None
    return p


def main() -> int:
    cfg = load_config(None, {
        "llm": {"mock": True, "cache_path": fresh("qa_logic_cache.db")},
        "game": {"db_path": fresh("qa_logic.db")},
    })
    game = Game(cfg)
    game.auto_story = False
    game.new_world(seed=4242)

    def q(line: str) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            game.execute(line)
        return buf.getvalue()

    def place_npc_next_to_player() -> dict:
        npc = game.store.npcs_near(game.world_id, game.player.x, game.player.y, 30, limit=1)[0]
        game.store.update_npc(npc["id"], x=game.player.x, y=game.player.y + 1)
        game.store.commit()
        return game.store.get_npc(npc["id"])

    # ---------------------------------------------------------------- combat → talk
    npc = place_npc_next_to_player()
    out = q(f"attack {npc['name']}")
    check("攻击会产生伤害提示", "造成" in out, out.strip().splitlines()[0] if out.strip() else "")

    harmed = game.store.get_npc(npc["id"])
    sim = sim_state(harmed)
    check("被攻击后愤怒上升", sim["anger"] > sim_state(npc)["anger"], f"anger={sim['anger']:.0f}")
    check("被攻击后转为敌对", bool(harmed["hostile"]))
    check("被攻击后写入怨恨记忆",
          any(m["kind"] == "grudge" for m in game.store.npc_memories(npc["id"], limit=5)))
    rel = [r for r in game.store.relations_for(game.world_id, "npc", npc["name"])
           if r["other_kind"] == "player"]
    check("被攻击后生成 npc→player 的死敌边", bool(rel) and rel[0]["value"] < 0,
          rel[0]["kind"] if rel else "无")

    calls_before_talk = game.llm.stats["calls"]
    out = q(f"talk {npc['name']} 我们聊聊")
    check("攻击后无法正常对话", "没有回答" in out or "无法回应" in out,
          out.strip().splitlines()[0] if out.strip() else "")
    check("拒绝对话时不浪费 LLM 调用",
          game.llm.stats["calls"] == calls_before_talk,
          f"{game.llm.stats['calls'] - calls_before_talk} 次")
    check("拒绝对话时态度可见", "怒气" in out or "HP" in out)

    # ---------------------------------------------------------------- death gates
    dead = place_npc_next_to_player()
    for _ in range(30):
        game.execute("attack")
        if not game.store.get_npc(dead["id"])["alive"]:
            break
    state = game.store.get_npc(dead["id"])
    check("可以被打死", not state["alive"], f"alive={state['alive']}")

    out = q(f"talk {dead['name']}")
    check("死后无法交谈", "已经死了" in out, out.strip().splitlines()[0] if out.strip() else "")
    out = q(f"attack {dead['name']}")
    check("死后无法再被攻击", "倒下" in out or "没有" in out,
          out.strip().splitlines()[0] if out.strip() else "")

    # ---------------------------------------------------------------- 安抚路径
    victim = place_npc_next_to_player()
    q(f"attack {victim['name']}")
    game.store.update_npc(victim["id"], hostile=0, hp=game.store.get_npc(victim["id"])["hp_max"])
    from pcg.sim import sim_state as _ss
    v = game.store.get_npc(victim["id"])
    s = _ss(v)
    s["anger"] = 0.0
    st = dict(v.get("state") or {})
    st["sim"] = s
    game.store.update_npc(victim["id"], state=st)
    game.store.commit()
    ok, why = entities.can_talk(game.store, game.world_id, game.store.get_npc(victim["id"]))
    check("安抚（移除敌对+降温+治愈）后可以对话", ok, why)

    # ---------------------------------------------------------------- 昏迷/困境
    sleeper = place_npc_next_to_player()
    game.store.update_npc(sleeper["id"], status="重伤昏迷")
    game.store.commit()
    out = q(f"talk {sleeper['name']}")
    check("昏迷者无法对话", "无法回应" in out, out.strip().splitlines()[0] if out.strip() else "")

    # ---------------------------------------------------------------- 距离
    distant = [n for n in game.store.npcs_near(game.world_id, game.player.x, game.player.y,
                                               60, limit=20)
               if n["id"] not in {sleeper["id"]}]
    far = next((n for n in distant
                if abs(n["x"] - game.player.x) + abs(n["y"] - game.player.y) > 4), None)
    if far:
        out = q(f"talk {far['name']} 喂")
        check("远处的目标给出距离而不是笼统失败", "格" in out or "不在" in out,
              out.strip().splitlines()[0] if out.strip() else "")
    else:
        check("远处的目标给出距离而不是笼统失败", True, "（本图没有远处 NPC，跳过）")

    # ---------------------------------------------------------------- 敌对会还手
    foe = place_npc_next_to_player()
    game.store.update_npc(foe["id"], hostile=1, atk=5)
    game.store.commit()
    hp_before = game.player.hp
    out = q("move s")          # into the NPC: the move is refused, the turn still runs
    check("敌对 NPC 会在玩家行动时攻击", game.player.hp < hp_before or "扑了上来" in out,
          f"HP {hp_before} -> {game.player.hp}")

    # ---------------------------------------------------------------- 敌意会冷却
    calm = place_npc_next_to_player()
    game.store.update_npc(calm["id"], hostile=1, hp=game.store.get_npc(calm["id"])["hp_max"])
    v = game.store.get_npc(calm["id"])
    s = _ss(v)
    s["anger"] = 0.0
    s["grudge_tick"] = 0
    st = dict(v.get("state") or {})
    st["sim"] = s
    game.store.update_npc(calm["id"], state=st)
    game.store.commit()
    game.evolution.sim.step(24 * 30, 24 * 30, game.player.x, game.player.y)
    check("长期冷静后敌意会解除",
          not game.store.get_npc(calm["id"])["hostile"],
          f"hostile={game.store.get_npc(calm['id'])['hostile']}")

    # ---------------------------------------------------------------- 敌对时情绪自洽
    mood = game.store.get_npc(calm["id"])["mood"]
    check("解除敌意后情绪来自需求公式而不是残留文本",
          mood in {"平静", "愉悦", "疲惫", "焦躁", "消沉", "愤怒", "虚弱", "警觉", ""},
          f"mood={mood!r}")

    # ---------------------------------------------------------------- 结算
    print("\n--- 汇总 ---")
    bad = [r for r in RESULTS if not r[1]]
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} 通过")
    game.close()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
