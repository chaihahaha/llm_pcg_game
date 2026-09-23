"""Game orchestration + text REPL.

The player never sees the whole world: they see a 16x16 window, and can
"click" a tile by inspecting it.  Exploration drives time forward, time drives
multi-LOD evolution, and evolution writes back to SQLite.
"""
from __future__ import annotations

import logging
import random
import sys
from typing import Any, Dict, List, Optional

from . import entities, render
from .config import cfg_get
from .db import CacheStore, Store
from .evolution import EvolutionEngine
from .llm import LLMClient
from .narrative import Narrator
from .rng import hash_int
from .world import LOD_CHUNK, WorldManager

_DIRS = ["w", "a", "s", "d", "n", "e", "s", "w"]


def _setup_logger(verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger("pcg")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    if verbose:
        logger.setLevel(logging.INFO)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.WARNING)
    return logger


class Game:
    def __init__(self, cfg: Dict[str, Any], logger=None, verbose: bool = False):
        self.cfg = cfg
        self.logger = logger or _setup_logger(verbose)
        db_path = str(cfg_get(cfg, "game.db_path", "data/world.db"))
        self.store = Store(db_path)
        cache_path = str(cfg_get(cfg, "llm.cache_path", "data/llm_cache.db"))
        self.cache = CacheStore(cache_path)
        self.llm = LLMClient(cfg, store=self.cache, logger=self.logger)
        self.wm = WorldManager(self.store, self.llm, cfg, logger=self.logger)
        self.narrator: Optional[Narrator] = None
        self.evolution: Optional[EvolutionEngine] = None
        self.player: Optional[entities.Player] = None
        self.world_id: Optional[int] = None
        self.view_size = int(cfg_get(cfg, "world.view_size", 16))
        self.auto_story = True

    # -------------------------------------------------------------- lifecycle
    def new_world(self, seed: Optional[int] = None, hint: str = "") -> Dict[str, Any]:
        world = self.wm.create(seed=seed, hint=hint)
        self.world_id = self.wm.world_id
        self.store.set_meta("tick", 0)
        sx = int(cfg_get(self.cfg, "world.start_x", 8))
        sy = int(cfg_get(self.cfg, "world.start_y", 8))
        self.wm.ensure_area(sx - 2, sy - 2, sx + 2, sy + 2)
        sx, sy = self.wm.find_spawn(sx, sy)
        self.store.create_player(
            self.world_id, "旅人", sx, sy,
            int(cfg_get(self.cfg, "game.player_hp", 30)),
            int(cfg_get(self.cfg, "game.player_atk", 5)),
            int(cfg_get(self.cfg, "game.player_def", 2)),
        )
        self._wire()
        half = self.view_size // 2
        self.wm.ensure_area(sx - half, sy - half, sx - half + self.view_size - 1,
                            sy - half + self.view_size - 1)
        self.store.commit()
        if self.auto_story and self.player:
            self.narrator.maybe_new_story(self.player)
        return world

    def load_world(self, world_id: Optional[int] = None) -> Dict[str, Any]:
        if world_id is None:
            row = self.store.conn.execute("SELECT id FROM worlds ORDER BY id DESC LIMIT 1").fetchone()
            if not row:
                raise ValueError("数据库里没有世界，请先 --new")
            world_id = int(row["id"])
        world = self.wm.load(world_id)
        self.world_id = world_id
        self._wire()
        if self.player is None:
            raise ValueError("这个世界还没有玩家数据")
        return world

    def check_backend_matches(self) -> None:
        """Refuse to edit a save authored by a different backend.

        Mock and real models produce structurally identical but semantically
        different content, so letting them write to the same save silently
        replaces real prose with fabricated prose.
        """
        if cfg_get(self.cfg, "game.allow_mixed_backend", False):
            return
        world = self.store.get_world(self.world_id)
        saved = str((world.get("data") or {}).get("backend", "") or "")
        current = self.llm.backend_name
        if saved and saved != current:
            raise ValueError(
                f"存档由 {saved} 后端生成，当前是 {current} 后端。"
                f"混用会把已有内容覆盖为另一种来源的文本。\n"
                f"如确要如此，请加 --allow-mixed（或配置 game.allow_mixed_backend=true）。"
            )

    def _wire(self) -> None:
        self.world_id = self.wm.world_id
        self.narrator = Narrator(self.store, self.llm, self.cfg, self.wm, self.logger)
        self.evolution = EvolutionEngine(self.store, self.llm, self.cfg, self.wm, self.logger)
        self.player = entities.Player(self.store, self.world_id)

    def save(self) -> None:
        self.store.commit()

    def teleport(self, x: int, y: int) -> None:
        """Move the player without generating the intervening map.

        Used by tests/QA to exercise "walk far away, come back" without paying
        for dozens of chunk generations, and handy for debugging.
        """
        assert self.player is not None
        half = self.view_size // 2
        self.wm.ensure_area(x - half, y - half, x - half + self.view_size - 1,
                            y - half + self.view_size - 1)
        self.store.update_player(self.world_id, x=x, y=y, updated_tick=self.tick())
        self.player.refresh()
        self.wm.mark_explored(x, y)
        self.save()

    def close(self) -> None:
        try:
            self.store.commit()
            self.store.close()
        finally:
            self.cache.close()

    # ------------------------------------------------------------------- tick
    def tick(self) -> int:
        return int(self.store.get_meta("tick", 0) or 0)

    def weather(self) -> str:
        if not self.player:
            return ""
        chunk = self.wm.ensure_node(LOD_CHUNK, self.player.x, self.player.y)
        return str((chunk.get("data") or {}).get("weather", ""))

    # ---------------------------------------------------------------- display
    def draw(self) -> str:
        assert self.player is not None
        quest = self.narrator.current_quest()
        return render.render_status(self.player, self.tick(), self.weather(), quest, self.llm.stats)

    def look(self) -> str:
        assert self.player is not None
        return render.render_view(self.wm, self.player.x, self.player.y, self.view_size, self.player)

    # -------------------------------------------------------------- commands
    def execute(self, line: str) -> bool:
        line = (line or "").strip()
        if not line:
            return True
        parts = line.split()
        cmd = parts[0].lower()
        rest = parts[1:]

        if cmd in ("quit", "exit", "q"):
            return False
        if cmd in ("help", "h", "?"):
            print(render.render_help())
            return True
        if cmd in ("look", "l"):
            print(self.draw())
            print(self.look())
            return True
        if cmd in ("move", "go", "m"):
            if not rest:
                print("用法：move <方向>，例：move d")
                return True
            self._do_move(rest[0])
            return True
        if cmd in ("goto",):
            if len(rest) < 2:
                print("用法：goto <x> <y>")
                return True
            try:
                tx, ty = int(rest[0]), int(rest[1])
            except ValueError:
                print("坐标必须是整数。")
                return True
            self._do_goto(tx, ty)
            return True
        if cmd in ("inspect", "i", "x"):
            self._do_inspect(rest)
            return True
        if cmd in ("talk", "t"):
            self._do_talk(" ".join(rest))
            return True
        if cmd in ("attack", "a"):
            self._do_attack(" ".join(rest))
            return True
        if cmd in ("wait", "w8"):
            hours = 1
            if rest:
                try:
                    hours = max(1, min(720, int(rest[0])))
                except ValueError:
                    pass
            self._report_advance(hours, f"你原地等待了 {hours} 小时。", mode="world")
            print(self.draw())
            return True
        if cmd == "story":
            self._show_story()
            return True
        if cmd == "newstory":
            self.narrator.maybe_new_story(self.player, force=True)
            self._show_story()
            return True
        if cmd == "journal":
            n = 12
            if rest:
                try:
                    n = max(1, min(60, int(rest[0])))
                except ValueError:
                    pass
            self._show_journal(n)
            return True
        if cmd == "world":
            self._show_world()
            return True
        if cmd == "stats":
            self._show_stats()
            return True
        if cmd == "auto":
            n = 10
            if rest:
                try:
                    n = max(1, min(200, int(rest[0])))
                except ValueError:
                    pass
            self._auto(n)
            return True
        print(f"未知指令：{cmd}（输入 help 查看指令）")
        return True

    # -- individual actions
    def _do_move(self, direction: str) -> None:
        assert self.player is not None
        ok, reason = self.player.move(direction)
        if not ok:
            print(reason)
            self._resolve_npc_turn()
            return
        self.wm.mark_explored(self.player.x, self.player.y)
        self._report_advance(1)
        print(self.draw())
        print(self.look())

    def _do_goto(self, tx: int, ty: int) -> None:
        assert self.player is not None
        dx = (tx > self.player.x) - (tx < self.player.x)
        dy = (ty > self.player.y) - (ty < self.player.y)
        if dx == 0 and dy == 0:
            print("你已经在那里了。")
            return
        ok, reason = self.player.move_to(self.player.x + dx, self.player.y + dy)
        if not ok:
            print(reason)
            return
        self.wm.mark_explored(self.player.x, self.player.y)
        self._report_advance(1)
        print(self.draw())
        print(self.look())

    def _do_inspect(self, args: List[str]) -> None:
        assert self.player is not None
        if len(args) >= 2:
            try:
                x, y = int(args[0]), int(args[1])
            except ValueError:
                print("坐标必须是整数。")
                return
        else:
            x, y = self.player.x, self.player.y
        print(render.render_tile_info(self.wm, x, y))

    def _do_talk(self, rest: str) -> None:
        assert self.player is not None
        name, line = "", rest
        tokens = rest.split(maxsplit=1)
        if tokens:
            near = self.store.npcs_at(self.world_id, self.player.x, self.player.y)
            if not near:
                near = self.store.npcs_near(self.world_id, self.player.x, self.player.y, 2)
            if any(n["name"] == tokens[0] for n in near):
                name = tokens[0]
                line = tokens[1] if len(tokens) > 1 else ""
        npc = self._npc_target(name, verb="交谈")
        if not npc:
            return
        if not line:
            line = "你好。"
        result = self.narrator.talk(self.player, npc, line)
        print(f"{npc['name']}：{result['reply']}")
        if result.get("mood"):
            print(f"（{npc['name']}现在 {result['mood']}）")
        if result.get("quest"):
            print(f"※ 新任务「{result['quest']['title']}」："
                  f"{result['quest'].get('data', {}).get('objective', '')}")

    def _do_attack(self, name: str) -> None:
        assert self.player is not None
        npc = self._npc_target(name, verb="攻击")
        if not npc:
            return
        result = entities.attack(self.store, self.world_id, self.player, npc, self.tick())
        for line in result["log"]:
            print(line)
        if result.get("ok"):
            place = self.narrator.place_name(npc["x"], npc["y"])
            flavor = self.narrator.combat_flavor(self.player.data["name"], npc["name"],
                                                 result["result"], place)
            if flavor:
                print(f"…{flavor}")
        if not self.player.alive:
            print("你倒下了。世界仍在运转……")
        print(self.draw())

    def _npc_target(self, name: str, verb: str = "交谈"):
        """Resolve a talk/attack target, explaining *why* it is not usable.

        "附近没有人" is unhelpful when the person you named is standing four
        tiles away inside a thicket: the player needs to be told the distance
        and the fact that they are not adjacent.
        """
        assert self.player is not None
        npc = entities.find_npc(self.store, self.world_id, name, self.player.x, self.player.y)
        if npc:
            return npc
        if name:
            far = self.store.find_npc_by_name(self.world_id, name)
            if far:
                d = abs(far["x"] - self.player.x) + abs(far["y"] - self.player.y)
                print(f"{name} 在 ({far['x']},{far['y']})，离你 {d} 格，无法{verb}（需要相邻）。")
                return None
            print(f"这里没有叫「{name}」的人。")
            return None
        near = self.store.npcs_near(self.world_id, self.player.x, self.player.y, 6, limit=5)
        if near:
            listing = "，".join(f"{n['name']}[{n['x']},{n['y']}]" for n in near)
            print(f"附近没有相邻的人，无法{verb}。可见：{listing}")
        else:
            print(f"附近没有可{verb}的目标。")
        return None

    def _resolve_npc_turn(self) -> None:
        assert self.player is not None
        for npc in self.store.npcs_at(self.world_id, self.player.x, self.player.y):
            for line in entities.npc_turn(self.store, self.world_id, npc,
                                          self.player.x, self.player.y, self.tick()):
                print(line)
        self.player.refresh()

    def _progress(self, lod: int, node: dict, elapsed: int) -> None:
        """Say what the model is working on, so a slow local model is legible
        instead of looking like a freeze."""
        name = {0: "世界", 1: "区域", 2: "子区域", 3: "地块"}.get(lod, "?")
        extra = f"，补算 {elapsed} 小时" if elapsed > 48 else ""
        print(f"  ⟳ 推演{name}「{node.get('name', '')}」{extra} …", flush=True)

    def _advance(self, hours: int, note: str = "", mode: str = "local") -> str:
        """Move time forward; returns what is worth telling the player.

        Ordinary steps use ``mode="local"`` (only the tile you stand in and its
        ancestors), so walking never triggers a minutes-long world-wide sweep.
        ``wait`` uses ``mode="world"`` to catch the rest of the world up.
        """
        assert self.player is not None
        events = self.evolution.advance(hours, self.player.x, self.player.y, mode=mode,
                                        on_scope=self._progress)
        self._resolve_npc_turn()
        self.save()
        lines: List[str] = []
        if note:
            lines.append(note)
        for ev in events[:6]:
            lines.append(f"※ {ev['summary']}")
        if self.auto_story and self.narrator.current_quest() is None:
            q = self.narrator.maybe_new_story(self.player)
            if q:
                lines.append(f"※ 新任务「{q['title']}」：{q.get('data', {}).get('objective', '')}")
        return "\n".join(lines)

    def _report_advance(self, hours: int, note: str = "", mode: str = "local") -> None:
        msg = self._advance(hours, note, mode=mode)
        if msg:
            print(msg)

    def _auto(self, steps: int) -> None:
        assert self.player is not None
        rng = random.Random(hash_int("auto", self.world_id, self.player.x, self.player.y,
                                     mod=2 ** 31))
        moved = 0
        for _ in range(steps):
            if not self.player.alive:
                break
            dirs = ["w", "d", "s", "a"]
            rng.shuffle(dirs)
            for d in dirs:
                if self.player.move(d)[0]:
                    moved += 1
                    break
            self.wm.mark_explored(self.player.x, self.player.y)
            self.evolution.advance(1, self.player.x, self.player.y, mode="local",
                                   on_scope=self._progress)
            self._resolve_npc_turn()
        self.save()
        print(f"自动探索 {steps} 步（实际移动 {moved} 格），当前座标 ({self.player.x},{self.player.y})")
        print(self.draw())
        print(self.look())

    # -- info
    def _show_story(self) -> None:
        q = self.narrator.current_quest()
        if not q:
            print("当前没有任务。可以 newstory 让说书人生成一条。")
            return
        d = q.get("data") or {}
        print(f"【{q['title']}】")
        print(f"背景：{q['summary']}")
        if d.get("objective"):
            print(f"目标：{d['objective']}")
        if d.get("stakes"):
            print(f"后果：{d['stakes']}")
        if d.get("hint"):
            print(f"线索：{d['hint']}")

    def _show_journal(self, n: int) -> None:
        assert self.player is not None
        rows = self.store.recent_events(self.world_id, limit=n)
        if not rows:
            print("还没有任何记录。")
            return
        for ev in reversed(rows):
            day, hour = ev["tick"] // 24 + 1, ev["tick"] % 24
            lod = {0: "世界", 1: "区域", 2: "子区域", 3: "地块"}.get(ev["lod"], "?")
            print(f"[第{day}天{hour:02d}时·{lod}] {ev['summary']}")

    def _show_world(self) -> None:
        world = self.store.get_world(self.world_id)
        print(f"世界：{world['name']}（{world['era']}）")
        print(f"魔法：{world['data'].get('magic_system','')}")
        print(f"科技：{world['data'].get('tech_baseline','')}")
        print(f"概要：{world['summary']}")
        print("国家：")
        for n in self.store.list_nations(self.world_id):
            print(f"  · {n['name']}｜{n['race']}｜{n['gov']}｜科技 {n['tech']}｜魔法 {n['magic']}"
                  f"｜矿产 {'、'.join(n['resources'])}")
            if n["summary"]:
                print(f"    {n['summary']}")

    def _show_stats(self) -> None:
        stats = self.store.stats(self.world_id)
        print(f"世界数据库：节点{stats['nodes']} 格子{stats['tiles']} 物体{stats['objects']} "
              f"NPC{stats['npcs']} 事件{stats['events']} 国家{stats['nations']}")
        if self.llm.degraded:
            print("⚠ LLM 已降级为 Mock 后端：此后内容为虚构模拟，不是模型输出！")
        print(f"LLM：后端 {self.llm.backend_name}｜实调 {self.llm.stats['calls']}｜"
              f"缓存命中 {self.llm.stats['cache_hits']}｜估算输入 {self.llm.stats['prompt_tokens']} tok｜"
              f"输出 {self.llm.stats['completion_tokens']} tok")
        print(f"LLM 缓存条数：{self.cache.cache_count()}（{self.cache.path}）")

    # ------------------------------------------------------------------- loop
    def run(self, script: Optional[List[str]] = None, echo: bool = True) -> None:
        if script is not None:
            for line in script:
                if echo:
                    print(f"> {line}")
                if not self.execute(line):
                    break
            return
        print(render.render_help())
        print(self.draw())
        print(self.look())
        while True:
            try:
                line = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            try:
                if not self.execute(line):
                    break
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive
                self.logger.exception("command failed: %s", exc)
                print(f"[错误] {exc}")
        self.save()
