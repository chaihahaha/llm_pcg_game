"""The 16x16 view must never show a symbol it cannot explain.

Reported by a player: the screen was full of `s` and the legend did not mention
it.  The legend had been a hardcoded subset of ten terrains.  These tests pin
down the real invariant — *every* glyph in the view is accounted for — and that
the model's own local names for terrains reach the legend and `inspect`.
"""
from __future__ import annotations

import contextlib
import io
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pcg import render  # noqa: E402
from pcg.config import load_config  # noqa: E402
from pcg.game import Game  # noqa: E402
from pcg.terrain import TERRAIN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# glyphs that are self-explanatory or listed on the marker line
ALWAYS_KNOWN = set("@†")


def fresh(name: str) -> str:
    p = os.path.join(ROOT, "data", name)
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(p + suf):
            os.remove(p + suf)
    return p


class TestLegend(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(None, {
            "llm": {"mock": True, "cache_path": fresh("test_render_cache.db")},
            "game": {"db_path": fresh("test_render.db")},
        })
        self.game = Game(self.cfg)
        self.game.auto_story = False
        self.game.new_world(seed=11)

    def tearDown(self):
        self.game.close()

    def view_rows_and_legend(self):
        g = self.game
        out = render.render_view(g.wm, g.player.x, g.player.y, 16, g.player)
        rows = [ln for ln in out.splitlines() if re.match(r"^\s*-?\d+\S", ln)]
        legend = "\n".join(ln for ln in out.splitlines() if ln.startswith(("符号：", "      ")))
        return rows, legend

    def test_every_glyph_in_the_view_is_explained(self):
        """Walk the neighbourhood: every glyph is either in the legend or is a
        person's name (which the legend covers as 汉字=人物, and 附近人物 lists)."""
        g = self.game
        unexplained = set()
        for _ in range(40):
            for d in ("w", "a", "s", "d"):
                g.execute(f"move {d}")
                rows, legend = self.view_rows_and_legend()
                people = {n["name"][0] for n in
                          g.store.npcs_near(g.world_id, g.player.x, g.player.y, 9,
                                            limit=50, alive_only=False)}
                for line in rows:
                    body = re.split(r"\s*<-", re.sub(r"^\s*-?\d+", "", line))[0]
                    for ch in body:
                        if ch in ALWAYS_KNOWN or ch in people:
                            continue
                        if ch not in legend:
                            unexplained.add(ch)
        self.assertFalse(unexplained,
                         f"这些符号出现在画面上但图例没有解释：{sorted(unexplained)}")

    def test_legend_covers_all_terrain_that_can_be_shown(self):
        g = self.game
        # force every terrain into the world and make sure none is left out
        x0, y0 = g.player.x - 8, g.player.y - 8
        for i, terrain in enumerate(TERRAIN):
            g.store.upsert_tile(g.world_id, x0 + (i % 16), y0 + (i // 16), terrain,
                                biome=terrain)
        g.store.commit()
        tiles = g.wm.view_tiles(g.player.x, g.player.y, 16)
        legend = render.legend_line(g.wm, tiles, [])
        missing = [t for t in {v["terrain"] for v in tiles.values()}
                   if TERRAIN[t][0] not in legend]
        self.assertFalse(missing, f"图例缺少地形：{missing}")

    def test_model_local_names_reach_the_legend_and_inspect(self):
        g = self.game
        g.wm.merge_glossary({"s": "锈砂脊", "m": "酸泥滩"})
        g.wm.ensure_node(3, g.player.x, g.player.y)
        g.store.upsert_tile(g.world_id, g.player.x, g.player.y, "sand", biome="sand")
        g.store.commit()
        tiles = g.wm.view_tiles(g.player.x, g.player.y, 16)
        legend = render.legend_line(g.wm, tiles, [])
        self.assertIn("锈砂脊", legend)
        info = render.render_tile_info(g.wm, g.player.x, g.player.y, describe=False)
        self.assertIn("锈砂脊", info)

    def test_symbols_are_never_explained_contradictorily(self):
        """`+` was printed both as ruins terrain and as a 'ruin' object kind."""
        g = self.game
        g.store.add_object(g.world_id, g.player.x + 1, g.player.y, "ruin", "断墙",
                           "半截石墙。")
        g.store.commit()
        tiles = g.wm.view_tiles(g.player.x, g.player.y, 16)
        objs = g.store.objects_near(g.world_id, g.player.x, g.player.y, 9, alive_only=False)
        legend = render.legend_line(g.wm, tiles, objs)
        for sym in set(re.findall(r"(?m)^\s*(\S)=", legend)) | set(
                re.findall(r"([^\s=])=[^\s]+", legend)):
            entries = [e for e in re.findall(rf"(?:^|\s){re.escape(sym)}=([^\s]+)", legend)]
            labels = {e for e in entries}
            self.assertLessEqual(len(labels), 1,
                                 f"符号 {sym!r} 有互相矛盾的解释：{labels}")

    def test_legend_command_lists_the_full_table(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.game.execute("legend")
        out = buf.getvalue()
        for terrain, (sym, _solid, label) in TERRAIN.items():
            self.assertIn(sym, out, f"legend 命令漏掉了 {terrain}")
            self.assertIn(label, out)


if __name__ == "__main__":
    unittest.main()
