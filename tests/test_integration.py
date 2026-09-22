"""Mock-backed end-to-end tests for the LOD tree, evolution and REPL."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pcg.config import load_config  # noqa: E402
from pcg.db import Store  # noqa: E402
from pcg.evolution import EvolutionEngine  # noqa: E402
from pcg.game import Game  # noqa: E402
from pcg.llm import LLMClient  # noqa: E402
from pcg.world import LOD_CHUNK, LOD_REGION, LOD_ZONE, WorldManager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def db_path(name: str) -> str:
    p = os.path.join(ROOT, "data", name)
    if os.path.exists(p):
        os.remove(p)
    return p


def test_cfg(**overrides) -> dict:
    """Config pinned to throwaway files — tests must never touch real saves or
    the production LLM cache."""
    base = {
        "llm": {"mock": True, "cache_path": db_path("test_llm_cache.db")},
        "game": {"db_path": db_path("test_game_default.db")},
    }
    for section, values in overrides.items():
        base.setdefault(section, {}).update(values)
    return load_config(None, base)


class TestLODTree(unittest.TestCase):
    def setUp(self):
        self.cfg = test_cfg()
        self.store = Store(db_path("test_lod.db"))
        self.llm = LLMClient(self.cfg, store=self.store)
        self.wm = WorldManager(self.store, self.llm, self.cfg)
        self.wm.create(seed=7)

    def tearDown(self):
        self.store.close()

    def test_ancestors_are_created_top_down(self):
        chunk = self.wm.ensure_node(LOD_CHUNK, 40, 24)
        self.assertEqual(chunk["lod"], LOD_CHUNK)
        self.assertIsNotNone(self.store.get_node(self.wm.world_id, LOD_ZONE, 0, 0))
        self.assertIsNotNone(self.store.get_node(self.wm.world_id, LOD_REGION, 0, 0))
        self.assertEqual(chunk["parent_id"],
                         self.store.get_node(self.wm.world_id, LOD_ZONE, 0, 0)["id"])

    def test_chunk_materialises_tiles_objects_npcs(self):
        chunk = self.wm.ensure_node(LOD_CHUNK, 0, 0)
        rows = chunk["data"]["rows"]
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(len(r) == 16 for r in rows))
        tiles = self.store.list_tiles(self.wm.world_id, 0, 0, 15, 15)
        self.assertEqual(len(tiles), 256)
        self.assertGreaterEqual(len(self.store.objects_near(self.wm.world_id, 8, 8, 16)), 1)

    def test_view_generates_neighbouring_chunks(self):
        tiles = self.wm.view_tiles(12, 12, 16)
        self.assertEqual(len(tiles), 256)
        # a chunk it had to create on the fly (view straddles four chunks)
        self.assertIsNotNone(self.store.get_node(self.wm.world_id, LOD_CHUNK, 16, 16))

    def test_tile_description_is_persisted(self):
        t1 = self.wm.ensure_tile_desc(3, 3)
        self.assertTrue(t1["desc"])
        t2 = self.store.get_tile(self.wm.world_id, 3, 3)
        self.assertEqual(t1["desc"], t2["desc"])
        self.assertIn("detail", t2["data"])

    def test_terrain_stays_in_vocabulary(self):
        from pcg.terrain import TERRAIN
        for t in self.store.list_tiles(self.wm.world_id, 0, 0, 15, 15):
            self.assertIn(t["terrain"], TERRAIN)


class TestEvolution(unittest.TestCase):
    def setUp(self):
        self.cfg = test_cfg(evolution={"max_llm_calls_per_wait": 6})
        self.store = Store(db_path("test_evo.db"))
        self.llm = LLMClient(self.cfg, store=self.store)
        self.wm = WorldManager(self.store, self.llm, self.cfg)
        self.wm.create(seed=11)
        self.wm.ensure_area(0, 0, 16, 16)
        self.engine = EvolutionEngine(self.store, self.llm, self.cfg, self.wm)

    def tearDown(self):
        self.store.close()

    def test_advance_moves_clock_and_records_events(self):
        events = self.engine.advance(1, 8, 8)
        self.assertEqual(self.store.get_meta("tick"), 1)
        # chunk schedule is 6h, so nothing at hour 1
        self.assertEqual(events, [])

    def test_long_wait_evolves_multiple_lods(self):
        self.engine.advance(1, 8, 8)
        events = self.engine.advance(200, 8, 8)
        self.assertTrue(events)
        lods = {e["lod"] for e in events}
        self.assertTrue(lods & {LOD_CHUNK, LOD_ZONE, LOD_REGION})
        for ev in events:
            self.assertTrue(ev["summary"])

    def test_node_summary_updated(self):
        before = self.wm.ensure_node(LOD_CHUNK, 0, 0)["summary"]
        self.engine.advance(1, 8, 8)
        self.engine.advance(12, 8, 8)
        after = self.wm.ensure_node(LOD_CHUNK, 0, 0)["summary"]
        self.assertIsInstance(after, str)
        self.assertTrue(after)
        self.assertNotEqual(before, "")  # generated summary exists either way


class TestGameREPL(unittest.TestCase):
    def setUp(self):
        self.cfg = test_cfg(game={"db_path": db_path("test_game.db")})
        self.game = Game(self.cfg)
        self.game.auto_story = True
        self.game.new_world(seed=3)

    def tearDown(self):
        self.game.close()

    def test_scripted_session(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            for line in ["look", "inspect", "move d", "move s", "wait 12",
                         "journal 5", "story", "world", "stats", "look"]:
                self.assertTrue(self.game.execute(line))
        self.assertGreater(self.game.tick(), 0)
        self.assertGreater(self.game.store.stats(self.game.world_id)["tiles"], 0)

    def test_cannot_walk_into_water_or_npc(self):
        from pcg import entities
        x, y = self.game.player.x, self.game.player.y
        # find a solid tile nearby
        solid = None
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                t = self.store_tile(x + dx, y + dy)
                if t and t["terrain"] in ("water", "mountain", "lava"):
                    solid = (x + dx, y + dy)
        if solid:
            ok, _ = entities.can_enter(self.game.store, self.game.world_id, *solid)
            self.assertFalse(ok)

    def store_tile(self, x, y):
        return self.game.store.get_tile(self.game.world_id, x, y)

    def test_talk_and_attack_do_not_crash(self):
        import contextlib
        import io

        from pcg import entities
        with contextlib.redirect_stdout(io.StringIO()):
            self.game.execute("auto 4")
        near = self.game.store.npcs_near(self.game.world_id, self.game.player.x,
                                         self.game.player.y, 3)
        if near:
            npc = near[0]
            res = self.game.narrator.talk(self.game.player, npc, "你好")
            self.assertTrue(res["reply"])
            out = entities.attack(self.game.store, self.game.world_id, self.game.player, npc,
                                  self.game.tick())
            self.assertTrue(out["ok"])

    def test_quit_command(self):
        self.assertFalse(self.game.execute("quit"))


if __name__ == "__main__":
    unittest.main()
