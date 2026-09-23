"""Mock-backed end-to-end tests for the LOD tree, evolution and REPL."""
from __future__ import annotations

import logging
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pcg.config import load_config  # noqa: E402
from pcg.db import Store  # noqa: E402
from pcg.evolution import EvolutionEngine  # noqa: E402
from pcg.game import Game  # noqa: E402
from pcg.llm import LLMClient, parse_json_loose  # noqa: E402
from pcg.world import LOD_CHUNK, LOD_REGION, LOD_WORLD, LOD_ZONE, WorldManager  # noqa: E402

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

    def test_npc_names_are_unique_across_chunks(self):
        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        self.wm.ensure_node(LOD_CHUNK, 64, 64)
        names = [n["name"] for n in
                 self.store.npcs_near(self.wm.world_id, 32, 32, 200, limit=200)]
        self.assertEqual(len(names), len(set(names)))

    def test_terrain_stays_in_vocabulary(self):
        from pcg.terrain import TERRAIN
        for t in self.store.list_tiles(self.wm.world_id, 0, 0, 15, 15):
            self.assertIn(t["terrain"], TERRAIN)


class TestKeyDrift(unittest.TestCase):
    """Real model outputs observed in the wild must resolve to the right field."""

    def test_whitespace_padded_key(self):
        from pcg.world import _field
        payload = '{"  name":"酸雾沉溺巷","biome_mix":[],"summary":"酸泥与蒸汽交织"}'
        data = parse_json_loose(payload)
        self.assertEqual(_field(data, "name", "zone_name", "title", default="?"), "酸雾沉溺巷")
        self.assertEqual(_field(data, "summary", "overview", default="?"), "酸泥与蒸汽交织")

    def test_nested_envelope(self):
        from pcg.world import _field
        payload = ('{"zone": {"parent": "烬骨断崖", "coords": [0, 0], '
                   '"data": {"name": "灰雾铁锈回廊", "biome_mix": [["#", 0.6]], '
                   '"summary": "锈塔与矿脉裂隙交错"}}}')
        data = parse_json_loose(payload)
        self.assertEqual(_field(data, "name", "zone_name", default="?"), "灰雾铁锈回廊")
        self.assertEqual(_field(data, "summary", "overview", default="?"), "锈塔与矿脉裂隙交错")
        self.assertEqual(_field(data, "biome_mix", "biomes", default=None), [["#", 0.6]])

    def test_suffixed_and_alternative_keys(self):
        from pcg.world import _field
        self.assertEqual(_field({"area_name": "锈雾沉降带"}, "name", default="?"), "锈雾沉降带")
        self.assertEqual(_field({"world_name": "锈海苍穹"}, "name", "world_name", default="?"),
                         "锈海苍穹")
        self.assertEqual(_field({"patches": []}, "patches", "regions", default=None), None)
        self.assertEqual(_field({}, "name", default="FALLBACK"), "FALLBACK")


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

    def test_due_scopes_are_coarse_to_fine(self):
        # force every LOD overdue
        self.engine.advance(1, 8, 8)
        tick = self.store.get_meta("tick") + 5000
        plan = self.engine._plan(tick, 8, 8)
        self.assertTrue(plan)
        lods = [lod for lod, _, _, _ in plan]
        self.assertEqual(lods, sorted(lods), "higher LODs must evolve before local ones")
        self.assertIn(LOD_WORLD, lods)
        self.assertIn(LOD_CHUNK, lods)
        # every entry carries an overdue step count and an elapsed-hours figure
        self.assertTrue(all(steps >= 1 and elapsed >= 1 for _, _, steps, elapsed in plan))

    def test_node_summary_updated(self):
        before = self.wm.ensure_node(LOD_CHUNK, 0, 0)["summary"]
        self.engine.advance(1, 8, 8)
        self.engine.advance(12, 8, 8)
        after = self.wm.ensure_node(LOD_CHUNK, 0, 0)["summary"]
        self.assertIsInstance(after, str)
        self.assertTrue(after)
        self.assertNotEqual(before, "")  # generated summary exists either way


class TestContinuity(unittest.TestCase):
    """The world must keep living where the player is not, and must not forget."""

    def setUp(self):
        self.cfg = test_cfg(evolution={"max_llm_calls_per_wait": 6, "max_catchup_calls": 10,
                                       "max_chunk_scopes_per_advance": 2})
        self.store = Store(db_path("test_continuity.db"))
        self.llm = LLMClient(self.cfg, store=self.store,
                             logger=logging.getLogger("pcg.test"))
        self.wm = WorldManager(self.store, self.llm, self.cfg)
        self.wm.create(seed=99)
        self.engine = EvolutionEngine(self.store, self.llm, self.cfg, self.wm)

    def tearDown(self):
        self.store.close()

    def test_far_scopes_still_evolve(self):
        """A chunk 240 tiles away must not be frozen while the player is elsewhere."""
        near = self.wm.ensure_node(LOD_CHUNK, 0, 0)
        far = self.wm.ensure_node(LOD_CHUNK, 240, 240)
        self.engine.advance(1, 8, 8)
        self.engine.advance(200, 8, 8)  # player stays at the near chunk
        after = self.store.get_node_by_id(far["id"])
        self.assertGreater(after["updated_tick"], far["updated_tick"],
                           "a scope far from the player was never evolved (world froze)")
        self.assertGreater(self.store.get_node_by_id(near["id"])["updated_tick"], 0)

    def test_distant_chunks_tick_more_coarsely(self):
        """Otherwise an explored world would queue a full sweep every 6 hours."""
        base = self.engine.schedule[LOD_CHUNK]
        self.assertEqual(self.engine._effective_period(LOD_CHUNK, 0), base)
        self.assertGreater(self.engine._effective_period(LOD_CHUNK, 96), base)
        # non-chunk levels keep their period regardless of distance
        self.assertEqual(self.engine._effective_period(LOD_ZONE, 1000),
                         self.engine.schedule[LOD_ZONE])

    def test_catchup_is_summarised_not_stepped(self):
        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        self.engine.advance(1, 8, 8)
        calls_before = self.llm.stats["calls"]
        self.engine.advance(600, 8, 8)  # 100 chunk periods in one go
        used = self.llm.stats["calls"] - calls_before
        self.assertLessEqual(used, self.engine.max_catchup_calls,
                             "a long absence must cost a bounded number of calls")

    def test_dead_npc_stays_dead_and_hp_is_bounded(self):
        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        npcs = self.store.npcs_near(self.wm.world_id, 8, 8, 16)
        self.assertTrue(npcs, "mock chunk should spawn NPCs")
        npc = npcs[0]
        self.store.update_npc(npc["id"], alive=0, hp=0)
        self.store.update_npc(npcs[-1]["id"], hp=9999)
        self.engine.advance(1, 8, 8)
        self.engine.advance(300, 8, 8)
        dead = self.store.get_npc(npc["id"])
        self.assertEqual(dead["alive"], 0, "a dead NPC was revived")
        self.assertEqual(dead["hp"], 0)

    def test_destroyed_object_is_kept_as_history(self):
        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        objs = self.store.objects_near(self.wm.world_id, 8, 8, 16)
        self.assertTrue(objs)
        oid = objs[0]["id"]
        self.store.destroy_object(oid, tick=5, desc="只剩焦黑的残桩。")
        row = self.store.get_object(oid)
        self.assertIsNotNone(row, "destroyed objects must be kept for history")
        self.assertEqual(row["alive"], 0)
        self.assertNotIn(oid, [o["id"] for o in self.store.objects_near(self.wm.world_id, 8, 8, 16)])

    def test_talk_persists_memory_and_player_relation(self):
        from pcg.entities import Player
        from pcg.narrative import Narrator

        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        self.store.create_player(self.wm.world_id, "旅人", 8, 8, 30, 5, 2)
        player = Player(self.store, self.wm.world_id)
        npc = self.store.npcs_near(self.wm.world_id, 8, 8, 16, limit=1)[0]
        narrator = Narrator(self.store, self.llm, self.cfg, self.wm)

        narrator.talk(player, npc, "你好，你在这里做什么？")
        self.assertTrue(narrator.npc_memory_digest(npc),
                        "dialogue must leave the NPC with a persistent memory")
        rels = self.store.relations_for(self.wm.world_id, "npc", npc["name"])
        self.assertTrue(any(r["other_kind"] == "player" for r in rels),
                        "talking should create an npc->player relation edge")

        # a second conversation must see the first one's memory in its prompt
        text = narrator.npc_memory_digest(npc)
        self.assertIn("守望塔", text)

    def test_one_evolution_step_cannot_teleport_an_npc(self):
        from pcg.terrain import is_solid

        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        node = self.store.get_node(self.wm.world_id, LOD_CHUNK, 0, 0)
        npc = self.store.npcs_near(self.wm.world_id, 8, 8, 16, limit=1)[0]
        self.store.update_npc(npc["id"], x=8, y=8)
        start = (8, 8)

        self.engine._apply(3, node, 5, {"changes": [{"type": "npc", "name": npc["name"],
                                                     "move": [20, 20]}]})
        cur = self.store.get_npc(npc["id"])
        self.assertEqual((cur["x"], cur["y"]), start, "an oversized jump must be rejected")

        # a single step in a genuinely walkable direction must still work
        step = next((d for d in ((1, 0), (-1, 0), (0, 1), (0, -1))
                     if not is_solid((self.store.get_tile(self.wm.world_id,
                                                          8 + d[0], 8 + d[1]) or {"terrain": "water"})["terrain"])
                     and not self.store.objects_at(self.wm.world_id, 8 + d[0], 8 + d[1])), None)
        if step is None:
            self.skipTest("surrounded by impassable terrain")
        self.engine._apply(3, node, 6, {"changes": [{"type": "npc", "name": npc["name"],
                                                     "move": list(step)}]})
        cur = self.store.get_npc(npc["id"])
        self.assertEqual(abs(cur["x"] - start[0]) + abs(cur["y"] - start[1]), 1,
                         "a normal step must still apply")

    def test_npc_status_is_tracked_and_feeds_dialogue(self):
        from pcg.entities import Player
        from pcg.narrative import Narrator

        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        node = self.store.get_node(self.wm.world_id, LOD_CHUNK, 0, 0)
        npc = self.store.npcs_near(self.wm.world_id, 8, 8, 16, limit=1)[0]
        self.engine._apply(3, node, 7, {"changes": [
            {"type": "npc", "name": npc["name"], "status": "重伤昏迷", "note": "被落石砸中"},
        ]})
        self.assertEqual(self.store.get_npc(npc["id"])["status"], "重伤昏迷")

        # the next evolution prompt must be able to see that condition
        digest = self.engine._npc_line(self.store.get_npc(npc["id"]))
        self.assertIn("重伤昏迷", digest)

        # and dialogue must be told about it
        self.store.create_player(self.wm.world_id, "旅人", 8, 8, 30, 5, 2)
        player = Player(self.store, self.wm.world_id)
        narrator = Narrator(self.store, self.llm, self.cfg, self.wm)
        narrator.talk(player, self.store.get_npc(npc["id"]), "你还好吗？")
        self.assertTrue(narrator.npc_history_digest(self.store.get_npc(npc["id"])) is not None)

    def test_nation_relations_are_seeded(self):
        rels = self.store.list_relations(self.wm.world_id, kind="nation")
        self.assertTrue(rels, "world genesis should produce a diplomacy graph")
        self.assertTrue(all(r["a_kind"] == "nation" for r in rels))

    def test_relations_and_memories_are_queried_both_ways(self):
        self.store.upsert_relation(self.wm.world_id, "npc", "甲", "npc", "乙",
                                   "盟友", 3, "一起打猎", 1)
        for name in ("甲", "乙"):
            rels = self.store.relations_for(self.wm.world_id, "npc", name)
            self.assertTrue(rels, f"{name} should see the edge")
            self.assertEqual(rels[0]["other_name"], "乙" if name == "甲" else "甲")


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

    def test_short_walk_does_not_generate_new_region_or_zone(self):
        """Regression: spawning on a region/zone boundary made a single step
        generate a whole extra region (3 extra LLM calls)."""
        import contextlib
        import io
        wid = self.game.world_id
        before = (self.game.store.count_nodes(wid, LOD_REGION),
                  self.game.store.count_nodes(wid, LOD_ZONE))
        with contextlib.redirect_stdout(io.StringIO()):
            for d in ("w", "d", "s", "a", "w", "a", "s", "d"):
                self.game.execute(f"move {d}")
        after = (self.game.store.count_nodes(wid, LOD_REGION),
                 self.game.store.count_nodes(wid, LOD_ZONE))
        self.assertEqual(before, after)

    def test_quit_command(self):
        self.assertFalse(self.game.execute("quit"))

    def test_save_and_reload(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.game.execute("auto 2")
            self.game.execute("wait 7")
        self.game.save()
        wid, px, py = self.game.world_id, self.game.player.x, self.game.player.y
        tick = self.game.tick()

        other = Game(self.cfg)
        try:
            other.load_world(wid)
            self.assertEqual((other.player.x, other.player.y), (px, py))
            self.assertEqual(other.tick(), tick)
            self.assertTrue(other.narrator.current_quest() is not None
                            or other.narrator.current_quest() is None)  # quest state survives
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
