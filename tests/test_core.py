"""Unit tests (stdlib unittest, no third-party deps).

Run:  python3 -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pcg import rng, terrain  # noqa: E402
from pcg.config import load_config  # noqa: E402
from pcg.db import Store  # noqa: E402
from pcg.llm import MockBackend, parse_json_loose  # noqa: E402
from pcg.tokens import clamp_text, estimate_tokens  # noqa: E402


def tmp_db(name: str) -> str:
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, name)
    if os.path.exists(path):
        os.remove(path)
    return path


class TestTokens(unittest.TestCase):
    def test_estimate_scales(self):
        self.assertGreater(estimate_tokens("你好世界"), estimate_tokens("hi"))
        self.assertEqual(estimate_tokens(""), 0)

    def test_clamp(self):
        text = "一二三四五六七八九十" * 100
        out = clamp_text(text, 50)
        self.assertLessEqual(estimate_tokens(out), 60)
        self.assertIn("截断", out)


class TestRng(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(rng.hash_int("a", 1, mod=100), rng.hash_int("a", 1, mod=100))
        self.assertNotEqual(rng.hash_int("a", 1, mod=1000), rng.hash_int("a", 2, mod=1000))

    def test_noise_range(self):
        for x in range(0, 60, 7):
            v = rng.fbm(1, x, x * 2, 24.0)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_weighted_pick_membership(self):
        pairs = [("a", 1.0)]
        self.assertEqual(rng.weighted_pick(pairs, "seed"), "a")


class TestTerrain(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(terrain.normalize("Woods"), "forest")
        self.assertEqual(terrain.normalize("水面"), "grass")  # unknown -> fallback
        self.assertEqual(terrain.normalize("", "water"), "water")

    def test_mix(self):
        mix = terrain.normalize_mix([["grass", 0.5], ["Forest", 0.5]])
        self.assertEqual(dict(mix)["forest"], 0.5)
        self.assertTrue(terrain.normalize_mix(None))

    def test_legend_and_symbol(self):
        self.assertIn("forest", terrain.legend_text())
        self.assertEqual(terrain.symbol_of("forest"), "T")
        self.assertTrue(terrain.is_solid("water"))
        self.assertFalse(terrain.is_solid("grass"))


class TestJsonLoose(unittest.TestCase):
    def test_fenced(self):
        self.assertEqual(parse_json_loose('```json\n{"a": 1}\n```'), {"a": 1})

    def test_trailing_comma_and_quotes(self):
        self.assertEqual(parse_json_loose("{'a': [1, 2,]}"), {"a": [1, 2]})

    def test_prose_wrapped(self):
        self.assertEqual(parse_json_loose('好的：\n{"a": "b"}\n以上'), {"a": "b"})

    def test_garbage(self):
        self.assertIsNone(parse_json_loose("not json at all"))


class TestStore(unittest.TestCase):
    def setUp(self):
        self.store = Store(tmp_db("test_store.db"))
        self.wid = self.store.create_world("测试界", 42, "第一纪", "概要", {"x": 1})

    def tearDown(self):
        self.store.close()

    def test_world_roundtrip(self):
        w = self.store.get_world(self.wid)
        self.assertEqual(w["name"], "测试界")
        self.assertEqual(w["data"], {"x": 1})

    def test_nodes_and_tiles(self):
        nid = self.store.upsert_node(self.wid, 3, 0, 0, 16, 16, "chunk", "c", "s", {}, None)
        self.assertEqual(self.store.get_node(self.wid, 3, 0, 0)["id"], nid)
        self.store.upsert_tile(self.wid, 3, 4, "forest", biome="forest")
        self.assertEqual(self.store.get_tile(self.wid, 3, 4)["terrain"], "forest")
        self.assertEqual(len(self.store.list_tiles(self.wid, 0, 0, 5, 5)), 1)
        self.assertEqual(self.store.count_nodes(self.wid, 3), 1)

    def test_objects_and_npcs(self):
        oid = self.store.add_object(self.wid, 1, 2, "rock", "石头", "硬", solid=1)
        self.assertEqual(self.store.objects_at(self.wid, 1, 2)[0]["id"], oid)
        nid = self.store.add_npc(self.wid, 1, 2, "阿岚", race="人族", role="铁匠")
        self.assertEqual(self.store.find_npc_by_name(self.wid, "阿岚")["id"], nid)
        self.assertEqual(self.store.count_npcs(self.wid), 1)
        self.store.update_npc(nid, alive=0)
        self.assertEqual(self.store.count_npcs(self.wid), 0)

    def test_events_and_story(self):
        self.store.add_event(self.wid, 3, 3, None, 1, 1, "wildlife", "野猪出没")
        self.assertEqual(len(self.store.recent_events(self.wid, limit=5)), 1)
        self.assertEqual(len(self.store.events_for_player(self.wid, 1, 1, 2)), 1)
        self.store.set_story(self.wid, 3, "任务", "摘要")
        self.assertEqual(self.store.current_story(self.wid)["title"], "任务")
        self.store.resolve_story(self.store.current_story(self.wid)["id"])
        self.assertIsNone(self.store.current_story(self.wid))

    def test_player(self):
        self.store.create_player(self.wid, "旅人", 8, 8, 30, 5, 2)
        p = self.store.get_player(self.wid)
        self.assertEqual((p["x"], p["y"], p["hp"]), (8, 8, 30))
        self.store.update_player(self.wid, hp=10)
        self.assertEqual(self.store.get_player(self.wid)["hp"], 10)

    def test_llm_cache(self):
        self.store.cache_put("k1", "world", '{"a":1}')
        self.assertEqual(self.store.cache_get("k1"), '{"a":1}')
        self.assertEqual(self.store.cache_count(), 1)


class TestMockBackend(unittest.TestCase):
    def test_every_task_returns_json(self):
        import json

        from pcg import prompts
        be = MockBackend()
        bible = prompts.world_bible({"name": "界", "era": "纪"}, [])
        cases = [
            ("world", prompts.world_task(1)),
            ("region", prompts.region_task(1, "界", 0, 0, 512)),
            ("zone", prompts.zone_task(1, {"name": "r", "summary": "s", "data": {}}, 0, 0, 128)),
            ("chunk", prompts.chunk_task(1, {"name": "z", "summary": "s", "data": {}}, 0, 0, 16)),
            ("tile", prompts.tile_task(1, {"terrain": "grass", "x": 0, "y": 0}, "z", "s", "", "")),
            ("evolve_chunk", prompts.evolve_task("chunk", 1, "d", "", "", "", 1, 1, True)),
            ("dialogue", prompts.dialogue_task(1, {"name": "A"}, "w", "p", "", "", "你好", "")),
            ("story", prompts.story_task(1, "p", "w", "", "")),
            ("flavor", prompts.flavor_task(1, "我", "它", "命中", "旷野")),
        ]
        for task, text in cases:
            raw, finish = be.chat(prompts.build(bible, text), 512, 0.8, True)
            self.assertEqual(finish, "stop")
            data = json.loads(raw)
            self.assertIsInstance(data, dict, task)
        # chunk rows must be a 16-char 16-row matrix
        raw, _ = be.chat(prompts.build(bible, prompts.chunk_task(1, {"name": "z", "data": {}}, 0, 0, 16)),
                         512, 0.8, True)
        rows = json.loads(raw)["rows"]
        self.assertEqual(len(rows), 16)
        self.assertTrue(all(len(r) == 16 for r in rows))


class TestConfig(unittest.TestCase):
    def test_defaults_merge(self):
        cfg = load_config(None, {"llm": {"mock": True}, "game": {"db_path": "x.db"}})
        self.assertTrue(cfg["llm"]["mock"])
        self.assertEqual(cfg["game"]["db_path"], "x.db")
        self.assertEqual(cfg["world"]["chunk_size"], 16)


if __name__ == "__main__":
    unittest.main()
