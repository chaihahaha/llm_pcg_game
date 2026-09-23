"""Free-form actions, the runtime patch layer, and the sandbox that guards it."""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pcg import sandbox  # noqa: E402
from pcg.config import load_config  # noqa: E402
from pcg.db import Store  # noqa: E402
from pcg.rules import RuntimeRules  # noqa: E402
from pcg.sandbox import SandboxError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fresh(name: str) -> str:
    p = os.path.join(ROOT, "data", name)
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(p + suf):
            os.remove(p + suf)
    return p


class TestSandbox(unittest.TestCase):
    def test_allows_useful_lambdas(self):
        fn = sandbox.compile_hook("lambda ctx: len(ctx['reply']) <= 10")
        self.assertTrue(fn({"reply": "短"}))
        self.assertFalse(fn({"reply": "x" * 40}))
        fn2 = sandbox.compile_hook("ctx['terrain'] == 'forest'")
        self.assertTrue(fn2({"terrain": "forest"}))

    def test_rejects_imports_and_dunder_and_io(self):
        for bad in (
            "lambda ctx: __import__('os').system('rm -rf /')",
            "lambda ctx: open('/etc/passwd').read()",
            "lambda ctx: ().__class__.__bases__",
            "lambda ctx: getattr(ctx, 'x')",
            "lambda ctx: eval('1+1')",
        ):
            with self.assertRaises(SandboxError, msg=bad):
                sandbox.compile_hook(bad)

    def test_rejects_while_and_imports_in_code_patches(self):
        for bad in ("import os", "while True:\n    pass", "class X:\n    pass",
                    "def f():\n    global x\n    x = 1"):
            self.assertIsNotNone(sandbox.validate_code(bad), bad)

    def test_code_patch_can_define_and_call_helpers(self):
        ns = sandbox.exec_patch(
            "def limit(text, n):\n"
            "    return text[:n]\n"
            "result = limit('abcdefghij', 4)\n",
            {},
        )
        self.assertEqual(ns["result"], "abcd")

    def test_code_patch_blocks_memory_bombs(self):
        self.assertIsNotNone(sandbox.validate_code("x = [0] * 10 ** 9"))
        self.assertIsNotNone(sandbox.validate_code("x = 999999999999"))


class TestRulesAndPatches(unittest.TestCase):
    def setUp(self):
        self.path = fresh("test_rules.db")
        self.store = Store(self.path)
        self.wid = self.store.create_world("测试界", 1, "第一纪", "概要", {})
        self.rules = RuntimeRules(self.store, self.wid)

    def tearDown(self):
        self.store.close()

    def test_rule_set_and_bounds(self):
        ok, _ = self.rules.set_rule("npc_speech_max_chars", 12)
        self.assertTrue(ok)
        self.assertEqual(self.rules.get("npc_speech_max_chars"), 12)
        ok, msg = self.rules.set_rule("npc_speech_max_chars", 999999)
        self.assertFalse(ok, "out-of-range values must be refused")
        ok, _ = self.rules.set_rule("nonexistent_rule", 1)
        self.assertFalse(ok)

    def test_speech_rule_actually_truncates(self):
        self.rules.set_rule("npc_speech_max_chars", 6)
        self.assertEqual(self.rules.sanitize_speech("这是一句很长的话"), "这是一句很长"[:6])
        self.rules.set_rule("forbidden_words", ["秘密"])
        self.assertIn("***", self.rules.sanitize_speech("这是秘密"))

    def test_terrain_rules_change_passability(self):
        from pcg.terrain import is_solid
        # second argument is "is it passable by default"
        self.assertFalse(self.rules.is_terrain_passable("forest", not is_solid("forest")))
        self.rules.set_rule("extra_passable_terrain", ["forest"])
        self.assertTrue(self.rules.is_terrain_passable("forest", not is_solid("forest")))

    def test_hook_add_and_sandbox_rejection(self):
        ok, _ = self.rules.add_hook("can_enter", "lambda ctx: True if ctx['terrain'] == 'forest' "
                                                  "else None")
        self.assertTrue(ok)
        self.assertIs(self.rules.call("can_enter", {"terrain": "forest"}), True)
        self.assertIsNone(self.rules.call("can_enter", {"terrain": "grass"}))
        ok, msg = self.rules.add_hook("can_enter", "lambda ctx: __import__('os')")
        self.assertFalse(ok)
        self.assertIn("沙箱", msg)

    def test_broken_stored_hook_is_disabled_not_fatal(self):
        self.store.upsert_patch(self.wid, "hook", "damage", expr="lambda ctx: 1/0", tick=0)
        self.store.commit()
        rules = RuntimeRules(self.store, self.wid)          # must not raise
        self.assertEqual(rules.call("damage", {}, default=7), 7)
        # a raising hook is caught, not propagated
        rules.hooks["damage"] = lambda ctx: 1 / 0
        self.assertEqual(rules.call("damage", {}, default=7), 7)

    def test_patches_persist_across_reload(self):
        self.rules.set_rule("move_cost_hours", 3)
        self.rules.add_hook("damage", "lambda ctx: 1")
        self.store.commit()
        again = RuntimeRules(self.store, self.wid)
        self.assertEqual(again.get("move_cost_hours"), 3)
        self.assertEqual(again.call("damage", {}), 1)

    def test_code_patch_persists_and_reapplies(self):
        api = {"set_rule": lambda key, value, reason="": self.rules.set_rule(key, value)[0]}
        self.rules.bind_api(api)
        ok, msg = self.rules.add_code_patch('api["set_rule"]("npc_speech_max_chars", 9)',
                                            reason="诅咒")
        self.assertTrue(ok, msg)
        self.assertEqual(self.rules.get("npc_speech_max_chars"), 9)

        again = RuntimeRules(self.store, self.wid)
        again.bind_api(api)
        again.run_code_patches()
        self.assertEqual(again.get("npc_speech_max_chars"), 9,
                         "a loaded save must re-apply its code patches")

    def test_bad_code_patch_is_rejected(self):
        self.rules.bind_api({"set_rule": lambda *a, **k: True})
        ok, msg = self.rules.add_code_patch("import os")
        self.assertFalse(ok)
        self.assertIn("沙箱", msg)


class TestFreeformActions(unittest.TestCase):
    def setUp(self):
        from pcg.game import Game
        cfg = load_config(None, {
            "llm": {"mock": True, "cache_path": fresh("test_act_cache.db")},
            "game": {"db_path": fresh("test_act.db")},
        })
        self.game = Game(cfg)
        self.game.auto_story = False
        self.game.new_world(seed=21)

    def tearDown(self):
        self.game.close()

    def run_line(self, line: str) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.game.execute(line)
        return buf.getvalue()

    def test_free_text_dig_creates_a_shaft_and_learns_the_action(self):
        g = self.game
        before = g.store.get_tile(g.world_id, g.player.x, g.player.y)["terrain"]
        out = self.run_line("我要向下挖一口通往地底的竖井")
        after = g.store.get_tile(g.world_id, g.player.x, g.player.y)
        self.assertNotEqual(after["terrain"], before)
        self.assertEqual(after["terrain"], "cave")
        self.assertIn("竖井", after["name"] + after["desc"])
        self.assertIn("学会新动作", out)
        self.assertTrue(g.store.get_action(g.world_id, "dig"))

    def test_unknown_command_is_treated_as_intent(self):
        out = self.run_line("随便翻找一下周围的碎石")
        self.assertIn("碎石块", out)
        self.assertIn("碎石块", self.game.player.inventory)

    def test_learned_action_replays_without_new_llm_call(self):
        self.run_line("我要向下挖一口竖井")
        calls = self.game.llm.stats["calls"]
        out = self.run_line("dig")            # exact name -> replay, no round-trip
        self.assertIn("复用已学会的动作", out)
        self.assertEqual(self.game.llm.stats["calls"], calls,
                         "replaying a stored action must not spend an LLM call")

    def test_world_rule_patch_changes_dialogue_immediately(self):
        g = self.game
        g.store.create_player(g.world_id, "旅人", g.player.x, g.player.y, 30, 5, 2)
        npc = g.store.npcs_near(g.world_id, g.player.x, g.player.y, 30, limit=1)[0]
        long_reply = g.narrator.talk(g.player, npc, "你好")["reply"]
        self.assertGreater(len(long_reply), 12)

        out = self.run_line("我要让所有 NPC 都不能说长句子")
        self.assertIn("※", out, "the patch must be announced")
        self.assertEqual(g.rules.get("npc_speech_max_chars"), 12)
        short = g.narrator.talk(g.player, npc, "再说一次")["reply"]
        self.assertLessEqual(len(short), 12,
                             "the curse must apply to the very next conversation")

    def test_rules_and_actions_are_listed(self):
        self.run_line("我要让所有 NPC 都不能说长句子")
        self.assertIn("npc_speech_max_chars", self.run_line("rules"))
        self.run_line("我要向下挖一口竖井")
        self.assertIn("dig", self.run_line("actions"))

    def test_patch_survives_save_and_reload(self):
        from pcg.game import Game
        self.run_line("我要让所有 NPC 都不能说长句子")
        wid = self.game.world_id
        other = Game(self.game.cfg)
        try:
            other.load_world(wid)
            self.assertEqual(other.rules.get("npc_speech_max_chars"), 12,
                             "a world-changing curse must outlive the session")
        finally:
            other.close()


if __name__ == "__main__":
    unittest.main()
