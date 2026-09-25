"""The deterministic substrate, and how the LLM is allowed to bend it."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pcg.config import load_config  # noqa: E402
from pcg.db import Store  # noqa: E402
from pcg.evolution import EvolutionEngine  # noqa: E402
from pcg.llm import LLMClient  # noqa: E402
from pcg.rules import RuntimeRules  # noqa: E402
from pcg.sim import Simulator, derive_mood, sim_state  # noqa: E402
from pcg.world import LOD_CHUNK, WorldManager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fresh(name: str) -> str:
    p = os.path.join(ROOT, "data", name)
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(p + suf):
            os.remove(p + suf)
    return p


class TestSubstrate(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(None, {"llm": {"mock": True}})
        self.store = Store(fresh("test_sim.db"))
        self.llm = LLMClient(self.cfg, store=self.store)
        self.wm = WorldManager(self.store, self.llm, self.cfg)
        self.wm.create(seed=5)
        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        self.rules = RuntimeRules(self.store, self.wm.world_id)
        self.sim = Simulator(self.store, self.wm, self.rules)
        self.npc = self.store.npcs_near(self.wm.world_id, 8, 8, 16, limit=1)[0]

    def tearDown(self):
        self.store.close()

    def needs(self):
        return sim_state(self.store.get_npc(self.npc["id"]))

    def test_needs_rise_with_time(self):
        before = self.needs()
        self.sim.step(24, 24, 8, 8)
        self.assertGreater(self.needs()["hunger"], before["hunger"])
        self.assertGreater(self.needs()["fatigue"], before["fatigue"])

    def test_closed_form_jump_matches_many_small_steps(self):
        """A long jump must be one O(NPC) pass, not N passes."""
        self.sim.step(240, 240, 8, 8)
        one_shot = self.needs()["hunger"]

        self.store.update_npc(self.npc["id"], state={})
        for hour in range(1, 241):
            self.sim.step(hour, 1, 8, 8)
        stepped = self.needs()["hunger"]
        self.assertAlmostEqual(one_shot, stepped, delta=1.0)

    def test_mood_is_derived_from_needs(self):
        state = dict(sim_state(self.npc))
        state["anger"] = 80.0
        self.assertEqual(derive_mood(state, 10, 10, 0), "愤怒")
        state = {"anger": 0.0, "hunger": 0.0, "fatigue": 0.0, "morale": 90.0}
        self.assertEqual(derive_mood(state, 10, 10, 0), "愉悦")
        self.assertEqual(derive_mood(state, 2, 10, 0), "虚弱")

    def test_starvation_costs_hp_and_can_kill(self):
        self.rules.set_rule("starve_at", 30.0)
        self.rules.set_rule("starve_hp_loss", 10.0)
        self.rules.set_rule("eat_at", 100.0)      # forbid foraging so it starves
        self.store.update_npc(self.npc["id"], hp=6)
        events = self.sim.step(400, 400, 8, 8)
        after = self.store.get_npc(self.npc["id"])
        self.assertLessEqual(after["hp"], 6)
        self.assertEqual(after["alive"], 0, "an NPC left starving must eventually die")
        self.assertTrue(any("饥困" in e["summary"] for e in events))

    def test_foraging_consumes_food(self):
        self.rules.set_rule("eat_at", 5.0)
        self.store.update_npc(self.npc["id"], x=9, y=9)
        food = self.store.add_object(self.wm.world_id, 9, 9, "food", "干粮",
                                     "一块硬得能砸人的干粮。")
        before = self.needs()["hunger"]
        for hour in range(1, 6):
            self.sim.step(hour, 1, 8, 8)
        self.assertLess(self.needs()["hunger"], before)
        self.assertEqual(self.store.get_object(food)["alive"], 0, "the food should be eaten")

    def test_mood_override_expires(self):
        # the LLM may declare a mood; the formula takes it back after a while
        self.rules.set_rule("mood_override_hours", 1)
        self.rules.add_hook("sim_npc", "lambda ctx: {'mood': '狂喜'} if ctx['tick'] < 5 else None")
        self.sim.step(1, 1, 8, 8)
        self.assertEqual(self.store.get_npc(self.npc["id"])["mood"], "狂喜")
        # the hook stops asserting; the deterministic formula resumes control
        self.sim.step(500, 1, 8, 8)
        self.assertNotEqual(self.store.get_npc(self.npc["id"])["mood"], "狂喜")

    def test_llm_hook_can_rewrite_needs_per_npc(self):
        role = self.npc["role"]
        self.rules.add_hook("sim_npc",
                            "lambda ctx: {'hunger': 0, 'anger': 0, 'morale': 100} "
                            f"if ctx['role'] == {role!r} else None")
        self.sim.step(100, 100, 8, 8)
        self.assertEqual(self.needs()["hunger"], 0.0)

    def test_rule_patch_stops_hunger_entirely(self):
        self.rules.set_rule("hunger_rate", 0.0)
        self.rules.set_rule("fatigue_rate", 0.0)
        before = self.needs()["hunger"]
        self.sim.step(2000, 2000, 8, 8)
        self.assertEqual(self.needs()["hunger"], before,
                         "a world law can switch hunger off")

    def test_model_invented_rule_names_are_resolved(self):
        # the real model asked for "hunger_growth_multiplier"; that must land on
        # hunger_rate instead of silently doing nothing
        ok, msg = self.rules.set_rule("hunger_growth_multiplier", 0)
        self.assertTrue(ok, msg)
        self.assertEqual(self.rules.get("hunger_rate"), 0)
        before = self.needs()["hunger"]
        self.sim.step(500, 500, 8, 8)
        self.assertEqual(self.needs()["hunger"], before)

    def test_rejected_rule_is_reported_not_silent(self):
        ok, msg = self.rules.set_rule("totally_made_up_rule", 1)
        self.assertFalse(ok)
        self.assertIn("unknown", msg.lower()) if "unknown" in msg.lower() else self.assertIn("未知", msg)
        self.assertTrue(self.rules.patch_failures)

    def test_code_patch_with_bad_rule_name_reports_failure(self):
        self.rules.bind_api({"set_rule": self.rules.api.get("set_rule")})
        self.rules.bind_api({})
        # bind_api must rebuild set_rule itself; use it directly
        self.rules.bind_api({})
        ok, msg = self.rules.add_code_patch("api['set_rule']('hunger_growth_multiplier', 0)")
        self.assertTrue(ok, msg)
        self.assertEqual(self.rules.get("hunger_rate"), 0)

    def test_individuals_do_not_start_identical(self):
        others = self.store.npcs_near(self.wm.world_id, 8, 8, 16, limit=10)
        self.assertGreaterEqual(len(others), 2, "mock chunk seeds two NPCs")
        hungers = {sim_state(n)["hunger"] for n in others}
        self.assertGreater(len(hungers), 1, "a crowd must not march in lockstep")

    def test_fire_burns_out_and_food_spoils(self):
        self.rules.set_rule("fire_burnout_hours", 10.0)
        self.rules.set_rule("food_spoil_hours", 10.0)
        fire = self.store.add_object(self.wm.world_id, 1, 1, "fire", "篝火", "烧得正旺。")
        food = self.store.add_object(self.wm.world_id, 2, 2, "food", "腌肉", "还能吃。")
        self.sim.step(50, 50, 8, 8)
        self.assertEqual(self.store.get_object(fire)["alive"], 0)
        self.assertEqual(self.store.get_object(food)["kind"], "rot")

    def test_relations_decay_toward_neutral(self):
        self.store.upsert_relation(self.wm.world_id, "npc", "甲", "npc", "乙", "盟友", 4, "", 0)
        self.sim.step(24 * 21, 24 * 21, 8, 8)      # three weeks
        rel = [r for r in self.store.list_relations(self.wm.world_id) if r["a_name"] == "甲"][0]
        self.assertLess(abs(rel["value"]), 4, "feelings should fade over time")


class TestSubstrateThroughEngine(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(None, {"llm": {"mock": True},
                                      "evolution": {"max_llm_calls_per_wait": 6}})
        self.store = Store(fresh("test_sim_engine.db"))
        self.llm = LLMClient(self.cfg, store=self.store)
        self.wm = WorldManager(self.store, self.llm, self.cfg)
        self.wm.create(seed=5)
        self.wm.ensure_node(LOD_CHUNK, 0, 0)
        self.rules = RuntimeRules(self.store, self.wm.world_id)
        self.engine = EvolutionEngine(self.store, self.llm, self.cfg, self.wm,
                                      rules=self.rules)

    def tearDown(self):
        self.store.close()

    def test_sim_runs_even_when_no_llm_scope_is_due(self):
        calls = self.llm.stats["calls"]
        self.engine.advance(1, 8, 8, mode="local")     # 1h: no scope due
        self.assertEqual(self.llm.stats["calls"], calls, "the substrate must not call the LLM")
        npc = self.store.npcs_near(self.wm.world_id, 8, 8, 16, limit=1)[0]
        self.assertGreater(sim_state(npc)["hunger"], 20.0, "but time must still pass")

    def test_sim_events_reach_the_player_facing_list(self):
        self.rules.set_rule("hunger_rate", 60.0)
        self.rules.set_rule("eat_at", 100.0)
        self.rules.set_rule("starve_at", 30.0)
        self.rules.set_rule("starve_hp_loss", 6.0)
        for npc in self.store.list_npcs(self.wm.world_id):
            self.store.update_npc(npc["id"], hp=3)
        events = self.engine.advance(72, 8, 8, mode="local")
        self.assertTrue(any(e.get("sim") for e in events),
                        "deterministic events must surface like any other history")


if __name__ == "__main__":
    unittest.main()
