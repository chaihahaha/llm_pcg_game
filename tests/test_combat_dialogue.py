"""Causal links between subsystems: the places where "logic holes" live.

The trigger was a real report — the player stabbed an NPC and the NPC then
chatted amiably.  Every test here asserts that two subsystems are *wired
together* rather than trusting the model to paper over the gap.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pcg import entities, render  # noqa: E402
from pcg.config import load_config  # noqa: E402
from pcg.game import Game  # noqa: E402
from pcg.sim import sim_state  # noqa: E402
from pcg.world import LOD_CHUNK  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fresh(name: str) -> str:
    p = os.path.join(ROOT, "data", name)
    for suf in ("", "-wal", "-shm"):
        if os.path.exists(p + suf):
            os.remove(p + suf)
    return p


class TestCombatDialogue(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(None, {
            "llm": {"mock": True, "cache_path": fresh("test_cd_cache.db")},
            "game": {"db_path": fresh("test_cd.db")},
        })
        self.game = Game(self.cfg)
        self.game.auto_story = False
        self.game.new_world(seed=4242)

    def tearDown(self):
        self.game.close()

    # -- helpers
    def say(self, line: str) -> str:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.game.execute(line)
        return buf.getvalue()

    def adjacent_npc(self) -> dict:
        g = self.game
        npc = g.store.npcs_near(g.world_id, g.player.x, g.player.y, 40, limit=1)[0]
        g.store.update_npc(npc["id"], x=g.player.x, y=g.player.y + 1)
        g.store.commit()
        return g.store.get_npc(npc["id"])

    # ------------------------------------------------------------------ combat
    def test_attack_makes_the_npc_hostile_with_memory_and_relation(self):
        npc = self.adjacent_npc()
        before = sim_state(npc)["anger"]
        self.say(f"attack {npc['name']}")
        after = self.game.store.get_npc(npc["id"])
        self.assertGreater(sim_state(after)["anger"], before)
        self.assertTrue(after["hostile"])
        self.assertTrue(any(m["kind"] == "grudge"
                            for m in self.game.store.npc_memories(npc["id"], limit=5)))
        rel = [r for r in self.game.store.relations_for(self.game.world_id, "npc", npc["name"])
               if r["other_kind"] == "player"]
        self.assertTrue(rel and rel[0]["value"] < 0)

    def test_cannot_talk_right_after_stabbing(self):
        npc = self.adjacent_npc()
        self.say(f"attack {npc['name']}")
        calls = self.game.llm.stats["calls"]
        out = self.say(f"talk {npc['name']} 我们聊聊")
        self.assertIn("没有回答", out)
        self.assertEqual(self.game.llm.stats["calls"], calls,
                         "a blocked conversation must not spend an LLM call")

    def test_hostile_neighbour_retaliates_when_player_acts(self):
        npc = self.adjacent_npc()
        self.game.store.update_npc(npc["id"], hostile=1, atk=6)
        self.game.store.commit()
        hp = self.game.player.hp
        self.say("move s")          # blocked by the NPC; the turn still resolves
        self.assertLess(self.game.player.hp, hp,
                        "an adjacent hostile NPC must get its turn")

    def test_attack_is_not_possible_at_range(self):
        g = self.game
        far = [n for n in g.store.list_npcs(g.world_id)
               if abs(n["x"] - g.player.x) + abs(n["y"] - g.player.y) > 5][0]
        hp = far["hp"]
        out = self.say(f"attack {far['name']}")
        self.assertEqual(g.store.get_npc(far["id"])["hp"], hp, "no remote stabbing")
        self.assertIn("格", out)

    # ------------------------------------------------------------------- death
    def test_dead_npc_cannot_be_talked_to_or_attacked(self):
        npc = self.adjacent_npc()
        for _ in range(40):
            self.game.execute("attack")
            if not self.game.store.get_npc(npc["id"])["alive"]:
                break
        self.assertEqual(self.game.store.get_npc(npc["id"])["alive"], 0)
        self.assertIn("已经死了", self.say(f"talk {npc['name']}"))
        self.assertIn("倒下", self.say(f"attack {npc['name']}"))

    def test_corpse_stays_on_the_map_and_is_inspectable(self):
        g = self.game
        npc = self.adjacent_npc()
        for _ in range(40):
            g.execute("attack")
            if not g.store.get_npc(npc["id"])["alive"]:
                break
        view = render.render_view(g.wm, g.player.x, g.player.y, 16, g.player)
        self.assertIn(render.CORPSE_SYM, view, "a killing must leave a trace")
        info = render.render_tile_info(g.wm, npc["x"], npc["y"], describe=False)
        self.assertIn("尸体", info)

    def test_down_player_cannot_act(self):
        g = self.game
        g.store.update_player(g.world_id, hp=0)
        g.player.refresh()
        out = self.say("do 我要挖一条竖井")
        self.assertIn("倒下", out)
        tile = g.store.get_tile(g.world_id, g.player.x, g.player.y)
        self.assertNotEqual(tile["terrain"], "cave", "a downed player must not reshape the map")
        self.assertIn("世界数据库", self.say("stats"))   # passive commands still work

    # ---------------------------------------------------------------- society
    def test_witness_reacts_to_a_killing(self):
        g = self.game
        victim = self.adjacent_npc()
        # a bystander who likes the victim
        witness = [n for n in g.store.list_npcs(g.world_id) if n["id"] != victim["id"]][0]
        g.store.update_npc(witness["id"], x=victim["x"] + 1, y=victim["y"])
        g.store.upsert_relation(g.world_id, "npc", witness["name"], "npc", victim["name"],
                                "盟友", 4, "过命的交情", 0)
        g.store.commit()
        for _ in range(40):
            g.execute("attack")
            if not g.store.get_npc(victim["id"])["alive"]:
                break
        after = g.store.get_npc(witness["id"])
        self.assertTrue(after["hostile"], "a friend's murderer should become an enemy")
        self.assertTrue(any(m["kind"] == "fact" for m in g.store.npc_memories(witness["id"])))

    def test_hostility_cools_only_after_the_configured_time(self):
        g = self.game
        npc = self.adjacent_npc()
        self.say(f"attack {npc['name']}")
        self.rules_hack(npc)
        # not yet: the grudge is fresh
        g.evolution.sim.step(24, 24, g.player.x, g.player.y)
        self.assertTrue(g.store.get_npc(npc["id"])["hostile"])
        # after the configured cooling window, and with anger spent, it relents
        g.evolution.sim.step(24 * 30, 24 * 30, g.player.x, g.player.y)
        self.assertFalse(g.store.get_npc(npc["id"])["hostile"])

    def rules_hack(self, npc: dict) -> None:
        """Heal and calm the NPC so only the *time* gate is under test."""
        g = self.game
        g.store.update_npc(npc["id"], hp=g.store.get_npc(npc["id"])["hp_max"])
        cur = g.store.get_npc(npc["id"])
        state = dict(cur.get("state") or {})
        sim = sim_state(cur)
        sim["anger"] = 0.0
        sim["grudge_tick"] = 0
        state["sim"] = sim
        g.store.update_npc(npc["id"], state=state)
        g.store.commit()

    # ------------------------------------------------------------ escape hatch
    def test_set_npc_effect_can_pacify_and_restore_dialogue(self):
        from pcg.effects import EffectContext, apply_program
        g = self.game
        npc = self.adjacent_npc()
        self.say(f"attack {npc['name']}")
        ctx = EffectContext(g.store, g.world_id, g.wm, g.rules, g.player, tick=1,
                            narrator=g.narrator, actions=g.actions, logger=g.logger)
        apply_program(ctx, {"effects": [
            {"op": "set_npc", "name": npc["name"], "hostile": 0, "anger": 0, "morale": 60,
             "hp": g.store.get_npc(npc["id"])["hp_max"], "mood": "警惕",
             "relation_value": 0, "relation_kind": "相识", "note": "玩家道歉并放下了武器。"},
        ]})
        ok, why = entities.can_talk(g.store, g.world_id, g.store.get_npc(npc["id"]))
        self.assertTrue(ok, why)

    # ------------------------------------------------------------- co-location
    def test_npcs_never_step_onto_the_player(self):
        """Regression: an NPC drifting onto your cell made it look like the
        character had vanished from the map."""
        import random
        g = self.game
        npc = self.adjacent_npc()
        g.store.update_npc(npc["id"], x=g.player.x, y=g.player.y + 1)
        g.store.commit()
        for _ in range(60):
            for d in ("w", "s", "d", "a"):
                g.execute(f"move {d}")
            for other in g.store.npcs_near(g.world_id, g.player.x, g.player.y, 2, limit=10):
                self.assertFalse(
                    other["x"] == g.player.x and other["y"] == g.player.y,
                    f"{other['name']} is standing on the player at "
                    f"({other['x']},{other['y']})")
        del random

    def test_engine_refuses_to_move_an_npc_onto_the_player(self):
        g = self.game
        node = g.store.get_node(g.world_id, LOD_CHUNK,
                                (g.player.x // 16) * 16, (g.player.y // 16) * 16)
        npc = self.adjacent_npc()
        g.evolution._apply(3, node, 5, {"changes": [
            {"type": "npc", "name": npc["name"],
             "move": [0 if g.player.x == npc["x"] else (1 if g.player.x > npc["x"] else -1),
                      1 if g.player.y > npc["y"] else -1]},
        ]})
        landed = g.store.get_npc(npc["id"])
        self.assertFalse(landed["x"] == g.player.x and landed["y"] == g.player.y)

    def test_renderer_reports_a_shared_cell_instead_of_hiding_it(self):
        g = self.game
        npc = g.store.npcs_near(g.world_id, g.player.x, g.player.y, 40, limit=1)[0]
        g.store.update_npc(npc["id"], x=g.player.x, y=g.player.y)   # force legacy state
        g.store.commit()
        view = render.render_view(g.wm, g.player.x, g.player.y, 16, g.player)
        self.assertIn("同格", view)
        self.assertIn(npc["name"], view)

    # ------------------------------------------------------------------ sanity
    def test_talk_to_a_named_npc_far_away_reports_distance_not_a_neighbour(self):
        """Regression: the name used to be dropped, so you addressed the corpse
        beside you instead of the person you named."""
        g = self.game
        adjacent = self.adjacent_npc()
        for _ in range(40):
            g.execute("attack")
            if not g.store.get_npc(adjacent["id"])["alive"]:
                break
        far = [n for n in g.store.list_npcs(g.world_id)
               if abs(n["x"] - g.player.x) + abs(n["y"] - g.player.y) > 5][0]
        out = self.say(f"talk {far['name']} 喂")
        self.assertIn(far["name"], out)
        self.assertIn("格", out)
        self.assertNotIn("已经死了", out)


if __name__ == "__main__":
    unittest.main()
