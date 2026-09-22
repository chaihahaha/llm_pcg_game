"""Story director + real-time NPC dialogue (both fully LLM-generated).

The narrator keeps the model's *working context* tiny by never resending the
world: it sends a world bible (fixed), a short player brief, and a digest of
recent events/dialogue.  Long-term memory lives in SQLite, not in the prompt.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import prompts
from .config import cfg_get
from .db import Store
from .llm import LLMClient
from .rng import hash_int
from .world import WorldManager


class Narrator:
    def __init__(self, store: Store, llm: LLMClient, cfg: Dict[str, Any], wm: WorldManager,
                 logger=None):
        self.store = store
        self.llm = llm
        self.cfg = cfg
        self.wm = wm
        self.logger = logger
        self.world_id = wm.world_id
        self.max_turns = int(cfg_get(cfg, "context.max_dialogue_turns", 6))
        self.max_event_digest = int(cfg_get(cfg, "context.max_event_digest", 12))

    # ------------------------------------------------------------------ briefs
    def player_brief(self, player) -> str:
        quest = self.current_quest()
        return (f"玩家：{player.data['name']}，等级 {player.data['level']}，"
                f"HP {player.hp}/{player.data['hp_max']}，金币 {player.data['gold']}，"
                f"坐标 ({player.x},{player.y})"
                + (f"，当前任务「{quest['title']}」" if quest else "，暂无任务"))

    def place_name(self, x: int, y: int) -> str:
        chunk = self.wm.ensure_node(3, x, y)
        zone = self.wm.ensure_node(2, x, y)
        return f"{zone.get('name','')}·{chunk.get('name','')}（{x},{y}）"

    def local_rumors(self, x: int, y: int) -> str:
        events = self.store.events_for_player(self.world_id, x, y, 48, self.max_event_digest)
        return prompts.digest_events(events, self.max_event_digest)

    # ------------------------------------------------------------------ story
    def current_quest(self) -> Optional[dict]:
        return self.store.current_story(self.world_id)

    def maybe_new_story(self, player, force: bool = False) -> Optional[dict]:
        if self.current_quest() and not force:
            return None
        quest = self.current_quest()
        recent = prompts.digest_events(
            self.store.events_for_player(self.world_id, player.x, player.y, 64, self.max_event_digest),
            self.max_event_digest)
        seed = hash_int("story", self.world_id, player.x, player.y, self.store.get_meta("tick", 0),
                        mod=2 ** 31)
        text = prompts.story_task(seed, self.player_brief(player), self.wm.world_bible(), recent,
                                  quest["summary"] if quest else "")
        msgs = prompts.build(self.wm.world_bible(), text)
        data = self.llm.json(msgs, task="story", default={})
        title = str(data.get("title") or "").strip()
        if not title:
            return None
        if quest:
            self.store.resolve_story(quest["id"], "replaced")
        self.store.set_story(
            self.world_id, int(self.store.get_meta("tick", 0) or 0), title,
            str(data.get("summary") or ""), "active",
            {"objective": str(data.get("objective") or ""),
             "stakes": str(data.get("stakes") or ""),
             "hint": str(data.get("hint") or "")},
        )
        self.store.add_event(self.world_id, int(self.store.get_meta("tick", 0) or 0), 3, None,
                             player.x, player.y, "story", f"新的线索：{title}")
        self.store.commit()
        return self.store.current_story(self.world_id)

    def complete_story(self) -> None:
        quest = self.current_quest()
        if quest:
            self.store.resolve_story(quest["id"], "done")
            self.store.commit()

    # --------------------------------------------------------------- dialogue
    def talk(self, player, npc: dict, line: str) -> Dict[str, Any]:
        hist_rows = self.store.recent_dialogue(npc["id"], self.max_turns * 2)
        history = "\n".join(
            f"{'玩家' if r['role'] == 'player' else npc['name']}：{r['content'][:120]}"
            for r in hist_rows)
        quest = self.current_quest()
        place = self.place_name(npc["x"], npc["y"])
        seed = hash_int("dlg", npc["id"], self.store.get_meta("tick", 0), mod=2 ** 31)
        text = prompts.dialogue_task(
            seed, npc, self.wm.world_bible(), place, self.local_rumors(npc["x"], npc["y"]),
            history, line, (f"{quest['title']}：{str(quest.get('data',{}).get('objective',''))}"
                            if quest else ""))
        msgs = prompts.build(self.wm.world_bible(), text)
        data = self.llm.json(msgs, task="dialogue", default={})
        reply = str(data.get("reply") or "……").strip()
        tick = int(self.store.get_meta("tick", 0) or 0)
        self.store.add_dialogue(self.world_id, npc["id"], tick, "player", line)
        self.store.add_dialogue(self.world_id, npc["id"], tick, "npc", reply)
        if data.get("mood"):
            self.store.update_npc(npc["id"], mood=str(data["mood"])[:16], updated_tick=tick)
        action = str(data.get("action") or "none")
        created_quest = None
        if action == "quest":
            ad = data.get("action_data") or {}
            title = str(ad.get("quest_title") or "").strip()
            if title:
                if quest:
                    self.store.resolve_story(quest["id"], "replaced")
                self.store.set_story(self.world_id, tick, title,
                                     str(ad.get("quest_summary") or ""), "active",
                                     {"source_npc": npc["name"], "objective": str(ad.get("quest_summary") or "")})
                created_quest = self.store.current_story(self.world_id)
        self.store.prune_dialogue(self.world_id)
        self.store.commit()
        return {"reply": reply, "mood": data.get("mood", ""), "action": action,
                "quest": created_quest}

    # ---------------------------------------------------------------- flavour
    def combat_flavor(self, attacker: str, defender: str, result: str, place: str) -> str:
        seed = hash_int("flavor", attacker, defender, self.store.get_meta("tick", 0), mod=2 ** 31)
        text = prompts.flavor_task(seed, attacker, defender, result, place)
        msgs = prompts.build(self.wm.world_bible(), text)
        data = self.llm.json(msgs, task="flavor", default={})
        return str(data.get("text") or "").strip()
