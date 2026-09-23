"""Multi-LOD evolution.

Every scope in the LOD tree evolves on its own clock:

    chunk  : every 6 game-hours   (local weather, wildlife, small changes)
    zone   : every 48 game-hours  (settlements, travellers, local politics)
    region : every 168 hours      (economy, mining, wars, migrations)
    world  : every 720 hours      (eras, magic tides, empires)

Crucially, a local evolution call is given **both** its own neighbouring
objects (from the DB) **and** a digest of the higher-LOD evolutions that
overlap it (world/region/zone events).  A world-level "magic tide rises" or a
region-level "mine output doubles" therefore shows up concretely as new
objects / NPC behaviour at the tile level.

The LLM only ever returns small structured deltas, which are whitelist-applied
to the database.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from . import prompts
from .config import cfg_get
from .db import Store
from .llm import LLMClient
from .rng import hash_int, hash_unit
from .terrain import normalize
from .tokens import estimate_tokens
from .world import LOD_CHUNK, LOD_NAMES, LOD_REGION, LOD_WORLD, LOD_ZONE, WorldManager

_MAX_TOKEN_TASKS = 2200


class EvolutionEngine:
    def __init__(self, store: Store, llm: LLMClient, cfg: Dict[str, Any], wm: WorldManager,
                 logger=None):
        self.store = store
        self.llm = llm
        self.cfg = cfg
        self.wm = wm
        self.logger = logger
        self.world_id = wm.world_id
        self.enabled = bool(cfg_get(cfg, "evolution.enabled", True))
        self.schedule = {
            LOD_CHUNK: int(cfg_get(cfg, "evolution.schedule.chunk", 6)),
            LOD_ZONE: int(cfg_get(cfg, "evolution.schedule.zone", 48)),
            LOD_REGION: int(cfg_get(cfg, "evolution.schedule.region", 168)),
            LOD_WORLD: int(cfg_get(cfg, "evolution.schedule.world", 720)),
        }
        self.max_calls = int(cfg_get(cfg, "evolution.max_llm_calls_per_wait", 4))
        self.near_radius = int(cfg_get(cfg, "evolution.neighbor_radius", 6))
        self.max_neighbors = int(cfg_get(cfg, "evolution.max_neighbors", 12))
        self.max_local_events = int(cfg_get(cfg, "evolution.max_local_events", 8))
        self.max_higher_events = int(cfg_get(cfg, "evolution.max_higher_events", 6))
        self.max_prompt_tokens = int(cfg_get(cfg, "context.max_prompt_tokens", 24000))
        self.catchup_steps = int(cfg_get(cfg, "evolution.catchup_steps", 3))
        self.max_catchup_calls = int(cfg_get(cfg, "evolution.max_catchup_calls", 8))

    # ------------------------------------------------------------------ tick
    def tick(self) -> int:
        return int(self.store.get_meta("tick", 0) or 0)

    def advance(self, hours: int, px: int, py: int) -> List[dict]:
        """Advance world time and evolve every due scope, anywhere in the world.

        Scopes are *not* limited to the player's vicinity: an area the player
        left behind must keep living, otherwise returning to it reveals a
        frozen museum.  When a scope is overdue by several periods the elapsed
        time is summarised in a single "catch-up" call rather than replayed
        step by step.
        """
        if hours <= 0:
            return []
        start = self.tick()
        new_tick = start + int(hours)
        self.store.set_meta("tick", new_tick)
        if not self.enabled:
            return []
        plan = self._plan(new_tick, px, py)
        budget = self.max_calls
        if any(steps >= self.catchup_steps for _, _, steps, _ in plan):
            budget = max(budget, self.max_catchup_calls)
        produced: List[dict] = []
        for lod, scope, steps, elapsed in plan[:budget]:
            try:
                produced.extend(self.evolve_scope(lod, scope, new_tick, px, py,
                                                  elapsed_hours=elapsed))
            except Exception as exc:  # noqa: BLE001 - never let one scope kill the tick
                if self.logger:
                    self.logger.warning("evolution failed at lod %s node %s: %s",
                                        lod, scope.get("id"), exc)
        self._npc_drift(px, py, new_tick)
        self.store.prune_events(self.world_id)
        self.store.commit()
        return produced

    def _plan(self, tick: int, px: int, py: int) -> List[Tuple[int, dict, int, int]]:
        """(lod, node, overdue_steps, elapsed_hours) for every due scope.

        Coarsest LOD first so this advance's macro events are available to the
        finer prompts that follow.  Nearest chunks come next, so a player who
        just walked somewhere always gets that place resolved first.
        """
        dist = lambda n: abs(n["x"] - px) + abs(n["y"] - py)  # noqa: E731
        higher: List[Tuple[int, dict, int, int]] = []
        for lod in (LOD_WORLD, LOD_REGION, LOD_ZONE):
            if lod != LOD_WORLD:
                # only scopes that have actually been generated
                nodes = sorted(self.store.nodes_of_lod(self.world_id, lod), key=dist)
            else:
                nodes = self.store.nodes_of_lod(self.world_id, LOD_WORLD)
            for node in nodes:
                elapsed = tick - int(node.get("updated_tick") or 0)
                steps = elapsed // max(1, self.schedule[lod])
                if steps >= 1:
                    higher.append((lod, node, steps, elapsed))
        higher.sort(key=lambda t: (t[0], dist(t[1])))

        chunks: List[Tuple[int, dict, int, int]] = []
        for node in sorted(self.store.nodes_of_lod(self.world_id, LOD_CHUNK), key=dist):
            elapsed = tick - int(node.get("updated_tick") or 0)
            steps = elapsed // max(1, self.schedule[LOD_CHUNK])
            if steps >= 1:
                chunks.append((LOD_CHUNK, node, steps, elapsed))

        local_budget = int(cfg_get(self.cfg, "evolution.max_chunk_scopes_per_advance", 2))
        plan = higher + chunks[:local_budget]
        plan += chunks[local_budget:]
        return plan

    # ---------------------------------------------------------------- scoping
    def evolve_scope(self, lod: int, node: dict, tick: int, px: int, py: int,
                     elapsed_hours: int = 0) -> List[dict]:
        day, hour = tick // 24 + 1, tick % 24
        scope_desc = self._scope_desc(lod, node)
        local_digest = prompts.digest_events(
            self.store.recent_events(self.world_id, lods=[lod], node_ids=[node["id"]],
                                     limit=self.max_local_events), self.max_local_events)
        higher_digest = self._higher_digest(lod, node)
        neighbor_digest = self._neighbor_digest(lod, node)

        allow_tiles = lod == LOD_CHUNK
        task = f"evolve_{LOD_NAMES[lod]}"
        seed = hash_int("evolve", self.world_id, node["id"], tick, mod=2 ** 31)
        text = prompts.evolve_task(LOD_NAMES[lod], seed, scope_desc, local_digest,
                                   higher_digest, neighbor_digest, hour, day, allow_tiles,
                                   elapsed_hours=elapsed_hours,
                                   period_hours=self.schedule[lod],
                                   bbox=(node["x"], node["y"], node["w"], node["h"]))
        text = self._fit_budget(text)
        msgs = prompts.build(self.wm.world_bible(), text)
        data = self.llm.json(msgs, task=task, default={})
        return self._apply(lod, node, tick, data)

    def _scope_desc(self, lod: int, node: dict) -> str:
        data = node.get("data") or {}
        bits = [f"{LOD_NAMES[lod]}「{node.get('name','')}」", f"概要：{str(node.get('summary',''))[:200]}"]
        if lod == LOD_WORLD:
            bits.append("国家：" + prompts.digest_nations(self.store.list_nations(self.world_id)))
            bits.append(f"纪元：{(self.store.get_world(self.world_id) or {}).get('era','')}")
        elif lod == LOD_REGION:
            bits.append(f"气候：{data.get('climate','')}")
            bits.append("文化：" + "、".join(data.get("cultures", [])[:3]))
            bits.append("危险：" + "、".join(data.get("hazards", [])[:3]))
        elif lod == LOD_ZONE:
            bits.append("危险：" + "、".join(data.get("hazards", [])[:3]))
            bits.append("地物：" + "、".join(f.get("name", "") for f in data.get("features", [])[:3]))
        else:
            bits.append(f"天气：{data.get('weather','')}")
            bits.append("地物：" + "、".join(f.get("name", "") for f in data.get("features", [])[:3]))
        if lod in (LOD_WORLD, LOD_REGION):
            rels = self.store.list_relations(self.world_id, kind="nation", limit=8)
            if rels:
                bits.append("国家关系：" + "；".join(
                    f"{r['a_name']}--{r['kind']}({r['value']})-->{r['b_name']}" for r in rels[:6]))
        if lod == LOD_CHUNK:
            npcs = self.store.npcs_near(self.world_id, node["x"] + 8, node["y"] + 8, 12, limit=8)
            rel_lines = []
            for n in npcs:
                for r in self.store.relations_for(self.world_id, "npc", n["name"], limit=3):
                    rel_lines.append(f"{n['name']}--{r['kind']}({r['value']})-->{r['other_name']}")
            if rel_lines:
                bits.append("人物关系：" + "；".join(rel_lines[:6]))
        stats = self.store.stats(self.world_id)
        bits.append(f"世界规模：地块{stats['tiles']}/物体{stats['objects']}/NPC{stats['npcs']}")
        return "｜".join(bits)

    def _higher_digest(self, lod: int, node: dict) -> str:
        """Digest of evolutions at LODs above this one that overlap it."""
        if lod == LOD_WORLD:
            return "（本层已是最高层级，需自行把握宏观趋势）"
        ancestors = []
        if lod == LOD_CHUNK:
            ancestors = [self.store.get_node(self.world_id, LOD_ZONE, node["x"], node["y"]),
                         self.store.get_node(self.world_id, LOD_REGION, node["x"], node["y"])]
        elif lod == LOD_ZONE:
            ancestors = [self.store.get_node(self.world_id, LOD_REGION, node["x"], node["y"])]
        node_ids = [a["id"] for a in ancestors if a]
        events = self.store.recent_events(self.world_id, lods=[LOD_REGION, LOD_WORLD, LOD_ZONE],
                                          limit=40)
        picked = []
        for ev in events:
            if ev["lod"] >= lod:
                continue
            if ev["node_id"] in node_ids or ev["lod"] == LOD_WORLD:
                picked.append(ev)
        picked = picked[:self.max_higher_events]
        if not picked:
            return "（近期无更高层级的重大演化）"
        return "；".join(f"[{LOD_NAMES.get(e['lod'],'?')}] {e['summary'][:70]}" for e in picked)

    def _neighbor_digest(self, lod: int, node: dict) -> str:
        if lod == LOD_CHUNK:
            cx, cy = node["x"], node["y"]
            objs = self.store.objects_near(self.world_id, cx + 8, cy + 8, 12, self.max_neighbors)
            # prefer the ones actually inside this chunk
            inside = [o for o in objs if cx <= o["x"] < cx + 16 and cy <= o["y"] < cy + 16]
            others = [o for o in objs if o not in inside]
            picks = (inside[: self.max_neighbors] + others[: max(0, self.max_neighbors - len(inside))])
            npcs = self.store.npcs_near(self.world_id, cx + 8, cy + 8, 12, limit=8)
            parts = []
            if picks:
                parts.append("物体：" + "；".join(
                    f"#{o['id']}[{o['x']},{o['y']}]{o['name']}({o['kind']})" for o in picks))
            if npcs:
                parts.append("NPC：" + "；".join(
                    f"[{n['name']}@{n['x']},{n['y']},{n['role']},HP{n['hp']},{n['mood']}]"
                    + (f"记得：{m['text'][:30]}" if (m := (self.store.npc_memories(n["id"], 1) or [None])[0])
                       else "")
                    for n in npcs))
            return "\n".join(parts)
        if lod == LOD_ZONE:
            subs = self.store.nodes_in_rect(self.world_id, LOD_CHUNK, node["x"], node["y"],
                                            node["x"] + self.wm.zone_size - 1,
                                            node["y"] + self.wm.zone_size - 1)
            return "；".join(f"{n['name']}({str(n.get('summary',''))[:40]})" for n in subs[:6])
        if lod == LOD_REGION:
            subs = self.store.nodes_in_rect(self.world_id, LOD_ZONE, node["x"], node["y"],
                                            node["x"] + self.wm.region_size - 1,
                                            node["y"] + self.wm.region_size - 1)
            return "；".join(f"{n['name']}({str(n.get('summary',''))[:40]})" for n in subs[:6])
        return prompts.digest_nations(self.store.list_nations(self.world_id))

    # ------------------------------------------------------------------ apply
    def _apply(self, lod: int, node: dict, tick: int, data: Dict[str, Any]) -> List[dict]:
        if not isinstance(data, dict):
            data = {}
        created: List[dict] = []
        summary = str(data.get("summary") or "").strip()
        weather = str(data.get("weather") or "").strip()

        node_data = dict(node.get("data") or {})
        if weather and lod == LOD_CHUNK:
            node_data["weather"] = weather[:20]
        changes = data.get("changes") if isinstance(data.get("changes"), list) else []

        for ch in changes[:6]:
            if not isinstance(ch, dict):
                continue
            try:
                self._apply_change(lod, node, tick, ch)
            except Exception as exc:  # noqa: BLE001
                if self.logger:
                    self.logger.debug("skip change %s: %s", ch, exc)

        # world era can be updated directly
        if lod == LOD_WORLD and data.get("era"):
            self.store.update_world(self.world_id, era=str(data["era"])[:40], updated_tick=tick)

        center_x = node["x"] + (node["w"] // 2)
        center_y = node["y"] + (node["h"] // 2)
        for ev in (data.get("events") or [])[:4]:
            if not isinstance(ev, dict):
                continue
            text = str(ev.get("text") or "").strip()
            if not text:
                continue
            kind = str(ev.get("kind") or "event")[:20]
            eid = self.store.add_event(self.world_id, tick, lod, node["id"],
                                       center_x, center_y, kind, text[:200])
            created.append({"id": eid, "tick": tick, "lod": lod, "kind": kind, "summary": text,
                            "x": center_x, "y": center_y, "node_id": node["id"]})

        self.store.update_node(node["id"], summary=summary or None, data=node_data, tick=tick)
        self.store.commit()
        return created

    def _apply_change(self, lod: int, node: dict, tick: int, ch: Dict[str, Any]) -> None:
        kind = str(ch.get("type") or "").strip()

        if kind == "new_object":
            x, y = int(ch.get("x", -1)), int(ch.get("y", -1))
            if not self._in_scope(node, x, y):
                return
            otype = str(ch.get("kind", "object"))[:16]
            self.store.add_object(self.world_id, x, y, otype, str(ch.get("name", ""))[:30],
                                  str(ch.get("desc", ""))[:200], tick=tick)

        elif kind == "new_npc":
            x, y = int(ch.get("x", -1)), int(ch.get("y", -1))
            if not self._in_scope(node, x, y):
                return
            name = str(ch.get("name", "")).strip()[:20]
            if not name or self.store.find_npc_by_name(self.world_id, name):
                return
            if self.store.count_npcs(self.world_id) >= 200:
                return
            self.store.add_npc(self.world_id, x, y, name, race=str(ch.get("race", ""))[:16],
                               role=str(ch.get("role", ""))[:20],
                               personality=str(ch.get("personality", ""))[:60],
                               state={"origin_node": node["id"]}, tick=tick)

        elif kind == "npc":
            name = str(ch.get("name", "")).strip()
            npc = self.store.find_npc_by_name(self.world_id, name) if name else None
            if not npc:
                return
            if not npc.get("alive", 1):
                return  # the dead do not evolve
            patch: Dict[str, Any] = {}
            if ch.get("hp_delta") is not None:
                hp = max(0, min(int(npc["hp_max"]), int(npc["hp"]) + int(ch["hp_delta"])))
                patch["hp"] = hp
                if hp == 0:
                    patch["alive"] = 0
            mv = ch.get("move")
            if isinstance(mv, (list, tuple)) and len(mv) == 2:
                nx, ny = int(npc["x"]) + int(mv[0]), int(npc["y"]) + int(mv[1])
                if self._in_scope(node, nx, ny):
                    tile = self.store.get_tile(self.world_id, nx, ny)
                    if tile is None or tile["terrain"] not in ("water", "mountain", "lava"):
                        patch["x"], patch["y"] = nx, ny
            if ch.get("mood"):
                patch["mood"] = str(ch["mood"])[:16]
            note = str(ch.get("note", ""))[:80]
            if note:
                state = dict(npc.get("state") or {})
                state["last"] = note
                patch["state"] = state
            patch["updated_tick"] = tick
            self.store.update_npc(npc["id"], **patch)

        elif kind == "object":
            oid = ch.get("id")
            if oid is None:
                return
            obj = self.store.get_object(int(oid))
            if obj is None:
                return
            if ch.get("destroyed"):
                if obj.get("alive", 1):
                    self.store.destroy_object(int(oid), tick=tick,
                                              desc=str(ch.get("desc", "") or "")[:200])
                    self.store.add_event(self.world_id, tick, lod, node["id"],
                                         obj["x"], obj["y"], "destruction",
                                         f"{obj['name']}被毁")
            else:
                patch: Dict[str, Any] = {}
                if ch.get("desc"):
                    # keep history: a new description is appended, never a rewrite
                    prev = (obj.get("state") or {}).get("last_desc") or obj.get("desc") or ""
                    patch["desc"] = (str(ch["desc"])[:200] if not prev
                                     else f"{str(ch['desc'])[:120]}（原为：{prev[:60]}）")
                if ch.get("hp") is not None:
                    patch["hp"] = max(0, min(int(obj.get("hp_max") or 0) or 10 ** 6,
                                             int(ch["hp"])))
                patch["updated_tick"] = tick
                self.store.update_object(int(oid), **patch)

        elif kind == "tile" and lod == LOD_CHUNK:
            x, y = int(ch.get("x", -1)), int(ch.get("y", -1))
            if not self._in_scope(node, x, y):
                return
            terrain = normalize(str(ch.get("terrain", ""))) if ch.get("terrain") else None
            self.store.update_tile_fields(self.world_id, x, y,
                                          terrain=terrain,
                                          desc=str(ch.get("desc", ""))[:200] or None,
                                          updated_tick=tick)

        elif kind == "relation":
            a_kind = str(ch.get("a_kind", "")).strip().lower()
            b_kind = str(ch.get("b_kind", "")).strip().lower()
            a_name = str(ch.get("a_name", "")).strip()[:32]
            b_name = str(ch.get("b_name", "")).strip()[:32]
            if a_kind not in ("nation", "npc") or b_kind not in ("nation", "npc"):
                return
            if not self._entity_exists(a_kind, a_name) or not self._entity_exists(b_kind, b_name):
                return
            self.store.upsert_relation(
                self.world_id, a_kind, a_name, b_kind, b_name,
                kind=str(ch.get("kind", "中立"))[:16],
                value=int(ch.get("value", 0) or 0),
                note=str(ch.get("note", ""))[:160], tick=tick,
            )

        elif kind == "memory":
            name = str(ch.get("name", "")).strip()
            npc = self.store.find_npc_by_name(self.world_id, name) if name else None
            if not npc:
                return
            self.store.add_memory(self.world_id, npc["id"], tick,
                                  str(ch.get("kind", "fact"))[:16],
                                  str(ch.get("text", ""))[:200],
                                  about=str(ch.get("about", ""))[:32])
            self.store.prune_memories(npc["id"], keep=12)

        elif kind == "nation" and lod in (LOD_REGION, LOD_WORLD):
            name = str(ch.get("name", "")).strip()
            if not name:
                return
            nation = self.store.get_nation(self.world_id, name)
            if not nation or not isinstance(ch.get("set"), dict):
                return
            allowed = {"gov", "tech", "magic", "summary"}
            patch = {k: str(v)[:120] for k, v in ch["set"].items() if k in allowed}
            if patch:
                self.store.upsert_nation(self.world_id, name, updated_tick=tick, **patch)

    def _entity_exists(self, kind: str, name: str) -> bool:
        if not name:
            return False
        if kind == "nation":
            return self.store.get_nation(self.world_id, name) is not None
        return self.store.find_npc_by_name(self.world_id, name) is not None

    def _in_scope(self, node: dict, x: int, y: int) -> bool:
        return node["x"] <= x < node["x"] + node["w"] and node["y"] <= y < node["y"] + node["h"]

    # ------------------------------------------------------------------ drift
    def _npc_drift(self, px: int, py: int, tick: int) -> None:
        """Tiny amount of Python-side NPC life so the world moves without LLM calls."""
        for npc in self.store.npcs_near(self.world_id, px, py, 10, limit=20):
            if hash_unit("drift", npc["id"], tick) > 0.35:
                continue
            dx, dy = 0, 0
            if npc.get("hostile"):
                dx = 1 if px > npc["x"] else (-1 if px < npc["x"] else 0)
                dy = 1 if py > npc["y"] else (-1 if py < npc["y"] else 0)
                if abs(px - npc["x"]) + abs(py - npc["y"]) <= 1:
                    continue
            else:
                if hash_unit("dx", npc["id"], tick) > 0.5:
                    dx = 1 if hash_unit("sx", npc["id"], tick) > 0.5 else -1
                else:
                    dy = 1 if hash_unit("sy", npc["id"], tick) > 0.5 else -1
            nx, ny = npc["x"] + dx, npc["y"] + dy
            tile = self.store.get_tile(self.world_id, nx, ny)
            if tile is None or tile["terrain"] in ("water", "mountain", "lava"):
                continue
            if self.store.objects_at(self.world_id, nx, ny):
                continue
            self.store.update_npc(npc["id"], x=nx, y=ny, updated_tick=tick)

    # ---------------------------------------------------------------- helpers
    def _fit_budget(self, prompt_text: str) -> str:
        if estimate_tokens(prompt_text) <= _MAX_TOKEN_TASKS:
            return prompt_text
        # crude but safe: drop the neighbour section lines until it fits
        lines = prompt_text.split("\n")
        out: List[str] = []
        for line in lines:
            out.append(line)
            if estimate_tokens("\n".join(out)) > _MAX_TOKEN_TASKS:
                out.pop()
                out.append("…（上下文过长已省略）")
                break
        return "\n".join(out)
