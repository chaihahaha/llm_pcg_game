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

    # ------------------------------------------------------------------ tick
    def tick(self) -> int:
        return int(self.store.get_meta("tick", 0) or 0)

    def advance(self, hours: int, px: int, py: int) -> List[dict]:
        """Advance world time and run every due LOD.  Returns new events."""
        if hours <= 0:
            return []
        start = self.tick()
        new_tick = start + int(hours)
        self.store.set_meta("tick", new_tick)
        if not self.enabled:
            return []
        produced: List[dict] = []
        calls = 0
        for lod, scope in self._due_scopes(new_tick, px, py):
            if calls >= self.max_calls:
                break
            try:
                events = self.evolve_scope(lod, scope, new_tick, px, py)
                produced.extend(events)
                calls += 1
            except Exception as exc:  # noqa: BLE001 - never let one scope kill the tick
                if self.logger:
                    self.logger.warning("evolution failed at lod %s node %s: %s",
                                        lod, scope.get("id"), exc)
        self._npc_drift(px, py, new_tick)
        self.store.prune_events(self.world_id)
        self.store.commit()
        return produced

    def _due_scopes(self, tick: int, px: int, py: int) -> List[Tuple[int, dict]]:
        """Scopes to evolve this advance, coarsest LOD first.

        Ordering matters: the world/region/zone steps of *this* advance must be
        applied before the local chunk step, because the chunk prompt consumes
        the higher-LOD events as its "higher digest".  A reserved local budget
        guarantees the coarse levels can never starve tile-level detail.
        """
        chunks: List[dict] = []
        cs = self.wm.chunk_size
        r = int(cfg_get(self.cfg, "evolution.neighbor_radius", 6))
        candidates = self.store.nodes_in_rect(self.world_id, LOD_CHUNK,
                                              px - cs * (r + 1), py - cs * (r + 1),
                                              px + cs * (r + 1), py + cs * (r + 1))
        candidates.sort(key=lambda n: (abs(n["x"] - px) + abs(n["y"] - py)))
        for node in candidates:
            if tick - int(node.get("updated_tick") or 0) >= self.schedule[LOD_CHUNK]:
                chunks.append(node)

        higher: List[Tuple[int, dict]] = []
        for lod in (LOD_WORLD, LOD_REGION, LOD_ZONE):
            node = self.wm.ensure_node(lod, px, py)
            if tick - int(node.get("updated_tick") or 0) >= self.schedule[lod]:
                higher.append((lod, node))

        local_budget = int(cfg_get(self.cfg, "evolution.max_chunk_scopes_per_advance", 2))
        chosen: List[Tuple[int, dict]] = [(LOD_CHUNK, c) for c in chunks[:local_budget]]
        remaining = max(0, self.max_calls - len(chosen))
        for lod, node in higher:
            if remaining <= 0:
                break
            chosen.append((lod, node))
            remaining -= 1
        for node in chunks[local_budget:]:
            if remaining <= 0:
                break
            chosen.append((LOD_CHUNK, node))
            remaining -= 1

        chosen.sort(key=lambda t: t[0])  # world -> region -> zone -> chunk
        return chosen

    # ---------------------------------------------------------------- scoping
    def evolve_scope(self, lod: int, node: dict, tick: int, px: int, py: int) -> List[dict]:
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
                                   higher_digest, neighbor_digest, hour, day, allow_tiles)
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
                    f"[{n['name']}@{n['x']},{n['y']},{n['role']},HP{n['hp']},{n['mood']}]" for n in npcs))
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
            patch: Dict[str, Any] = {}
            if ch.get("hp_delta") is not None:
                hp = max(0, int(npc["hp"]) + int(ch["hp_delta"]))
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
            if ch.get("destroyed"):
                self.store.delete_object(int(oid))
            else:
                patch = {}
                if ch.get("desc"):
                    patch["desc"] = str(ch["desc"])[:200]
                if ch.get("hp") is not None:
                    patch["hp"] = int(ch["hp"])
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
