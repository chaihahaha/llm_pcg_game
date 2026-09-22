"""Hierarchical (LOD) world generation.

The map is a tree:

    world (LOD 0)                -- cosmology, magic, tech, nations
      └ region (LOD 1, 512)      -- climate, biome mix, cultures
          └ zone (LOD 2, 128)    -- local area, notable sites, NPC seeds
              └ chunk (LOD 3, 16) -- a 16x16 tile matrix + features + NPCs
                  └ tile          -- one cell, lazily described
                      └ object    -- one entity on a tile

Only ancestors of the player's position are ever generated, and each level is
asked for a *summary* that becomes the entire context for its children.  That
is what keeps the model inside its context window: a chunk prompt never sees
more than one parent summary plus a handful of digests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from . import prompts
from .config import cfg_get
from .db import Store
from .llm import LLMClient
from .rng import fbm, hash_int, jitter, weighted_pick
from .terrain import (
    CODE_TO_TERRAIN,
    default_desc,
    is_solid,
    normalize,
    normalize_kind,
    normalize_mix,
    symbol_of,
)


def _field(data: dict, *keys, default=""):
    """First non-empty value among several plausible key spellings.

    Local models drift on key names (``world_name`` vs ``name``); being lenient
    here is far cheaper than another 45-second regeneration.
    """
    if not isinstance(data, dict):
        return default
    for k in keys:
        v = data.get(k)
        if v not in (None, "", [], {}):
            return v
    return default

LOD_WORLD, LOD_REGION, LOD_ZONE, LOD_CHUNK = 0, 1, 2, 3
LOD_NAMES = {LOD_WORLD: "world", LOD_REGION: "region", LOD_ZONE: "zone", LOD_CHUNK: "chunk"}

_OBJ_SOLID = {"rock": 1, "ruin": 1, "building": 1, "tree": 1, "arcane": 0, "plant": 0,
              "ore": 1, "water": 0, "track": 0, "item": 0, "altar": 1, "corpse": 0}


class WorldManager:
    def __init__(self, store: Store, llm: LLMClient, cfg: Dict[str, Any], logger=None,
                 world_id: Optional[int] = None):
        self.store = store
        self.llm = llm
        self.cfg = cfg
        self.logger = logger
        self.world_id = world_id
        self.region_size = int(cfg_get(cfg, "world.region_size", 512))
        self.zone_size = int(cfg_get(cfg, "world.zone_size", 128))
        self.chunk_size = int(cfg_get(cfg, "world.chunk_size", 16))
        self.max_nations = int(cfg_get(cfg, "world.max_nations", 8))
        self.max_npcs_per_chunk = int(cfg_get(cfg, "world.max_npcs_per_chunk", 3))

    # ------------------------------------------------------------------ world
    def create(self, seed: Optional[int] = None, hint: str = "") -> Dict[str, Any]:
        seed = int(seed if seed is not None else cfg_get(self.cfg, "world.seed", 1))
        msgs = prompts.build("", prompts.world_task(seed, hint))
        data = self.llm.json(msgs, task="world", max_tokens=1800, temperature=0.9, default={})

        name = str(_field(data, "name", "world_name", "title", default="无名之地"))[:24]
        era = str(_field(data, "era", "epoch", "age", default="黎明纪"))[:24]
        summary = str(_field(data, "summary", "overview", "description", default=""))
        world_data = {
            "cosmology": str(_field(data, "cosmology", "creation", "origin")),
            "magic_system": str(_field(data, "magic_system", "magic", "magic_rules")),
            "tech_baseline": str(_field(data, "tech_baseline", "tech", "technology")),
        }
        self.world_id = self.store.create_world(name, seed, era, summary, world_data)
        world_node_size = self.region_size * 8
        self.store.upsert_node(self.world_id, LOD_WORLD, 0, 0, world_node_size, world_node_size,
                               "world", name, summary, world_data, None, tick=0)

        for nation in (_field(data, "nations", "countries", "kingdoms", default=[]) or [])[: self.max_nations]:
            if not isinstance(nation, dict) or not nation.get("name"):
                continue
            self.store.upsert_nation(
                self.world_id, str(nation["name"]),
                race=str(nation.get("race", ""))[:24],
                gov=str(nation.get("gov", ""))[:24],
                tech=str(nation.get("tech", ""))[:24],
                magic=str(nation.get("magic", ""))[:24],
                capital_x=int(nation.get("capital_x", 0) or 0),
                capital_y=int(nation.get("capital_y", 0) or 0),
                resources=[str(r)[:16] for r in (nation.get("resources") or [])][:6],
                summary=str(nation.get("summary", ""))[:200],
            )
        self.store.add_event(self.world_id, 0, LOD_WORLD, None, 0, 0, "genesis",
                             f"世界「{name}」于{era}成形：" + summary[:80])
        self.store.commit()
        return self.store.get_world(self.world_id)

    def load(self, world_id: int) -> Dict[str, Any]:
        self.world_id = world_id
        w = self.store.get_world(world_id)
        if not w:
            raise ValueError(f"world {world_id} not found")
        return w

    def world_node(self) -> dict:
        node = self.store.get_node(self.world_id, LOD_WORLD, 0, 0)
        if node is None:
            raise RuntimeError("world node missing; call create() first")
        return node

    def world_bible(self, extra: str = "") -> str:
        world = self.store.get_world(self.world_id) or {}
        nations = self.store.list_nations(self.world_id)
        return prompts.world_bible(world, nations, extra)

    # ------------------------------------------------------------- coordinates
    def bbox(self, lod: int, tx: int, ty: int) -> Tuple[int, int, int]:
        size = {LOD_REGION: self.region_size, LOD_ZONE: self.zone_size,
                LOD_CHUNK: self.chunk_size, LOD_WORLD: self.region_size * 8}[lod]
        return (tx // size) * size, (ty // size) * size, size

    def ensure_node(self, lod: int, tx: int, ty: int) -> dict:
        nx, ny, size = self.bbox(lod, tx, ty)
        node = self.store.get_node(self.world_id, lod, nx, ny)
        if node:
            return node
        if lod == LOD_WORLD:
            return self.world_node()

        parent = self.ensure_node(lod - 1, nx, ny)
        tick = self.tick()
        if lod == LOD_REGION:
            name, summary, data = self._gen_region(parent, nx, ny, size)
        elif lod == LOD_ZONE:
            name, summary, data = self._gen_zone(parent, nx, ny, size)
        else:
            name, summary, data = self._gen_chunk(parent, nx, ny, size)

        self.store.upsert_node(self.world_id, lod, nx, ny, size, size, LOD_NAMES[lod],
                               name, summary, data, parent["id"], tick=tick)
        node = self.store.get_node(self.world_id, lod, nx, ny)
        self.store.add_event(self.world_id, tick, lod, node["id"], nx + size // 2, ny + size // 2,
                             "generation", f"生成{LOD_NAMES[lod]}「{name}」")
        if lod == LOD_CHUNK:
            self._materialize_chunk(node)
        self.store.commit()
        return self.store.get_node(self.world_id, lod, nx, ny)

    def chunk_at(self, tx: int, ty: int) -> dict:
        return self.ensure_node(LOD_CHUNK, tx, ty)

    def zone_at(self, tx: int, ty: int) -> dict:
        return self.ensure_node(LOD_ZONE, tx, ty)

    def region_at(self, tx: int, ty: int) -> dict:
        return self.ensure_node(LOD_REGION, tx, ty)

    def ancestors(self, tx: int, ty: int) -> List[dict]:
        return [self.ensure_node(lod, tx, ty) for lod in (LOD_REGION, LOD_ZONE, LOD_CHUNK)]

    def ensure_area(self, x0: int, y0: int, x1: int, y1: int) -> int:
        """Generate every chunk covering the rectangle.  Returns chunks created."""
        created = 0
        cs = self.chunk_size
        for cx in range((x0 // cs) * cs, x1 + 1, cs):
            for cy in range((y0 // cs) * cs, y1 + 1, cs):
                if self.store.get_node(self.world_id, LOD_CHUNK, cx, cy) is None:
                    self.ensure_node(LOD_CHUNK, cx, cy)
                    created += 1
        return created

    # ------------------------------------------------------------------ tick
    def tick(self) -> int:
        return int(self.store.get_meta("tick", 0) or 0)

    # -------------------------------------------------------- level generators
    def _gen_region(self, parent: dict, nx: int, ny: int, size: int) -> Tuple[str, str, dict]:
        seed = int(self.store.get_world(self.world_id)["seed"])
        hint = parent.get("summary", "")[:100]
        msgs = prompts.build(self.world_bible(),
                             prompts.region_task(seed, parent.get("name", ""), nx, ny, size, hint))
        data = self.llm.json(msgs, task="region", max_tokens=900, temperature=0.85, default={})
        name = str(_field(data, "name", "region_name", "title", default=f"无名区域{nx},{ny}"))[:30]
        summary = str(_field(data, "summary", "overview", default=""))
        mix = normalize_mix(_field(data, "biome_mix", "biomes", "terrain_mix", default=None))
        norm = {
            "climate": str(_field(data, "climate", "weather_pattern", default=""))[:40],
            "biome_mix": [[n, w] for n, w in mix],
            "cultures": [str(c)[:60] for c in (data.get("cultures") or [])][:5],
            "nations_present": [str(c)[:30] for c in (data.get("nations_present") or [])][:6],
            "features": self._clean_features(data.get("features"), size),
            "hazards": [str(h)[:40] for h in (data.get("hazards") or [])][:5],
        }
        return name, summary, norm

    def _gen_zone(self, parent: dict, nx: int, ny: int, size: int) -> Tuple[str, str, dict]:
        seed = int(self.store.get_world(self.world_id)["seed"])
        msgs = prompts.build(self.world_bible(), prompts.zone_task(seed, parent, nx, ny, size))
        data = self.llm.json(msgs, task="zone", max_tokens=900, temperature=0.8, default={})
        name = str(_field(data, "name", "zone_name", "title", default=f"无名之地{nx},{ny}"))[:30]
        summary = str(_field(data, "summary", "overview", default=""))
        mix_raw = _field(data, "biome_mix", "biomes", "terrain_mix", default=None)
        if not mix_raw:
            mix_raw = (parent.get("data") or {}).get("biome_mix")
        norm = {
            "biome_mix": [[n, w] for n, w in normalize_mix(mix_raw)],
            "features": self._clean_features(data.get("features"), size),
            "hazards": [str(h)[:40] for h in (data.get("hazards") or [])][:5],
            "npc_seeds": [{
                "name": str(s.get("name", ""))[:20],
                "race": str(s.get("race", ""))[:16],
                "role": str(s.get("role", ""))[:20],
                "personality": str(s.get("personality", ""))[:60],
            } for s in (data.get("npc_seeds") or [])[:3] if isinstance(s, dict)],
        }
        return name, summary, norm

    def _gen_chunk(self, parent: dict, nx: int, ny: int, size: int) -> Tuple[str, str, dict]:
        seed = int(self.store.get_world(self.world_id)["seed"])
        msgs = prompts.build(self.world_bible(), prompts.chunk_task(seed, parent, nx, ny, size))
        data = self.llm.json(msgs, task="chunk", max_tokens=1100, temperature=0.65, default={})
        name = f"地块 {nx},{ny}"
        summary = str(_field(data, "summary", "overview",
                             default=f"{parent.get('name','')}的一部分"))
        rows = self._clean_rows(data.get("rows"), seed, nx, ny, size, parent)
        norm = {
            "rows": rows,
            "weather": str(data.get("weather", ""))[:20] or "晴",
            "features": self._clean_features(data.get("features"), size),
            "npcs": [{
                "x": self._clamp_int(n.get("x"), 0, size - 1),
                "y": self._clamp_int(n.get("y"), 0, size - 1),
                "name": str(n.get("name", ""))[:20] or "无名者",
                "race": str(n.get("race", ""))[:16],
                "role": str(n.get("role", ""))[:20],
                "personality": str(n.get("personality", ""))[:60],
            } for n in (data.get("npcs") or [])[: self.max_npcs_per_chunk] if isinstance(n, dict)],
            "parent_zone": parent.get("name", ""),
        }
        return name, summary, norm

    # ------------------------------------------------------------- materialise
    def _materialize_chunk(self, node: dict) -> None:
        """Write tiles, objects and NPCs for a freshly generated chunk to the DB."""
        data = node.get("data") or {}
        rows = data.get("rows") or []
        cs = self.chunk_size
        ox, oy = node["x"], node["y"]
        tick = self.tick()
        for j in range(cs):
            row = rows[j] if j < len(rows) else ""
            for i in range(cs):
                terr = CODE_TO_TERRAIN.get(row[i], "grass") if i < len(row) else "grass"
                self.store.upsert_tile(self.world_id, ox + i, oy + j, terr,
                                       biome=normalize(terr), desc="", tick=tick)

        for feat in data.get("features", []):
            fx, fy = ox + int(feat["x"]), oy + int(feat["y"])
            kind = str(feat.get("kind", "object"))
            self.store.add_object(self.world_id, fx, fy, kind, str(feat.get("name", ""))[:30],
                                  str(feat.get("desc", ""))[:200],
                                  solid=_OBJ_SOLID.get(kind, 0), tick=tick)

        for n in data.get("npcs", []):
            self.store.add_npc(
                self.world_id, ox + int(n["x"]), oy + int(n["y"]),
                str(n.get("name", "无名者")), race=str(n.get("race", "")), role=str(n.get("role", "")),
                personality=str(n.get("personality", "")), tick=tick,
            )

    # ------------------------------------------------------------------- tiles
    def tile_at(self, x: int, y: int) -> dict:
        tile = self.store.get_tile(self.world_id, x, y)
        if tile is None:
            self.chunk_at(x, y)
            tile = self.store.get_tile(self.world_id, x, y)
        if tile is None:  # paranoid fallback
            self.store.upsert_tile(self.world_id, x, y, "grass", biome="grass")
            self.store.commit()
            tile = self.store.get_tile(self.world_id, x, y)
        return tile

    def ensure_tile_desc(self, x: int, y: int, force: bool = False) -> dict:
        tile = self.tile_at(x, y)
        if tile.get("desc") and not force:
            return tile
        chunk = self.ensure_node(LOD_CHUNK, x, y)
        zone = self.ensure_node(LOD_ZONE, x, y)
        neighbors = "、".join(
            f"{dx},{dy}:{self.tile_at(x+dx, y+dy)['terrain']}"
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)) if (dx, dy) != (0, 0)
        )
        objects = prompts.digest_objects(self.store.objects_at(self.world_id, x, y))
        seed = int(self.store.get_world(self.world_id)["seed"]) + x * 7919 + y * 104729
        msgs = prompts.build(self.world_bible(),
                             prompts.tile_task(seed, tile, zone.get("name", ""),
                                               chunk.get("summary", ""), neighbors, objects))
        data = self.llm.json(msgs, task="tile", max_tokens=400, temperature=0.7, default={})
        name = str(data.get("name") or default_desc(tile["terrain"]))
        desc = str(data.get("desc") or "")
        detail = str(data.get("detail") or "")
        data2 = dict(tile.get("data") or {})
        if detail:
            data2["detail"] = detail
        self.store.update_tile_fields(self.world_id, x, y, name=name, desc=desc, data=data2,
                                      explored=1, updated_tick=self.tick())
        self.store.commit()
        return self.store.get_tile(self.world_id, x, y)

    def mark_explored(self, x: int, y: int) -> None:
        self.store.update_tile_fields(self.world_id, x, y, explored=1, updated_tick=self.tick())

    # ----------------------------------------------------------------- helpers
    def _clean_features(self, raw, size: int) -> List[dict]:
        out: List[dict] = []
        for f in (raw or [])[:6]:
            if not isinstance(f, dict):
                continue
            out.append({
                "name": str(_field(f, "name", "title", default="无名之物"))[:30],
                "kind": normalize_kind(str(_field(f, "kind", "type", default="object"))),
                "desc": str(_field(f, "desc", "description", default=""))[:200],
                "x": self._clamp_int(f.get("x"), 0, size - 1) if f.get("x") is not None else None,
                "y": self._clamp_int(f.get("y"), 0, size - 1) if f.get("y") is not None else None,
            })
        for f in out:
            if f["x"] is None or f["y"] is None:
                f["x"] = hash_int(self.world_id, "fx", f["name"], mod=size)
                f["y"] = hash_int(self.world_id, "fy", f["name"], mod=size)
        return out

    def _clean_rows(self, raw, seed: int, nx: int, ny: int, size: int, parent: dict) -> List[str]:
        """Validate the code matrix; strip junk, pad/repair, or regenerate.

        Real models occasionally emit separators (``=.=.``), spaces, short rows,
        or degenerate repeats (``..,,..,,``).  Illegal characters are dropped,
        short rows are padded from seeded noise, and a matrix with too little
        variety is replaced wholesale by procedural terrain seeded from the
        parent's LLM-chosen biome mix.  A chunk is therefore always usable.
        """
        mix = normalize_mix((parent.get("data") or {}).get("biome_mix"))
        legal = set(CODE_TO_TERRAIN.keys())
        out: List[str] = []
        good = 0
        for j in range(size):
            row = raw[j] if isinstance(raw, list) and j < len(raw) else None
            if isinstance(row, list):
                row = "".join(str(c) for c in row)
            cleaned = "".join(ch for ch in str(row or "") if ch in legal)
            if len(cleaned) >= size:
                out.append(cleaned[:size])
                good += 1
            elif len(cleaned) >= size // 2:
                cells = list(cleaned)
                for i in range(len(cells), size):
                    cells.append(self._procedural_cell(seed, nx + i, ny + j, mix))
                out.append("".join(cells))
            else:
                out.append(self._procedural_row(seed, nx, ny, j, size, mix))

        if good == 0 or not self._rows_are_sane(out, size):
            if self.logger:
                self.logger.warning("chunk %s,%s terrain matrix degenerate -> procedural from "
                                    "parent biome mix", nx, ny)
            # keep the model's implied base biome if we can read one
            base = self._dominant_char(out)
            if base:
                mix = [(CODE_TO_TERRAIN[base], 0.55)] + [(n, w) for n, w in mix if n != CODE_TO_TERRAIN[base]]
            return [self._procedural_row(seed, nx, ny, j, size, mix) for j in range(size)]
        return out

    @staticmethod
    def _rows_are_sane(rows: List[str], size: int) -> bool:
        if not rows or len(rows) < size:
            return False
        chars = set("".join(rows))
        if len(chars) < 3:
            return False
        degenerate = 0
        half = size // 2
        for row in rows:
            if len(set(row)) <= 2 or row[:half] == row[half:]:
                degenerate += 1
        return degenerate < size // 2

    @staticmethod
    def _dominant_char(rows: List[str]) -> str:
        counts: Dict[str, int] = {}
        for row in rows:
            for ch in row:
                if ch in CODE_TO_TERRAIN:
                    counts[ch] = counts.get(ch, 0) + 1
        if not counts:
            return ""
        return max(counts.items(), key=lambda kv: kv[1])[0]

    @staticmethod
    def _procedural_cell(seed: int, wx: int, wy: int, mix: List[Tuple[str, float]]) -> str:
        n = fbm(seed, wx, wy, 24.0, octaves=3) + jitter(seed, wx, wy, 0.06)
        if n < 0.28:
            terr = "water"
        elif n < 0.40:
            terr = "sand"
        else:
            terr = weighted_pick(mix, seed, "t", wx, wy, default="grass")
        return symbol_of(terr)

    @classmethod
    def _procedural_row(cls, seed: int, nx: int, ny: int, j: int, size: int,
                        mix: List[Tuple[str, float]]) -> str:
        return "".join(cls._procedural_cell(seed, nx + i, ny + j, mix) for i in range(size))

    @staticmethod
    def _clamp_int(value, low: int, high: int) -> int:
        try:
            return max(low, min(high, int(value)))
        except (TypeError, ValueError):
            return low

    def find_spawn(self, near_x: int, near_y: int, max_radius: int = 10) -> Tuple[int, int]:
        """Find a passable tile with room to move, so the player is never boxed in."""
        def passable(x: int, y: int) -> bool:
            t = self.tile_at(x, y)
            if is_solid(t["terrain"]):
                return False
            if any(o.get("solid") for o in self.store.objects_at(self.world_id, x, y)):
                return False
            return True

        best = (near_x, near_y)
        for r in range(max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) != r:
                        continue
                    x, y = near_x + dx, near_y + dy
                    if not passable(x, y):
                        continue
                    exits = sum(1 for nx, ny in ((x+1, y), (x-1, y), (x, y+1), (x, y-1))
                                if passable(nx, ny))
                    if exits >= 2:
                        return x, y
                    best = (x, y)
        return best

    # ------------------------------------------------------------- view query
    def view_tiles(self, cx: int, cy: int, size: int) -> Dict[Tuple[int, int], dict]:
        x0, y0 = cx - size // 2, cy - size // 2
        self.ensure_area(x0, y0, x0 + size - 1, y0 + size - 1)
        tiles = self.store.list_tiles(self.world_id, x0, y0, x0 + size - 1, y0 + size - 1)
        return {(t["x"], t["y"]): t for t in tiles}
