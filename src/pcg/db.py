"""SQLite persistence layer.

Every game object the LLM ever invents — worlds, nations, LOD tree nodes,
tiles, objects, NPCs, events, dialogue, story — lives here.  The database is
the single source of truth: the LLM never holds world state in its context,
it only reads *digests* of rows we hand it and writes back deltas.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);

CREATE TABLE IF NOT EXISTS worlds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    seed INTEGER NOT NULL,
    era TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    data_json TEXT DEFAULT '{}',
    created_tick INTEGER DEFAULT 0,
    updated_tick INTEGER DEFAULT 0
);

-- The hierarchical LOD tree.  lod: 0=world 1=region 2=zone 3=chunk
CREATE TABLE IF NOT EXISTS nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    lod INTEGER NOT NULL,
    parent_id INTEGER,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    w INTEGER NOT NULL,
    h INTEGER NOT NULL,
    kind TEXT DEFAULT '',
    name TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    data_json TEXT DEFAULT '{}',
    created_tick INTEGER DEFAULT 0,
    updated_tick INTEGER DEFAULT 0,
    UNIQUE(world_id, lod, x, y)
);
CREATE INDEX IF NOT EXISTS idx_nodes_lookup ON nodes(world_id, lod, x, y);
CREATE INDEX IF NOT EXISTS idx_nodes_parent ON nodes(parent_id);

CREATE TABLE IF NOT EXISTS tiles (
    world_id INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    terrain TEXT NOT NULL DEFAULT 'grass',
    biome TEXT DEFAULT '',
    name TEXT DEFAULT '',
    desc TEXT DEFAULT '',
    explored INTEGER DEFAULT 0,
    data_json TEXT DEFAULT '{}',
    updated_tick INTEGER DEFAULT 0,
    PRIMARY KEY (world_id, x, y)
);
CREATE INDEX IF NOT EXISTS idx_tiles_rect ON tiles(world_id, x, y);

CREATE TABLE IF NOT EXISTS objects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    kind TEXT DEFAULT 'object',
    name TEXT DEFAULT '',
    desc TEXT DEFAULT '',
    solid INTEGER DEFAULT 0,
    alive INTEGER DEFAULT 1,
    hp INTEGER DEFAULT 0,
    hp_max INTEGER DEFAULT 0,
    state_json TEXT DEFAULT '{}',
    created_tick INTEGER DEFAULT 0,
    updated_tick INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_objects_pos ON objects(world_id, x, y);

CREATE TABLE IF NOT EXISTS npcs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    node_id INTEGER,
    x INTEGER NOT NULL,
    y INTEGER NOT NULL,
    name TEXT NOT NULL,
    race TEXT DEFAULT '',
    role TEXT DEFAULT '',
    faction TEXT DEFAULT '',
    personality TEXT DEFAULT '',
    appearance TEXT DEFAULT '',
    mood TEXT DEFAULT '平静',
    status TEXT DEFAULT '',
    hp INTEGER DEFAULT 10,
    hp_max INTEGER DEFAULT 10,
    atk INTEGER DEFAULT 3,
    def INTEGER DEFAULT 1,
    hostile INTEGER DEFAULT 0,
    alive INTEGER DEFAULT 1,
    state_json TEXT DEFAULT '{}',
    created_tick INTEGER DEFAULT 0,
    updated_tick INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_npcs_pos ON npcs(world_id, x, y);
CREATE INDEX IF NOT EXISTS idx_npcs_node ON npcs(node_id);

CREATE TABLE IF NOT EXISTS nations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    race TEXT DEFAULT '',
    gov TEXT DEFAULT '',
    tech TEXT DEFAULT '',
    magic TEXT DEFAULT '',
    capital_x INTEGER DEFAULT 0,
    capital_y INTEGER DEFAULT 0,
    resources_json TEXT DEFAULT '[]',
    relations_json TEXT DEFAULT '{}',
    summary TEXT DEFAULT '',
    updated_tick INTEGER DEFAULT 0,
    UNIQUE(world_id, name)
);

-- Relationship graph: nation<->nation, npc<->npc, npc<->faction (and later
-- creature/item).  Edges carry a signed value so they can decay and flip.
CREATE TABLE IF NOT EXISTS relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    a_kind TEXT NOT NULL,
    a_name TEXT NOT NULL,
    b_kind TEXT NOT NULL,
    b_name TEXT NOT NULL,
    kind TEXT DEFAULT '中立',
    value INTEGER DEFAULT 0,
    note TEXT DEFAULT '',
    updated_tick INTEGER DEFAULT 0,
    UNIQUE(world_id, a_kind, a_name, b_kind, b_name)
);
CREATE INDEX IF NOT EXISTS idx_rel_a ON relations(world_id, a_kind, a_name);
CREATE INDEX IF NOT EXISTS idx_rel_b ON relations(world_id, b_kind, b_name);

-- What an NPC knows.  This is the anti-amnesia ledger: dialogue and evolution
-- append here, and the dialogue prompt reads it back, so an NPC cannot calmly
-- contradict what it said or did three days ago.
CREATE TABLE IF NOT EXISTS npc_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    npc_id INTEGER NOT NULL,
    tick INTEGER DEFAULT 0,
    kind TEXT DEFAULT 'fact',
    about TEXT DEFAULT '',
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_npc ON npc_memory(npc_id, id);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    tick INTEGER NOT NULL,
    lod INTEGER DEFAULT 3,
    node_id INTEGER,
    x INTEGER DEFAULT 0,
    y INTEGER DEFAULT 0,
    kind TEXT DEFAULT 'event',
    summary TEXT NOT NULL,
    data_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_scope ON events(world_id, lod, tick);
CREATE INDEX IF NOT EXISTS idx_events_node ON events(node_id);

CREATE TABLE IF NOT EXISTS dialogue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    npc_id INTEGER,
    tick INTEGER DEFAULT 0,
    role TEXT DEFAULT 'npc',
    content TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_dialogue_npc ON dialogue(npc_id, id);

CREATE TABLE IF NOT EXISTS story (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    world_id INTEGER NOT NULL,
    tick INTEGER DEFAULT 0,
    title TEXT DEFAULT '',
    summary TEXT DEFAULT '',
    state TEXT DEFAULT 'active',
    data_json TEXT DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_story_world ON story(world_id, state);

CREATE TABLE IF NOT EXISTS player (
    world_id INTEGER PRIMARY KEY,
    name TEXT DEFAULT '旅人',
    x INTEGER DEFAULT 0,
    y INTEGER DEFAULT 0,
    hp INTEGER DEFAULT 30,
    hp_max INTEGER DEFAULT 30,
    atk INTEGER DEFAULT 5,
    def INTEGER DEFAULT 2,
    level INTEGER DEFAULT 1,
    xp INTEGER DEFAULT 0,
    gold INTEGER DEFAULT 0,
    state_json TEXT DEFAULT '{}',
    updated_tick INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    task TEXT DEFAULT '',
    response TEXT NOT NULL,
    created_at REAL
);
"""


def _dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _loads(text: Optional[str], default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


CACHE_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS llm_cache (
    key TEXT PRIMARY KEY,
    task TEXT DEFAULT '',
    response TEXT NOT NULL,
    created_at REAL
);
"""


class CacheStore:
    """Standalone LLM response cache so it survives world resets.

    A model call costs ~45 s on the local box; keeping the cache in its own
    file means re-rolling a world (or deleting a save) never re-pays for it.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(CACHE_SCHEMA)
        self.conn.commit()

    def cache_get(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
        return row["response"] if row else None

    def cache_put(self, key: str, task: str, response: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO llm_cache(key,task,response,created_at) VALUES(?,?,?,?)",
            (key, task, response, time.time()),
        )
        self.conn.commit()

    def cache_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) c FROM llm_cache").fetchone()["c"])

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()


class Store:
    """Thin DAO over sqlite3.  All access is synchronous and single-threaded."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        self._migrate()

    def _migrate(self) -> None:
        """Add columns introduced after a save was created (CREATE IF NOT EXISTS
        does not alter existing tables)."""
        cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(objects)")}
        if "alive" not in cols:
            self.conn.execute("ALTER TABLE objects ADD COLUMN alive INTEGER DEFAULT 1")
        npc_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(npcs)")}
        if "appearance" not in npc_cols:
            self.conn.execute("ALTER TABLE npcs ADD COLUMN appearance TEXT DEFAULT ''")
        if "status" not in npc_cols:
            self.conn.execute("ALTER TABLE npcs ADD COLUMN status TEXT DEFAULT ''")
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
        finally:
            self.conn.close()

    def commit(self) -> None:
        self.conn.commit()

    # ------------------------------------------------------------------ meta
    def get_meta(self, k: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
        return _loads(row["v"], default) if row else default

    def set_meta(self, k: str, v: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
            (k, _dumps(v)),
        )

    # ----------------------------------------------------------------- world
    def create_world(self, name: str, seed: int, era: str, summary: str, data: dict, tick: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO worlds(name,seed,era,summary,data_json,created_tick,updated_tick)"
            " VALUES(?,?,?,?,?,?,?)",
            (name, seed, era, summary, _dumps(data), tick, tick),
        )
        self.commit()
        return int(cur.lastrowid)

    def get_world(self, world_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM worlds WHERE id=?", (world_id,)).fetchone()
        return self._world(row) if row else None

    def update_world(self, world_id: int, **fields) -> None:
        allowed = {"name", "era", "summary", "updated_tick"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        if "data" in fields:
            sets.append("data_json=?")
            vals.append(_dumps(fields["data"]))
        if not sets:
            return
        vals.append(world_id)
        self.conn.execute(f"UPDATE worlds SET {', '.join(sets)} WHERE id=?", vals)

    @staticmethod
    def _world(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["data"] = _loads(d.pop("data_json", "{}"), {})
        return d

    # ------------------------------------------------------------------ nodes
    def upsert_node(
        self,
        world_id: int,
        lod: int,
        x: int,
        y: int,
        w: int,
        h: int,
        kind: str,
        name: str,
        summary: str,
        data: dict,
        parent_id: Optional[int],
        tick: int = 0,
    ) -> int:
        row = self.conn.execute(
            "SELECT id FROM nodes WHERE world_id=? AND lod=? AND x=? AND y=?",
            (world_id, lod, x, y),
        ).fetchone()
        if row:
            self.conn.execute(
                "UPDATE nodes SET name=?, summary=?, data_json=?, parent_id=COALESCE(?, parent_id),"
                " kind=?, updated_tick=? WHERE id=?",
                (name, summary, _dumps(data), parent_id, kind, tick, row["id"]),
            )
            self.commit()
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO nodes(world_id,lod,parent_id,x,y,w,h,kind,name,summary,data_json,created_tick,updated_tick)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (world_id, lod, parent_id, x, y, w, h, kind, name, summary, _dumps(data), tick, tick),
        )
        self.commit()
        return int(cur.lastrowid)

    def get_node(self, world_id: int, lod: int, x: int, y: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM nodes WHERE world_id=? AND lod=? AND x=? AND y=?",
            (world_id, lod, x, y),
        ).fetchone()
        return self._node(row) if row else None

    def get_node_by_id(self, node_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        return self._node(row) if row else None

    def update_node(self, node_id: int, summary: str | None = None, data: dict | None = None,
                    name: str | None = None, tick: int | None = None) -> None:
        sets, vals = [], []
        if summary is not None:
            sets.append("summary=?")
            vals.append(summary)
        if data is not None:
            sets.append("data_json=?")
            vals.append(_dumps(data))
        if name is not None:
            sets.append("name=?")
            vals.append(name)
        if tick is not None:
            sets.append("updated_tick=?")
            vals.append(tick)
        if not sets:
            return
        vals.append(node_id)
        self.conn.execute(f"UPDATE nodes SET {', '.join(sets)} WHERE id=?", vals)

    def children_of(self, parent_id: int) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM nodes WHERE parent_id=? ORDER BY id", (parent_id,)).fetchall()
        return [self._node(r) for r in rows]

    def nodes_in_rect(self, world_id: int, lod: int, x0: int, y0: int, x1: int, y1: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE world_id=? AND lod=? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?"
            " ORDER BY x, y",
            (world_id, lod, x0, x1, y0, y1),
        ).fetchall()
        return [self._node(r) for r in rows]

    def nodes_of_lod(self, world_id: int, lod: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM nodes WHERE world_id=? AND lod=? ORDER BY id", (world_id, lod)).fetchall()
        return [self._node(r) for r in rows]

    def count_nodes(self, world_id: int, lod: int | None = None) -> int:
        if lod is None:
            row = self.conn.execute("SELECT COUNT(*) c FROM nodes WHERE world_id=?", (world_id,)).fetchone()
        else:
            row = self.conn.execute("SELECT COUNT(*) c FROM nodes WHERE world_id=? AND lod=?",
                                    (world_id, lod)).fetchone()
        return int(row["c"])

    @staticmethod
    def _node(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["data"] = _loads(d.pop("data_json", "{}"), {})
        return d

    # ------------------------------------------------------------------ tiles
    def upsert_tile(self, world_id: int, x: int, y: int, terrain: str, biome: str = "",
                    name: str = "", desc: str = "", data: dict | None = None,
                    explored: int | None = None, tick: int = 0) -> None:
        self.conn.execute(
            "INSERT INTO tiles(world_id,x,y,terrain,biome,name,desc,explored,data_json,updated_tick)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(world_id,x,y) DO UPDATE SET terrain=excluded.terrain, biome=excluded.biome,"
            " name=CASE WHEN excluded.name<>'' THEN excluded.name ELSE tiles.name END,"
            " desc=CASE WHEN excluded.desc<>'' THEN excluded.desc ELSE tiles.desc END,"
            " data_json=excluded.data_json,"
            " explored=COALESCE(?, tiles.explored), updated_tick=excluded.updated_tick",
            (world_id, x, y, terrain, biome, name, desc, explored or 0, _dumps(data or {}), tick, explored),
        )

    def get_tile(self, world_id: int, x: int, y: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM tiles WHERE world_id=? AND x=? AND y=?", (world_id, x, y)
        ).fetchone()
        return self._tile(row) if row else None

    def list_tiles(self, world_id: int, x0: int, y0: int, x1: int, y1: int) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tiles WHERE world_id=? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?",
            (world_id, x0, x1, y0, y1),
        ).fetchall()
        return [self._tile(r) for r in rows]

    def update_tile_fields(self, world_id: int, x: int, y: int, **fields) -> None:
        allowed = {"terrain", "biome", "name", "desc", "explored", "updated_tick"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        if "data" in fields:
            sets.append("data_json=?")
            vals.append(_dumps(fields["data"]))
        if not sets:
            return
        vals += [world_id, x, y]
        self.conn.execute(f"UPDATE tiles SET {', '.join(sets)} WHERE world_id=? AND x=? AND y=?", vals)

    @staticmethod
    def _tile(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["data"] = _loads(d.pop("data_json", "{}"), {})
        return d

    # ---------------------------------------------------------------- objects
    def add_object(self, world_id: int, x: int, y: int, kind: str, name: str, desc: str = "",
                   solid: int = 0, hp: int = 0, state: dict | None = None, tick: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO objects(world_id,x,y,kind,name,desc,solid,hp,hp_max,state_json,created_tick,updated_tick)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (world_id, x, y, kind, name, desc, solid, hp, hp, _dumps(state or {}), tick, tick),
        )
        self.commit()
        return int(cur.lastrowid)

    def get_object(self, obj_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM objects WHERE id=?", (obj_id,)).fetchone()
        return self._obj(row) if row else None

    def objects_at(self, world_id: int, x: int, y: int, alive_only: bool = True) -> List[dict]:
        q = "SELECT * FROM objects WHERE world_id=? AND x=? AND y=?"
        if alive_only:
            q += " AND alive=1"
        rows = self.conn.execute(q + " ORDER BY id", (world_id, x, y)).fetchall()
        return [self._obj(r) for r in rows]

    def objects_near(self, world_id: int, x: int, y: int, r: int, limit: int = 50,
                     alive_only: bool = True) -> List[dict]:
        q = ("SELECT * FROM objects WHERE world_id=? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?")
        if alive_only:
            q += " AND alive=1"
        rows = self.conn.execute(q + " ORDER BY id LIMIT ?",
                                 (world_id, x - r, x + r, y - r, y + r, limit)).fetchall()
        return [self._obj(r) for r in rows]

    def update_object(self, obj_id: int, **fields) -> None:
        allowed = {"kind", "name", "desc", "solid", "hp", "hp_max", "updated_tick", "x", "y",
                   "alive"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        if "state" in fields:
            sets.append("state_json=?")
            vals.append(_dumps(fields["state"]))
        if not sets:
            return
        vals.append(obj_id)
        self.conn.execute(f"UPDATE objects SET {', '.join(sets)} WHERE id=?", vals)

    def delete_object(self, obj_id: int) -> None:
        self.conn.execute("DELETE FROM objects WHERE id=?", (obj_id,))

    def destroy_object(self, obj_id: int, tick: int = 0, desc: str = "") -> None:
        """Soft delete: keep the row so the history stays inspectable.

        A felled tree should leave a stump the player can still look at, not a
        hole in the record.
        """
        obj = self.get_object(obj_id)
        if not obj:
            return
        state = dict(obj.get("state") or {})
        state["destroyed"] = True
        state["destroyed_tick"] = tick
        if obj.get("desc"):
            state["last_desc"] = obj["desc"]
        self.conn.execute(
            "UPDATE objects SET alive=0, hp=0, desc=?, state_json=?, updated_tick=? WHERE id=?",
            (desc or f"（{obj['name']}的残迹）", _dumps(state), tick, obj_id),
        )

    @staticmethod
    def _obj(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["state"] = _loads(d.pop("state_json", "{}"), {})
        return d

    # ------------------------------------------------------------------- npcs
    def add_npc(self, world_id: int, x: int, y: int, name: str, race: str = "", role: str = "",
                faction: str = "", personality: str = "", appearance: str = "",
                node_id: Optional[int] = None,
                hp: int = 10, atk: int = 3, defense: int = 1, hostile: int = 0,
                state: dict | None = None, tick: int = 0) -> int:
        cur = self.conn.execute(
            "INSERT INTO npcs(world_id,node_id,x,y,name,race,role,faction,personality,appearance,"
            "mood,status,hp,hp_max,atk,def,hostile,alive,state_json,created_tick,updated_tick)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?)",
            (world_id, node_id, x, y, name, race, role, faction, personality, appearance[:80],
             "平静", "", hp, hp, atk, defense, hostile, _dumps(state or {}), tick, tick),
        )
        self.commit()
        return int(cur.lastrowid)

    def get_npc(self, npc_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM npcs WHERE id=?", (npc_id,)).fetchone()
        return self._npc(row) if row else None

    def npcs_at(self, world_id: int, x: int, y: int, alive_only: bool = True) -> List[dict]:
        q = "SELECT * FROM npcs WHERE world_id=? AND x=? AND y=?"
        if alive_only:
            q += " AND alive=1"
        rows = self.conn.execute(q + " ORDER BY id", (world_id, x, y)).fetchall()
        return [self._npc(r) for r in rows]

    def npcs_near(self, world_id: int, x: int, y: int, r: int, alive_only: bool = True,
                  limit: int = 50) -> List[dict]:
        q = "SELECT * FROM npcs WHERE world_id=? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?"
        if alive_only:
            q += " AND alive=1"
        rows = self.conn.execute(q + " ORDER BY id LIMIT ?",
                                 (world_id, x - r, x + r, y - r, y + r, limit)).fetchall()
        return [self._npc(r) for r in rows]

    def find_npc_by_name(self, world_id: int, name: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM npcs WHERE world_id=? AND name=? AND alive=1", (world_id, name)
        ).fetchone()
        return self._npc(row) if row else None

    def update_npc(self, npc_id: int, **fields) -> None:
        allowed = {"x", "y", "name", "race", "role", "faction", "personality", "appearance",
                   "mood", "status", "hp", "hp_max", "atk", "def", "hostile", "alive",
                   "updated_tick", "node_id"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        if "state" in fields:
            sets.append("state_json=?")
            vals.append(_dumps(fields["state"]))
        if not sets:
            return
        vals.append(npc_id)
        self.conn.execute(f"UPDATE npcs SET {', '.join(sets)} WHERE id=?", vals)

    def count_npcs(self, world_id: int, alive_only: bool = True) -> int:
        q = "SELECT COUNT(*) c FROM npcs WHERE world_id=?"
        if alive_only:
            q += " AND alive=1"
        return int(self.conn.execute(q, (world_id,)).fetchone()["c"])

    @staticmethod
    def _npc(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["state"] = _loads(d.pop("state_json", "{}"), {})
        return d

    # ---------------------------------------------------------------- nations
    def upsert_nation(self, world_id: int, name: str, **fields) -> int:
        row = self.conn.execute(
            "SELECT id FROM nations WHERE world_id=? AND name=?", (world_id, name)
        ).fetchone()
        if row:
            sets, vals = [], []
            for k in ("race", "gov", "tech", "magic", "capital_x", "capital_y", "summary", "updated_tick"):
                if k in fields:
                    sets.append(f"{k}=?")
                    vals.append(fields[k])
            if "resources" in fields:
                sets.append("resources_json=?")
                vals.append(_dumps(fields["resources"]))
            if "relations" in fields:
                sets.append("relations_json=?")
                vals.append(_dumps(fields["relations"]))
            if sets:
                vals.append(row["id"])
                self.conn.execute(f"UPDATE nations SET {', '.join(sets)} WHERE id=?", vals)
            return int(row["id"])
        cur = self.conn.execute(
            "INSERT INTO nations(world_id,name,race,gov,tech,magic,capital_x,capital_y,resources_json,"
            "relations_json,summary,updated_tick) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (world_id, name, fields.get("race", ""), fields.get("gov", ""), fields.get("tech", ""),
             fields.get("magic", ""), fields.get("capital_x", 0), fields.get("capital_y", 0),
             _dumps(fields.get("resources", [])), _dumps(fields.get("relations", {})),
             fields.get("summary", ""), fields.get("updated_tick", 0)),
        )
        self.commit()
        return int(cur.lastrowid)

    def list_nations(self, world_id: int) -> List[dict]:
        rows = self.conn.execute("SELECT * FROM nations WHERE world_id=? ORDER BY id", (world_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["resources"] = _loads(d.pop("resources_json", "[]"), [])
            d["relations"] = _loads(d.pop("relations_json", "{}"), {})
            out.append(d)
        return out

    def get_nation(self, world_id: int, name: str) -> Optional[dict]:
        for n in self.list_nations(world_id):
            if n["name"] == name:
                return n
        return None

    # ------------------------------------------------------------- relations
    def upsert_relation(self, world_id: int, a_kind: str, a_name: str, b_kind: str, b_name: str,
                        kind: str = "中立", value: int = 0, note: str = "",
                        tick: int = 0) -> None:
        if not a_name or not b_name or (a_kind, a_name) == (b_kind, b_name):
            return
        kind = (kind or "中立")[:16]
        value = max(-5, min(5, int(value)))
        row = self.conn.execute(
            "SELECT id, note FROM relations WHERE world_id=? AND a_kind=? AND a_name=?"
            " AND b_kind=? AND b_name=?",
            (world_id, a_kind, a_name, b_kind, b_name),
        ).fetchone()
        if row:
            merged_note = note or row["note"]
            self.conn.execute(
                "UPDATE relations SET kind=?, value=?, note=?, updated_tick=? WHERE id=?",
                (kind, value, merged_note[:160], tick, row["id"]),
            )
        else:
            self.conn.execute(
                "INSERT INTO relations(world_id,a_kind,a_name,b_kind,b_name,kind,value,note,updated_tick)"
                " VALUES(?,?,?,?,?,?,?,?,?)",
                (world_id, a_kind, a_name[:32], b_kind, b_name[:32], kind, value, note[:160], tick),
            )

    def list_relations(self, world_id: int, kind: str | None = None,
                       name: str | None = None, limit: int = 200) -> List[dict]:
        q = "SELECT * FROM relations WHERE world_id=?"
        params: List[Any] = [world_id]
        if kind:
            q += " AND (a_kind=? OR b_kind=?)"
            params += [kind, kind]
        if name:
            q += " AND (a_name=? OR b_name=?)"
            params += [name, name]
        q += " ORDER BY id LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self.conn.execute(q, params).fetchall()]

    def relations_for(self, world_id: int, ref_kind: str, ref_name: str,
                      limit: int = 20) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM relations WHERE world_id=? AND ((a_kind=? AND a_name=?)"
            " OR (b_kind=? AND b_name=?)) ORDER BY ABS(value) DESC, id LIMIT ?",
            (world_id, ref_kind, ref_name, ref_kind, ref_name, limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            # present the edge from the queried node's point of view
            if d["a_kind"] == ref_kind and d["a_name"] == ref_name:
                d["other_kind"], d["other_name"], d["outward"] = d["b_kind"], d["b_name"], True
            else:
                d["other_kind"], d["other_name"], d["outward"] = d["a_kind"], d["a_name"], False
            out.append(d)
        return out

    def count_relations(self, world_id: int) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM relations WHERE world_id=?", (world_id,)).fetchone()["c"])

    # ---------------------------------------------------------------- memory
    def add_memory(self, world_id: int, npc_id: int, tick: int, kind: str, text: str,
                   about: str = "") -> None:
        text = (text or "").strip()
        if not text or not npc_id:
            return
        self.conn.execute(
            "INSERT INTO npc_memory(world_id,npc_id,tick,kind,about,text) VALUES(?,?,?,?,?,?)",
            (world_id, npc_id, tick, (kind or "fact")[:16], (about or "")[:32], text[:200]),
        )

    def npc_memories(self, npc_id: int, limit: int = 12, kinds: Sequence[str] | None = None) -> List[dict]:
        q = "SELECT * FROM npc_memory WHERE npc_id=?"
        params: List[Any] = [npc_id]
        if kinds:
            q += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            params += list(kinds)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in reversed(self.conn.execute(q, params).fetchall())]

    def count_memories(self, world_id: int) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM npc_memory WHERE world_id=?", (world_id,)).fetchone()["c"])

    def prune_memories(self, npc_id: int, keep: int = 12) -> None:
        """Keep ``keep`` recent memories; older ones are condensed by
        ``Narrator.consolidate_memories`` before they reach this point."""
        self.conn.execute(
            "DELETE FROM npc_memory WHERE npc_id=? AND id NOT IN"
            " (SELECT id FROM npc_memory WHERE npc_id=? ORDER BY id DESC LIMIT ?)",
            (npc_id, npc_id, keep),
        )

    def memory_count(self, npc_id: int) -> int:
        return int(self.conn.execute(
            "SELECT COUNT(*) c FROM npc_memory WHERE npc_id=?", (npc_id,)).fetchone()["c"])

    # ---------------------------------------------------------------- events
    def add_event(self, world_id: int, tick: int, lod: int, node_id: Optional[int],
                  x: int, y: int, kind: str, summary: str, data: dict | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO events(world_id,tick,lod,node_id,x,y,kind,summary,data_json) VALUES(?,?,?,?,?,?,?,?,?)",
            (world_id, tick, lod, node_id, x, y, kind, summary, _dumps(data or {})),
        )
        return int(cur.lastrowid)

    def recent_events(self, world_id: int, lods: Sequence[int] | None = None, limit: int = 20,
                      node_ids: Sequence[int] | None = None, kinds: Sequence[str] | None = None) -> List[dict]:
        q = "SELECT * FROM events WHERE world_id=?"
        params: List[Any] = [world_id]
        if lods:
            q += " AND lod IN (%s)" % ",".join("?" * len(lods))
            params += list(lods)
        if node_ids:
            q += " AND node_id IN (%s)" % ",".join("?" * len(node_ids))
            params += list(node_ids)
        if kinds:
            q += " AND kind IN (%s)" % ",".join("?" * len(kinds))
            params += list(kinds)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(q, params).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = _loads(d.pop("data_json", "{}"), {})
            out.append(d)
        return out

    def events_mentioning(self, world_id: int, name: str, limit: int = 6) -> List[dict]:
        if not name:
            return []
        rows = self.conn.execute(
            "SELECT * FROM events WHERE world_id=? AND summary LIKE ? ORDER BY id DESC LIMIT ?",
            (world_id, f"%{name}%", limit),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["data"] = _loads(d.pop("data_json", "{}"), {})
            out.append(d)
        return list(reversed(out))

    def events_for_player(self, world_id: int, x: int, y: int, r: int, limit: int = 30) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE world_id=? AND x BETWEEN ? AND ? AND y BETWEEN ? AND ?"
            " ORDER BY tick DESC, id DESC LIMIT ?",
            (world_id, x - r, x + r, y - r, y + r, limit),
        ).fetchall()
        out = []
        for r2 in rows:
            d = dict(r2)
            d["data"] = _loads(d.pop("data_json", "{}"), {})
            out.append(d)
        return out

    def prune_events(self, world_id: int, keep: int = 4000) -> None:
        row = self.conn.execute("SELECT COUNT(*) c FROM events WHERE world_id=?", (world_id,)).fetchone()
        if row and row["c"] > keep:
            self.conn.execute(
                "DELETE FROM events WHERE world_id=? AND id NOT IN"
                " (SELECT id FROM events WHERE world_id=? ORDER BY id DESC LIMIT ?)",
                (world_id, world_id, keep),
            )

    # -------------------------------------------------------------- dialogue
    def add_dialogue(self, world_id: int, npc_id: Optional[int], tick: int, role: str, content: str) -> None:
        self.conn.execute(
            "INSERT INTO dialogue(world_id,npc_id,tick,role,content) VALUES(?,?,?,?,?)",
            (world_id, npc_id, tick, role, content),
        )

    def recent_dialogue(self, npc_id: int, limit: int = 8) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM dialogue WHERE npc_id=? ORDER BY id DESC LIMIT ?", (npc_id, limit)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def prune_dialogue(self, world_id: int, keep_per_npc: int = 40) -> None:
        """Per-NPC window, so a chatty NPC cannot erase another one's history."""
        self.conn.execute(
            "DELETE FROM dialogue WHERE world_id=? AND id NOT IN ("
            "  SELECT id FROM ("
            "    SELECT id, ROW_NUMBER() OVER (PARTITION BY npc_id ORDER BY id DESC) rn"
            "    FROM dialogue WHERE world_id=?"
            "  ) WHERE rn <= ?"
            ")",
            (world_id, world_id, keep_per_npc),
        )

    # ----------------------------------------------------------------- story
    def set_story(self, world_id: int, tick: int, title: str, summary: str,
                  state: str = "active", data: dict | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO story(world_id,tick,title,summary,state,data_json) VALUES(?,?,?,?,?,?)",
            (world_id, tick, title, summary, state, _dumps(data or {})),
        )
        self.commit()
        return int(cur.lastrowid)

    def current_story(self, world_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM story WHERE world_id=? AND state='active' ORDER BY id DESC LIMIT 1", (world_id,)
        ).fetchone()
        if not row:
            return None
        d = dict(row)
        d["data"] = _loads(d.pop("data_json", "{}"), {})
        return d

    def story_history(self, world_id: int, limit: int = 10) -> List[dict]:
        rows = self.conn.execute(
            "SELECT * FROM story WHERE world_id=? ORDER BY id DESC LIMIT ?", (world_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    def resolve_story(self, story_id: int, state: str = "done") -> None:
        self.conn.execute("UPDATE story SET state=? WHERE id=?", (state, story_id))

    # ---------------------------------------------------------------- player
    def create_player(self, world_id: int, name: str, x: int, y: int, hp: int, atk: int, defense: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO player(world_id,name,x,y,hp,hp_max,atk,def,level,xp,gold,state_json,updated_tick)"
            " VALUES(?,?,?,?,?,?,?,?,1,0,0,'{}',0)",
            (world_id, name, x, y, hp, hp, atk, defense),
        )
        self.commit()

    def get_player(self, world_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM player WHERE world_id=?", (world_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["state"] = _loads(d.pop("state_json", "{}"), {})
        return d

    def update_player(self, world_id: int, **fields) -> None:
        allowed = {"name", "x", "y", "hp", "hp_max", "atk", "def", "level", "xp", "gold", "updated_tick"}
        sets, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                sets.append(f"{k}=?")
                vals.append(v)
        if "state" in fields:
            sets.append("state_json=?")
            vals.append(_dumps(fields["state"]))
        if not sets:
            return
        vals.append(world_id)
        self.conn.execute(f"UPDATE player SET {', '.join(sets)} WHERE world_id=?", vals)

    # ------------------------------------------------------------- llm cache
    def cache_get(self, key: str) -> Optional[str]:
        row = self.conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
        return row["response"] if row else None

    def cache_put(self, key: str, task: str, response: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO llm_cache(key,task,response,created_at) VALUES(?,?,?,?)",
            (key, task, response, time.time()),
        )

    def cache_count(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) c FROM llm_cache").fetchone()["c"])

    # ------------------------------------------------------------ statistics
    def stats(self, world_id: int) -> Dict[str, int]:
        q = self.conn
        return {
            "nodes": q.execute("SELECT COUNT(*) c FROM nodes WHERE world_id=?", (world_id,)).fetchone()["c"],
            "tiles": q.execute("SELECT COUNT(*) c FROM tiles WHERE world_id=?", (world_id,)).fetchone()["c"],
            "objects": q.execute("SELECT COUNT(*) c FROM objects WHERE world_id=?", (world_id,)).fetchone()["c"],
            "npcs": q.execute("SELECT COUNT(*) c FROM npcs WHERE world_id=? AND alive=1", (world_id,)).fetchone()["c"],
            "events": q.execute("SELECT COUNT(*) c FROM events WHERE world_id=?", (world_id,)).fetchone()["c"],
            "nations": q.execute("SELECT COUNT(*) c FROM nations WHERE world_id=?", (world_id,)).fetchone()["c"],
            "llm_cache": q.execute("SELECT COUNT(*) c FROM llm_cache").fetchone()["c"],
        }
