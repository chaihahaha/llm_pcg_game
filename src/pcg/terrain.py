"""Canonical terrain vocabulary shared by every LOD.

The LLM is always asked to pick from this fixed vocabulary (single-char codes)
so responses stay tiny and always parseable.  Anything unrecognised is folded
back to ``grass`` and raw terrain is regenerated from seeded noise.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# name -> (symbol shown in the 16x16 view, blocks movement, default description)
TERRAIN: Dict[str, Tuple[str, bool, str]] = {
    "water":      ("~", True,  "水"),
    "shallow":    ("-", False, "浅滩"),
    "grass":      (".", False, "草甸"),
    "tall_grass": (",", False, "高草"),
    "forest":     ("T", True,  "树林"),
    "hill":       ("^", False, "缓坡"),
    "mountain":   ("#", True,  "山岩"),
    "sand":       ("s", False, "沙地"),
    "swamp":      ("m", False, "泥沼"),
    "farm":       ("f", False, "农田"),
    "road":       ("R", False, "道路"),
    "ruins":      ("+", False, "遗迹"),
    "arcane":     ("%", False, "魔力渗漏"),
    "lava":       ("!", True,  "熔岩"),
    "snow":       ("*", False, "积雪"),
    "ice":        ("_", False, "冰面"),
    "cave":       ("O", True,  "洞穴"),
    "building":   ("b", True,  "建筑"),
}

CODE_TO_TERRAIN: Dict[str, str] = {v[0]: k for k, v in TERRAIN.items()}

ALIASES: Dict[str, str] = {
    "water": "water", "lake": "water", "river": "water", "sea": "water", "ocean": "water",
    "pond": "water", "stream": "water",
    "shallow": "shallow", "shallows": "shallow", "ford": "shallow", "shore": "shallow",
    "grass": "grass", "grassland": "grass", "meadow": "grass", "plain": "grass", "prairie": "grass",
    "tall_grass": "tall_grass", "reeds": "tall_grass", "scrub": "tall_grass", "bush": "tall_grass",
    "forest": "forest", "woods": "forest", "wood": "forest", "tree": "forest", "grove": "forest",
    "jungle": "forest",
    "hill": "hill", "hills": "hill", "ridge": "hill", "slope": "hill",
    "mountain": "mountain", "mountains": "mountain", "rock": "mountain", "cliff": "mountain",
    "peak": "mountain", "stone": "mountain", "boulder": "mountain",
    "sand": "sand", "desert": "sand", "dune": "sand", "beach": "sand",
    "swamp": "swamp", "marsh": "swamp", "bog": "swamp", "wetland": "swamp", "mud": "swamp",
    "farm": "farm", "field": "farm", "cropland": "farm", "orchard": "farm", "ranch": "farm",
    "road": "road", "path": "road", "trail": "road", "street": "road", "bridge": "road",
    "ruins": "ruins", "ruin": "ruins", "wreck": "ruins", "rubble": "ruins", "ancient": "ruins",
    "arcane": "arcane", "magic": "arcane", "mana": "arcane", "ley": "arcane", "crystal": "arcane",
    "lava": "lava", "magma": "lava", "volcanic": "lava", "volcano": "lava",
    "snow": "snow", "tundra": "snow", "frost": "snow", "glacier": "ice",
    "ice": "ice", "frozen": "ice",
    "cave": "cave", "cavern": "cave", "tunnel": "cave", "mine": "cave",
    "building": "building", "house": "building", "village": "building", "town": "building",
    "city": "building", "tower": "building", "hut": "building", "camp": "building",
}

KIND_ALIASES: Dict[str, str] = {
    "rock": ["rock", "stone", "boulder", "岩石", "巨石", "山石"],
    "ore": ["ore", "矿脉", "矿石", "矿物"],
    "ruin": ["ruin", "ruins", "wreck", "废墟", "遗迹", "残骸", "遗址"],
    "plant": ["plant", "tree", "vegetation", "植物", "树木", "作物", "林木"],
    "arcane": ["arcane", "magic", "mana", "crystal", "魔力", "魔法", "符文", "水晶", "裂隙"],
    "water": ["water", "spring", "well", "water_source", "泉", "温泉", "湖", "水"],
    "building": ["building", "house", "tower", "hut", "camp", "建筑", "房屋", "塔", "营地", "井"],
    "lava": ["lava", "magma", "volcano", "熔岩", "岩浆", "火山"],
    "item": ["item", "loot", "物品", "遗物"],
    "altar": ["altar", "shrine", "祭坛", "神龛"],
    "creature": ["creature", "beast", "monster", "生物", "野兽"],
    "track": ["track", "trail", "痕迹", "足迹"],
}


def normalize_kind(name: str, fallback: str = "object") -> str:
    if not name:
        return fallback
    key = str(name).strip().lower()
    for canon, aliases in KIND_ALIASES.items():
        for a in aliases:
            if a in key:
                return canon
    return fallback


DEFAULT_MIX: List[Tuple[str, float]] = [
    ("grass", 0.34), ("tall_grass", 0.14), ("forest", 0.16), ("hill", 0.12),
    ("water", 0.07), ("farm", 0.06), ("road", 0.03), ("ruins", 0.05),
    ("sand", 0.02), ("swamp", 0.01),
]


def normalize(name: str, fallback: str = "grass") -> str:
    if not name:
        return fallback
    raw = str(name).strip()
    # models often answer with the single-character code ("m", "#") or with a
    # code+name pair ("m swamp"), so try the code first.
    if raw in CODE_TO_TERRAIN:
        return CODE_TO_TERRAIN[raw]
    key = raw.lower().replace(" ", "_").replace("-", "_")
    if key in TERRAIN:
        return key
    if key in ALIASES:
        return ALIASES[key]
    for char, canon in CODE_TO_TERRAIN.items():
        if raw.startswith(char + " ") or raw.startswith(char + "/"):
            return canon
    for alias, canon in ALIASES.items():
        if alias in key:
            return canon
    return fallback


def symbol_of(terrain: str) -> str:
    return TERRAIN.get(terrain, TERRAIN["grass"])[0]


def is_solid(terrain: str) -> bool:
    return TERRAIN.get(terrain, TERRAIN["grass"])[1]


def default_desc(terrain: str) -> str:
    return TERRAIN.get(terrain, TERRAIN["grass"])[2]


def legend_text(sep: str = " | ") -> str:
    """Unambiguous code legend: ``<char> <name>`` pairs.

    ``sym=name`` was ambiguous (models emitted literal ``=`` in the matrix), so
    the symbol and its name are separated by whitespace and pairs by ``|``.
    """
    return sep.join(f"{v[0]} {k}" for k, v in TERRAIN.items())


def valid_code(ch: str) -> str:
    return CODE_TO_TERRAIN.get(ch, "")


def normalize_mix(raw, fallback: List[Tuple[str, float]] | None = None) -> List[Tuple[str, float]]:
    """Accept [["grass",0.4],...] or {"grass":0.4} and normalise names."""
    out: List[Tuple[str, float]] = []
    if isinstance(raw, dict):
        items = list(raw.items())
    elif isinstance(raw, list):
        items = []
        for entry in raw:
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                items.append((entry[0], entry[1]))
            elif isinstance(entry, str):
                items.append((entry, 1.0))
    else:
        items = []
    for name, weight in items:
        canon = normalize(str(name))
        try:
            w = float(weight)
        except (TypeError, ValueError):
            w = 0.0
        if w > 0:
            out.append((canon, w))
    return out if out else list(fallback or DEFAULT_MIX)
