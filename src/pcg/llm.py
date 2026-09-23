"""OpenAI-compatible LLM client.

Design notes
------------
* No third-party deps — stdlib ``urllib`` only, so the game runs anywhere.
* Every call is keyed and cached in the SQLite ``llm_cache`` table.  Re-running
  the same world costs zero LLM calls, and a slow local model (the test server
  runs ~14 tok/s) is only ever asked once per unique prompt.
* A deterministic ``MockBackend`` mirrors every task so the game is fully
  playable and unit-testable with no server at all.
* Message layout is frozen-prefix friendly (stable system prompt first, then a
  per-world "bible", then the dynamic task) so llama.cpp prompt caching kicks in.
"""
from __future__ import annotations

import hashlib
import json
import re
import socket
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .config import cfg_get
from .rng import hash_int, pick
from .tokens import estimate_messages, estimate_tokens

try:  # optional, only for nicer logs
    from . import __version__  # noqa: F401
except Exception:  # pragma: no cover
    pass


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- json repair

_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.S)


def _clean_keys(obj: Any) -> Any:
    """Strip whitespace that models sometimes emit inside JSON object keys.

    Observed in the wild: ``{"  name": "酸雾沉溺巷"}``.  Keys are the one place
    whitespace is never meaningful, so normalising here fixes every consumer at
    once instead of patching each field lookup.
    """
    if isinstance(obj, dict):
        return {str(k).strip(): _clean_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_keys(v) for v in obj]
    return obj


def parse_json_loose(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response.

    Handles: code fences, leading/trailing prose, trailing commas, and the
    common ``{'a': 1}`` single-quote style.
    """
    if text is None:
        return None
    raw = text.strip()
    if not raw:
        return None
    m = _FENCE.search(raw)
    if m:
        raw = m.group(1).strip()

    candidates = [raw]
    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            candidates.append(raw[start:end + 1])

    for cand in candidates:
        for attempt in (cand,
                        re.sub(r",\s*([}\]])", r"\1", cand),
                        re.sub(r",\s*([}\]])", r"\1", cand).replace("'", '"')):
            try:
                return _clean_keys(json.loads(attempt))
            except (ValueError, TypeError):
                continue
    return None


# --------------------------------------------------------------------------- transport

class _HTTPBackend:
    def __init__(self, cfg: Dict[str, Any]):
        self.base_url = str(cfg_get(cfg, "llm.base_url", "http://127.0.0.1:8080")).rstrip("/")
        self.model = cfg_get(cfg, "llm.model", "local")
        self.api_key = cfg_get(cfg, "llm.api_key", "sk-local")
        self.timeout = int(cfg_get(cfg, "llm.timeout", 300))
        self.enable_thinking = bool(cfg_get(cfg, "llm.enable_thinking", False))

    def chat(self, messages: List[dict], max_tokens: int, temperature: float,
             json_mode: bool) -> tuple[str, str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if not choices:
            raise LLMError(f"empty choices: {str(body)[:300]}")
        msg = choices[0].get("message") or {}
        content = msg.get("content") or ""
        if not content.strip() and msg.get("reasoning_content"):
            content = msg["reasoning_content"]
        return content, str(choices[0].get("finish_reason") or "")

    def ping(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/v1/models",
                                         headers={"Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False


# --------------------------------------------------------------------------- mock

_TERRAINS = ["grass", "tall_grass", "forest", "hill", "mountain", "water", "sand",
             "swamp", "farm", "road", "ruins", "arcane", "snow", "lava"]
_MOCK_RACES = ["人族", "精灵", "矮人", "兽人", "龙裔", "灰烬族", "海民", "沙隐族"]
_MOCK_ROLES = ["铁匠", "游侠", "学者", "祭司", "商贩", "卫兵", "草药师", "吟游诗人"]
_MOCK_SURNAME = ["", "·银叶", "·铁须", "·砂歌", "·星语", "·潮生"]
_MOCK_GIVEN = ["阿岚", "卡尔", "薇拉", "拓海", "砾岩", "暮鸦", "潮音", "灰烬", "绿霭", "白岩"]


class MockBackend:
    """Deterministic stand-in for the LLM.  Never touches the network."""

    def chat(self, messages: List[dict], max_tokens: int, temperature: float,
             json_mode: bool) -> tuple[str, str]:
        task = self._task(messages)
        seed = self._seed(messages)
        return json.dumps(self._gen(task, messages, seed), ensure_ascii=False), "stop"

    def ping(self) -> bool:
        return True

    # -- helpers
    @staticmethod
    def _text(messages: List[dict]) -> str:
        return "\n".join(str(m.get("content", "")) for m in messages)

    def _task(self, messages: List[dict]) -> str:
        m = re.search(r"\[\[TASK:([a-zA-Z_]+)\]\]", self._text(messages))
        return m.group(1) if m else "generic"

    def _seed(self, messages: List[dict]) -> int:
        m = re.search(r"\[\[SEED:(-?\d+)\]\]", self._text(messages))
        return int(m.group(1)) if m else 0

    def _field(self, messages: List[dict], name: str, default: str = "") -> str:
        m = re.search(rf"\[\[{name}:(.*?)\]\]", self._text(messages))
        return m.group(1).strip() if m else default

    def _scope(self, messages: List[dict]) -> tuple[int, int, int, int]:
        """Parse the [[SCOPE:x,y,w,h]] marker so mock deltas land in world coords."""
        m = re.search(r"\[\[SCOPE:(-?\d+),(-?\d+),(\d+),(\d+)\]\]", self._text(messages))
        if not m:
            return 0, 0, 16, 16
        return tuple(int(g) for g in m.groups())  # type: ignore[return-value]

    def _npc_names(self, messages: List[dict]) -> List[str]:
        return re.findall(r"\[([^\]@,]+)@-?\d+,-?\d+,", self._text(messages))

    def _gen(self, task: str, messages: List[dict], seed: int) -> dict:
        fn = getattr(self, f"_t_{task}", None)
        if fn is None:
            return {"ok": True, "note": f"mock:{task}"}
        return fn(messages, seed)

    # -- world tree
    def _t_world(self, messages, seed) -> dict:
        names = ["翠环界", "双月之境", "灰烬大陆", "潮汐环带"]
        magics = ["元素铭文：以矿物为媒介驱动元素", "魂潮：情绪可凝结为可塑能量",
                  "星轨术：天象决定法术强度"]
        nations = []
        for i in range(3):
            nations.append({
                "name": f"{_MOCK_GIVEN[i]}王国",
                "race": pick(_MOCK_RACES, seed, "race", i),
                "gov": pick(["君主制", "长老议会", "商团共和", "神权"], seed, "gov", i),
                "tech": pick(["青铜", "铁器", "蒸汽萌芽", "符文机械"], seed, "tech", i),
                "magic": pick(["低魔", "中魔", "高魔"], seed, "magic", i),
                "resources": ["铁矿", "灵木", "盐"],
                "summary": f"以{pick(['农耕','矿业','海运'], seed, 'eco', i)}为立国之本。",
                "capital_x": 100 + i * 160,
                "capital_y": 90 + i * 140,
            })
        relations = []
        for i in range(len(nations)):
            j = (i + 1) % len(nations)
            relations.append({
                "a": nations[i]["name"], "b": nations[j]["name"],
                "kind": pick(["敌对", "盟友", "贸易", "冷战", "世仇"], seed, "rel", i),
                "value": hash_int(seed, "v", i, mod=11) - 5,
                "note": f"围绕{pick(['矿脉','水源','商路','圣地'], seed, 'rnote', i)}的长期争端。",
            })
        return {
            "name": pick(names, seed, "wname"),
            "era": pick(["青铜纪", "铁火纪", "星陨纪"], seed, "era"),
            "relations": relations,
            "cosmology": "世界由三枚沉眠的星核支撑，星核呼吸形成季节与魔力潮汐。",
            "magic_system": pick(magics, seed, "magic"),
            "tech_baseline": "铁器与水力机械并存，魔导装置稀有。",
            "summary": "大陆由数个彼此猜忌的王国分割，边境遍布古代遗迹与失控的魔力节点。",
            "nations": nations,
        }

    def _t_region(self, messages, seed) -> dict:
        return {
            "name": f"{pick(['青岚','赤砂','寒霄','暮雾','碧潮','黑曜'], seed, 'rn')}之地",
            "climate": pick(["温带湿润", "干旱多风", "寒冷漫长", "季风交替"], seed, "cl"),
            "biome_mix": [["grass", 0.30], ["forest", 0.25], ["hill", 0.18],
                          ["farm", 0.12], ["water", 0.09], ["ruins", 0.06]],
            "cultures": ["以季节祭典维系秩序", "崇尚契约与记账", "口传史诗的游牧群体"],
            "nations_present": ["（自治领地）"],
            "features": [
                {"name": "断裂星核", "kind": "arcane", "desc": "半埋于地表的晶体，夜间渗出发光雾气。"},
                {"name": "旧关隘", "kind": "ruins", "desc": "被藤蔓吞没的古代通行税卡。"},
            ],
            "hazards": ["魔力风暴", "盗匪"],
            "summary": "一片被低矮丘陵与零散林地切碎的农牧混合区，遗迹散布。",
        }

    def _t_zone(self, messages, seed) -> dict:
        return {
            "name": f"{pick(['芦荡','石岬','麦野','鸦林','盐碱','雾谷'], seed, 'zn')}",
            "biome_mix": [["grass", 0.34], ["tall_grass", 0.16], ["forest", 0.18],
                          ["hill", 0.12], ["water", 0.08], ["farm", 0.07], ["ruins", 0.05]],
            "features": [
                {"name": "半塌的守望塔", "kind": "ruins", "desc": "石砌塔身只剩三层，内有旧火痕。"},
                {"name": "浅溪渡口", "kind": "water", "desc": "供旅人涉水的浅滩，卵石密布。"},
            ],
            "hazards": ["野狼群", "夜间的魔力低语"],
            "npc_seeds": [
                {"name": pick(_MOCK_GIVEN, seed, "n1"), "race": "人族", "role": "草药师",
                 "personality": "谨慎、健谈，讨厌雨"},
                {"name": pick(_MOCK_GIVEN, seed, "n2"), "race": "矮人", "role": "铁匠",
                 "personality": "暴躁但守信"},
            ],
            "summary": "缓坡草地与零散林地交错，一条浅溪穿行其间，有废弃遗迹。",
        }

    def _t_chunk(self, messages, seed) -> dict:
        biome = self._field(messages, "BIOME", "grass")
        return {
            "summary": "缓坡与疏林交错，可见旧石堆。",
            "weather": pick(["晴，微风", "阴，湿冷", "小雨", "薄雾", "闷热"], seed, "w"),
            "patches": [
                {"terrain": biome, "x": 0, "y": 0, "w": 16, "h": 16},
                {"terrain": "water", "x": hash_int(seed, "p1", mod=6) + 1,
                 "y": hash_int(seed, "p2", mod=6) + 1, "w": 5, "h": 3},
                {"terrain": "forest", "x": hash_int(seed, "p3", mod=8) + 4,
                 "y": hash_int(seed, "p4", mod=8) + 4, "w": 6, "h": 6},
                {"terrain": "ruins", "x": 11, "y": 2, "w": 4, "h": 4},
            ],
            "features": [
                {"x": 3 + hash_int(seed, "f1", mod=10), "y": 5, "kind": "rock",
                 "name": "裂纹巨岩", "desc": "表面有细密裂纹，敲击时发出空洞回声。"},
                {"x": 11, "y": 9, "kind": "ruin", "name": "倒伏的石碑",
                 "desc": "字迹被苔藓覆盖，依稀可辨一个古老的族徽。"},
            ],
            "npcs": [
                {"x": 6, "y": 6, "name": pick(_MOCK_GIVEN, seed, "cn") + pick(_MOCK_SURNAME, seed, "cs"),
                 "race": pick(_MOCK_RACES, seed, "cr"), "role": pick(_MOCK_ROLES, seed, "crole"),
                 "personality": "多疑，但提到古代遗迹时话会变多。",
                 "appearance": "左手戴着褪色的铜护腕，右眼有一道白色旧疤。"},
                {"x": 9, "y": 11, "name": pick(_MOCK_GIVEN, seed, "cn2") + pick(_MOCK_SURNAME, seed, "cs2"),
                 "race": pick(_MOCK_RACES, seed, "cr2"), "role": pick(_MOCK_ROLES, seed, "crole2"),
                 "personality": "寡言，习惯先观察再开口。",
                 "appearance": "总披着沾满灰的羊毛斗篷，右脚微跛。"},
            ],
            "relations": [
                {"a": pick(_MOCK_GIVEN, seed, "cn") + pick(_MOCK_SURNAME, seed, "cs"),
                 "b": pick(_MOCK_GIVEN, seed, "cn2") + pick(_MOCK_SURNAME, seed, "cs2"),
                 "kind": "雇主", "value": 2, "note": "一人雇另一人看守遗迹入口。"},
            ],
        }

    def _t_tile(self, messages, seed) -> dict:
        terrain = self._field(messages, "TERRAIN", "grass")
        names = {"forest": "密林", "water": "浅水", "hill": "缓坡", "mountain": "裸岩",
                 "ruins": "残垣", "farm": "田垄", "road": "旧道", "swamp": "泥沼",
                 "sand": "沙地", "tall_grass": "高草", "grass": "草甸", "arcane": "魔力渗漏点"}
        label = names.get(terrain, terrain)
        return {
            "name": label,
            "desc": f"一片{label}。土壤{['干燥','微湿','松软'][hash_int(seed,'s',mod=3)]}，"
                    f"风带来淡淡的{['草腥','土腥','铁锈','花香'][hash_int(seed,'d',mod=4)]}味。",
            "detail": "细看可见零星的昆虫足迹与旧年的枯茎。",
        }

    # -- evolution
    def _t_evolve_chunk(self, messages, seed) -> dict:
        ox, oy, w, h = self._scope(messages)
        names = self._npc_names(messages)
        mx = ox + (w // 2 if w > 2 else 0)
        my = oy + (h // 2 if h > 2 else 0)
        changes: List[dict] = [
            {"type": "new_object", "x": mx, "y": my, "kind": "track", "name": "兽径泥坑",
             "desc": "新鲜翻起的泥土，边缘有蹄印。"},
        ]
        if names:
            changes.append({"type": "memory", "name": names[0], "kind": "goal",
                            "text": "我决定盯住那条新出现的兽径。"})
            changes.append({"type": "npc", "name": names[0], "mood": "警觉",
                            "status": "在兽径旁蹲守", "note": "发现新鲜蹄印后决定留下观察。"})
        if len(names) >= 2:
            changes.append({"type": "relation", "a_kind": "npc", "a_name": names[0],
                            "b_kind": "npc", "b_name": names[1], "kind": "敌对",
                            "value": -2, "note": "为争夺兽径上的猎物起了争执。"})
        return {
            "summary": "疏林边缘出现新的兽径，遗留下被翻动的土。",
            "weather": pick(["转阴", "落雨", "放晴", "起风"], seed, "w2"),
            "events": [{"kind": "wildlife", "text": "一头野猪在草甸上翻掘根部，留下泥坑。"}],
            "changes": changes,
        }

    def _t_evolve_zone(self, messages, seed) -> dict:
        return {
            "summary": "渡口的行人增多，遗迹附近出现拾荒者。",
            "events": [{"kind": "economy", "text": "商队沿浅溪渡口扎营，短暂带动了本地交换。"}],
            "changes": [],
        }

    def _t_evolve_region(self, messages, seed) -> dict:
        return {
            "summary": "区域内的铁矿开采加速，边境摩擦升温。",
            "events": [{"kind": "politics", "text": "两国就矿脉归属互相递交了措辞强硬的文书。"}],
            "changes": [{"type": "nation", "name": "", "set": {"tech": "铁器普及"}}],
        }

    def _t_evolve_world(self, messages, seed) -> dict:
        return {
            "summary": "魔力潮汐进入丰期，各地遗迹活动增强。",
            "era": "星陨纪·魔力丰期",
            "events": [{"kind": "magic", "text": "全球魔力潮汐上升，遗迹封印普遍松动。"}],
            "changes": [],
        }

    # -- narrative
    def _t_dialogue(self, messages, seed) -> dict:
        npc = self._field(messages, "NPC", "陌生人")
        return {
            "reply": f"（{npc}压低声音）你也是为那处遗迹来的？前几日夜里，塔顶有光。"
                     "别走溪北的草甸，那里有东西在翻土。",
            "mood": "警惕",
            "action": "none",
            "memories": [{"kind": "fact", "about": "守望塔",
                          "text": "夜里塔顶有光，我亲眼见过两次。"}],
            "relation_changes": [{"kind": "相识", "value": 1,
                                  "note": "这个旅人愿意听我把话说完。"}],
        }

    def _t_story(self, messages, seed) -> dict:
        return {
            "title": "塔顶的光",
            "summary": "玩家听闻半塌的守望塔在夜间发光，可能有未被记录的魔力节点。",
            "objective": "前往守望塔附近（遗迹格）勘察并在夜间观察。",
            "stakes": "若放任不管，魔力渗漏可能吸引危险生物。",
        }

    def _t_flavor(self, messages, seed) -> dict:
        return {"text": pick(["你的攻击擦过它的肩，溅起一线尘土。",
                              "钝响之后，它踉跄半步，獠牙上挂着草屑。",
                              "对方闷哼一声，退到了倒木后面。"], seed, "fl")}


# --------------------------------------------------------------------------- client

class LLMClient:
    def __init__(self, cfg: Dict[str, Any], store=None, logger=None):
        self.cfg = cfg
        self.store = store
        self.logger = logger
        self.mock = bool(cfg_get(cfg, "llm.mock", False))
        self.auto_fallback = bool(cfg_get(cfg, "llm.auto_fallback_mock", True))
        self.use_cache = bool(cfg_get(cfg, "llm.cache", True))
        self.default_max = int(cfg_get(cfg, "llm.max_tokens_out", 768))
        self.retries = int(cfg_get(cfg, "llm.retries", 2))
        self.degraded = False
        self.backend_name = "mock" if self.mock else "http"
        self.backend = MockBackend() if self.mock else _HTTPBackend(cfg)
        self.stats = {"calls": 0, "cache_hits": 0, "mock_calls": 0, "errors": 0,
                      "prompt_tokens": 0, "completion_tokens": 0}

    # -- public
    def ping(self) -> bool:
        try:
            return bool(self.backend.ping())
        except Exception:
            return False

    def chat(self, messages: List[dict], task: str = "generic", max_tokens: Optional[int] = None,
             temperature: Optional[float] = None, json_mode: bool = True) -> str:
        max_tokens = int(max_tokens or self.default_max)
        temperature = float(self.cfg.get("llm", {}).get("temperature", 0.85) if temperature is None else temperature)

        key = self._key(messages, task, max_tokens, temperature)
        if self.use_cache and self.store is not None:
            hit = self.store.cache_get(key)
            if hit is not None:
                self.stats["cache_hits"] += 1
                if self.logger:
                    self.logger.info("llm✓ %s (cache hit)", task)
                return hit

        if self.logger:
            self.logger.info("llm→ %s (≈%d tok in, max %d out)", task,
                             estimate_messages(messages), max_tokens)
        started = time.perf_counter()
        text, finish = self._invoke(messages, max_tokens, temperature, json_mode)
        # A truncated JSON object is worse than useless: spend one more call on a
        # bigger budget, but only if it actually buys us a parseable result.
        if json_mode and finish == "length" and max_tokens < 4000:
            bigger = min(4000, max(1200, max_tokens * 3))
            if self.logger:
                self.logger.warning("llm: response truncated at %d tokens, retrying with %d",
                                    max_tokens, bigger)
            text2, _ = self._invoke(messages, bigger, temperature, json_mode)
            if parse_json_loose(text2) is not None:
                text, max_tokens = text2, bigger
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += estimate_messages(messages)
        self.stats["completion_tokens"] += estimate_tokens(text)
        if self.logger:
            self.logger.info("llm← %s in %.1fs (finish=%s, %d chars)", task,
                             time.perf_counter() - started, finish or "stop", len(text or ""))
        if self.use_cache and self.store is not None and text:
            self.store.cache_put(key, task, text)
        return text

    def json(self, messages: List[dict], task: str = "generic", max_tokens: Optional[int] = None,
             temperature: Optional[float] = None, default: Any = None) -> Any:
        text = self.chat(messages, task=task, max_tokens=max_tokens, temperature=temperature, json_mode=True)
        parsed = parse_json_loose(text)
        if parsed is not None:
            return parsed
        # One repair round-trip with a nudge; different cache key, lower temp.
        if self.logger:
            self.logger.warning("llm: unparseable JSON for task=%s (len=%d), retrying",
                                task, len(text or ""))
        repair = list(messages) + [{
            "role": "user",
            "content": "上一条输出无法解析为 JSON。请只输出一个合法的 JSON 对象，"
                       "不要任何解释、markdown 代码块或多余文字。",
        }]
        text2 = self.chat(repair, task=f"{task}_repair",
                          max_tokens=max(1200, int(max_tokens or self.default_max)),
                          temperature=0.3, json_mode=True)
        parsed2 = parse_json_loose(text2)
        if parsed2 is not None:
            return parsed2
        if self.logger:
            self.logger.warning("llm: repair failed for task=%s", task)
        return {} if default is None else default

    def note(self, which: str) -> None:
        self.stats[which] = self.stats.get(which, 0) + 1

    # -- internals
    def _key(self, messages, task, max_tokens, temperature) -> str:
        h = hashlib.sha256()
        h.update(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        # backend identity is part of the key: a Mock response must never be
        # replayed as if it came from the real model (or vice versa).
        h.update(f"|{self.backend_name}|{self.model_name()}|{task}|"
                 f"{max_tokens}|{temperature:.3f}".encode("utf-8"))
        return h.hexdigest()

    def model_name(self) -> str:
        return getattr(self.backend, "model", "mock")

    def _invoke(self, messages, max_tokens, temperature, json_mode) -> tuple[str, str]:
        last_err: Optional[Exception] = None
        budget = max_tokens
        for attempt in range(self.retries + 1):
            try:
                return self.backend.chat(messages, budget, temperature, json_mode)
            except urllib.error.HTTPError as exc:
                # The server answered, so this is a request problem.  The usual
                # cause is prompt+completion exceeding the context window, which
                # shrinking the completion budget fixes.
                last_err = exc
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:200]
                except Exception:  # pragma: no cover
                    pass
                if self.logger:
                    self.logger.error("llm HTTP %s: %s", exc.code, detail)
                if exc.code == 400 and budget > 256:
                    budget = max(256, budget // 2)
                    if self.logger:
                        self.logger.warning("retrying with max_tokens=%d", budget)
                    continue
                break
            except Exception as exc:  # noqa: BLE001 - transport agnostic
                last_err = exc
                if (self.backend_name == "http" and self.auto_fallback
                        and self._looks_like_connection_error(exc)):
                    self._switch_to_mock(str(exc))
                    try:
                        return self.backend.chat(messages, budget, temperature, json_mode)
                    except Exception as exc2:  # pragma: no cover
                        last_err = exc2
                if self.logger:
                    self.logger.warning("llm attempt %d/%d failed: %s", attempt + 1, self.retries + 1, exc)
                time.sleep(min(4.0, 0.8 * (attempt + 1)))
        self.stats["errors"] += 1
        raise LLMError(f"LLM call failed after {self.retries + 1} attempts: {last_err}")

    def _switch_to_mock(self, reason: str) -> None:
        self.mock = True
        self.degraded = True
        self.backend_name = "mock"
        self.backend = MockBackend()
        if self.logger:
            self.logger.error(
                "LLM unreachable (%s) — DEGRADED: falling back to the deterministic MockBackend; "
                "content from here on is fabricated and must not be mistaken for model output",
                reason)

    @staticmethod
    def _looks_like_connection_error(exc: Exception) -> bool:
        """True only for genuine transport failures.

        ``HTTPError`` is a subclass of ``URLError`` which is a subclass of
        ``OSError``, so a naive isinstance check would treat *any* server error
        (e.g. 400 context-overflow) as "unreachable" and silently swap in the
        Mock backend — replacing real generated content with fabricated data.
        A server that answered is by definition reachable.
        """
        if isinstance(exc, urllib.error.HTTPError):
            return False
        if isinstance(exc, (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError)):
            return True
        if isinstance(exc, OSError):
            text = str(exc).lower()
            return "refused" in text or "timed out" in text or "unreachable" in text
        return False
