#!/usr/bin/env python3
"""Ping the local model and time a couple of real generations.

    python3 scripts/check_llm.py
    python3 scripts/check_llm.py --generations 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from pcg import prompts  # noqa: E402
from pcg.config import load_config  # noqa: E402
from pcg.llm import LLMClient  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=2)
    args = ap.parse_args()

    cfg = load_config()
    cfg["llm"]["mock"] = False
    cfg["llm"]["cache"] = False  # always a real call for a connectivity check
    client = LLMClient(cfg, store=None)

    base = cfg["llm"]["base_url"]
    print(f"探测 {base} …")
    if not client.ping():
        print("✗ 无法连接。请确认 llama.cpp / OpenAI 兼容服务已在 127.0.0.1:8080 运行。")
        return 1
    print("✓ 服务可达")

    t0 = time.time()
    world = client.json(prompts.build("", prompts.world_task(1234)), task="world")
    dt = time.time() - t0
    print(f"✓ world 生成 {dt:.1f}s -> {world.get('name')}｜{world.get('era')}")
    print(f"  魔法：{world.get('magic_system', '')}")
    for nation in (world.get("nations") or [])[:4]:
        print(f"  · {nation.get('name')} ({nation.get('race')}, {nation.get('gov')})")

    bible = prompts.world_bible(world, world.get("nations") or [])
    for i in range(1, args.generations):
        t0 = time.time()
        chunk = client.json(prompts.build(bible, prompts.chunk_task(1234 + i,
                                                                   {"name": "测试地", "summary": "缓坡草地",
                                                                    "data": {"biome_mix": [["grass", 0.6], ["forest", 0.4]]}},
                                                                   0, 0, 16)),
                            task="chunk")
        rows = chunk.get("rows") or []
        ok = len(rows) == 16 and all(len(str(r)) == 16 for r in rows)
        print(f"✓ chunk 生成 {time.time()-t0:.1f}s -> 行数 {len(rows)}, 尺寸合法 {ok}, "
              f"features {len(chunk.get('features') or [])}, npcs {len(chunk.get('npcs') or [])}")
        if not ok:
            print("  原始 rows:", rows[:3])

    print(f"统计：{client.stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
