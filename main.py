#!/usr/bin/env python3
"""Entry point.

    python main.py --new                 # 用本机 127.0.0.1:8080 的模型新建世界
    python main.py --new --mock          # 不联网，用确定性 Mock 后端（快速试玩）
    python main.py --load                # 继续最近的存档
    python main.py --script scripts/demo.txt   # 非交互跑一段指令
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pcg.config import load_config  # noqa: E402
from pcg.game import Game  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LLM 驱动的开放世界 roguelike")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--new", action="store_true", help="新建世界（覆盖互动存档中的 world id）")
    g.add_argument("--load", action="store_true", help="读取数据库里最近的世界")
    p.add_argument("--world-id", type=int, default=None, help="指定要加载的世界 id")
    p.add_argument("--seed", type=int, default=None, help="世界种子")
    p.add_argument("--hint", type=str, default="", help="给世界生成器的额外要求")
    p.add_argument("--db", type=str, default=None, help="数据库路径（默认 data/world.db）")
    p.add_argument("--config", type=str, default=None, help="配置文件路径")
    p.add_argument("--mock", action="store_true", help="强制使用确定性 Mock 后端（不访问 LLM）")
    p.add_argument("--script", type=str, default=None, help="按文件中的指令逐行执行（非交互）")
    p.add_argument("--no-story", action="store_true", help="不自动生成任务线")
    p.add_argument("--verbose", "-v", action="store_true", help="打印每次 LLM 调用与耗时")
    p.add_argument("--allow-mixed", action="store_true",
                   help="允许用与存档不同来源的后端继续写入（会覆盖已有文本，慎用）")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    overrides = {"llm": {}, "game": {}}
    if args.mock:
        overrides["llm"]["mock"] = True
    if args.db:
        overrides["game"]["db_path"] = args.db
    if args.allow_mixed:
        overrides["game"]["allow_mixed_backend"] = True
    cfg = load_config(args.config, overrides)

    game = Game(cfg, verbose=args.verbose)
    game.auto_story = not args.no_story
    try:
        if args.new or not args.load and args.world_id is None and not _has_world(game):
            world = game.new_world(seed=args.seed, hint=args.hint)
            print(f"已创建世界：{world['name']}（{world['era']}）")
            print(f"魔法：{world['data'].get('magic_system','')}")
            print(f"科技：{world['data'].get('tech_baseline','')}")
            print(f"概要：{world['summary']}")
            backend = game.llm.backend_name
            print(f"LLM 后端：{backend}"
                  + ("" if backend == "http" else "（未连接到 127.0.0.1:8080，使用确定性模拟）"))
        else:
            world = game.load_world(args.world_id)
            game.check_backend_matches()
            print(f"已载入世界：{world['name']}（{world['era']}），第 {game.tick()//24+1} 天")
        script = None
        if args.script:
            with open(args.script, "r", encoding="utf-8") as fh:
                script = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("#")]
        game.run(script=script)
    finally:
        game.close()
    return 0


def _has_world(game: Game) -> bool:
    row = game.store.conn.execute("SELECT COUNT(*) c FROM worlds").fetchone()
    return bool(row and row["c"])


if __name__ == "__main__":
    raise SystemExit(main())
