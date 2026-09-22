#!/usr/bin/env python3
"""End-to-end smoke test using the deterministic Mock backend (no network).

    python3 scripts/smoke_test.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def main() -> int:
    db = os.path.join(ROOT, "data", "smoke.db")
    if os.path.exists(db):
        os.remove(db)
    cmd = [sys.executable, os.path.join(ROOT, "main.py"), "--new", "--mock",
           "--db", db, "--no-story", "--script", os.path.join(ROOT, "scripts", "demo.txt")]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        print("✗ 冒烟测试失败：进程非零退出")
        return 1
    if "Traceback" in proc.stderr:
        print(proc.stderr, file=sys.stderr)
        print("✗ 冒烟测试失败：出现异常")
        return 1

    from pcg.db import Store
    store = Store(db)
    wid = int(store.conn.execute("SELECT id FROM worlds ORDER BY id DESC LIMIT 1").fetchone()["id"])
    stats = store.stats(wid)
    store.close()

    checks = [
        ("节点数 > 0", stats["nodes"] > 0),
        ("格子数 >= 256", stats["tiles"] >= 256),
        ("物体数 > 0", stats["objects"] > 0),
        ("事件数 > 0", stats["events"] > 0),
        ("世界已生成", stats["nations"] >= 3),
    ]
    ok = True
    for name, passed in checks:
        print(f"{'✓' if passed else '✗'} {name}  ({stats})")
        ok = ok and passed
    print("✓ 冒烟测试通过" if ok else "✗ 冒烟测试未通过")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
