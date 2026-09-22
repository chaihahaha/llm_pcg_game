# LLM-PCG 开放世界 Roguelike

一个**没有图形界面**的文字 roguelike：世界的历史演化（国家、民族、魔法、科技、矿产、农业、天气）、
每一格的地貌与物体描述、NPC 对话与故事线，全部由本地大模型实时生成；
移动、战斗、掉落等简单逻辑由 Python 完成。所有 LLM 产物写进 SQLite，随玩随存。

* 地图按 **树状 LOD 分层**（世界 → 区域 → 子区域 → 地块 → 格子 → 物体）自顶向下按需生成，
  子层只读父层摘要，因此上下文占用与**地图面积无关**。
* 演化按 LOD 各自的时钟推进，局部演化同时参考**数据库中的相邻对象**与**更高层级的演化结果**。
* 玩家只看得到周围 **16x16** 的格子；"点击"某格用 `inspect x y` 查看说明。

## 快速开始

```bash
# 1) 需要 OpenAI 兼容服务（默认 127.0.0.1:8080，llama.cpp 亦可）
python3 scripts/check_llm.py            # 连通性与生成速度自检

# 2) 新建世界（真实模型）
python3 main.py --new

# 3) 继续最近的存档
python3 main.py --load

# 4) 离线试玩：不需要模型，使用确定性 Mock 后端
python3 main.py --new --mock

# 5) 非交互跑一段指令
python3 main.py --new --mock --script scripts/demo.txt
```

零第三方依赖（仅 Python 3.10+ 标准库）。数据默认在 `data/world.db`。

## 指令

| 指令 | 说明 |
| --- | --- |
| `look` / `l` | 刷新周围 16x16 视图与状态栏 |
| `move <w\|a\|s\|d\|n\|e\|nw\|...>` | 移动一格（推进 1 小时） |
| `goto <x> <y>` | 朝目标走一格 |
| `inspect [x y]` | 查看某格说明（缺省脚下）——即"点击格子" |
| `talk [名字] [内容]` | 与相邻人物实时对话 |
| `attack [名字]` | 攻击相邻目标（Python 战斗 + LLM 战报） |
| `wait [小时]` | 原地等待，推进世界演化 |
| `story` / `newstory` | 查看 / 立即生成任务线 |
| `journal [n]` | 最近事件（含世界级大事） |
| `world` | 世界圣经（国家 / 魔法 / 科技） |
| `stats` | 数据库规模与 LLM 调用/缓存统计 |
| `auto <n>` | 自动探索 n 步 |
| `help` / `quit` | 帮助 / 退出（自动存档） |

加 `-v/--verbose` 可打印每次 LLM 调用的任务名、估算输入与耗时。

## 目录结构

```
main.py                 入口（--new/--load/--mock/--script）
configs/default.json    配置（模型、LOD 尺寸、演化周期、上下文预算）
src/pcg/
  config.py             配置合并与点号取值
  tokens.py             免依赖 token 估算与截断
  rng.py                确定性哈希与值噪声
  terrain.py            地形词表（单字符代码）
  prompts.py            提示词模板（稳定前缀、机器可读 task 标记）
  llm.py                OpenAI 兼容客户端 + 缓存 + JSON 修复 + Mock 后端
  db.py                 SQLite schema 与 DAO
  world.py              LOD 树生成与落库
  evolution.py          多 LOD 演化引擎
  entities.py           玩家 / 移动 / 战斗（纯 Python）
  narrative.py          故事线导演 + NPC 对话
  render.py             16x16 文本渲染与格子说明
  game.py               编排与 REPL
scripts/                check_llm.py / smoke_test.py / demo*.txt
tests/                  unittest（python3 -m unittest discover -s tests）
docs/DESIGN.md          架构与上下文预算策略
```

## 测试

```bash
python3 -m unittest discover -s tests -v   # 32 个用例，全部走 Mock，零网络
python3 scripts/smoke_test.py              # 端到端冒烟
```

## 配置要点

`configs/default.json`：

* `llm.context_limit` — 上下文上限（默认 65536，与本地服务 n_ctx 对齐）。
* `world.region_size / zone_size / chunk_size / view_size` — LOD 尺寸。
* `evolution.schedule` — 各 LOD 的演化周期（游戏小时）。
* `evolution.max_llm_calls_per_wait` — 单次等待的 LLM 调用上限（控制成本）。
* `context.max_prompt_tokens` — 单次提示词的软上限。

## 已知限制

* 本地模型每步生成约 30–60 秒（~14 tok/s），一次完整"新世界"约 3–4 分钟。
* 世界规模受上下文约束刻意受限：国家 3–5 个，NPC 上限 200，事件表自动裁剪。
* 地块布局由模型给出**地形矩形**（而非 256 字符矩阵，后者对小模型太难），
  未被覆盖处按父级 LLM 决定的 `biome_mix` 用确定性噪声补全。
* 地形矩阵由模型给出，若质量太差会自动回退为程序化生成（见 `docs/DESIGN.md` 第 7 节）。
