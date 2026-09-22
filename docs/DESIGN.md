# 设计说明

## 1. 目标

一个大模型驱动的开放世界 roguelike：世界的**历史演化**（国家、民族、魔法、科技、矿产、农业、天气）由 LLM 决定；
地图**随玩家探索逐步生成**；同一个游戏对象在每个时间步的**新状态**都要落库。

硬约束：本地模型上下文约 **64k**（训练 140k），实测生成速度约 **14 tok/s**。
因此设计的核心不是"让模型看到世界"，而是"让模型只看到**它此刻必须看到的那一小部分世界**"。

## 2. 树状生成粒度分层（LOD Tree）

```
LOD 0 世界 world        1 个节点     宇宙观 / 魔法体系 / 科技基线 / 国家列表
 └ LOD 1 区域 region    512x512 格   气候 / 生物群系权重 / 文化 / 显著地点 / 危险
    └ LOD 2 子区域 zone 128x128 格   细化地貌 / 地物 / NPC 种子
       └ LOD 3 地块 chunk 16x16 格   16x16 地形矩阵 + 地物 + NPC
          └ 格子 tile                单格：名 / 描述 / 细节（按需生成）
             └ 物体 object           单实体：名字 / 描述 / 状态 / 耐久
```

* 节点存 `nodes` 表：`(world_id, lod, x, y, w, h, name, summary, data_json, parent_id)`，
  唯一键 `(world_id, lod, x, y)`，天然构成树。
* **自顶向下按需生成**：`ensure_node(lod, tx, ty)` 会先确保父节点存在。
* **父节点的 `summary` 是子节点提示词的唯一上层上下文**（约 100-200 字），
  于是生成一个 chunk 的提示词里永远不会出现区域以外的内容。

这就是"分级 LOD 降低上下文需求"的落点：上下文量与**层数**成正比，与**地图面积**无关。

## 3. 上下文预算

| 手段 | 说明 |
| --- | --- |
| 固定 system prompt | 所有调用共用同一段 system，llama.cpp 前缀缓存可复用 |
| 世界圣经 world bible | 每世界一份、字节稳定的世界摘要（< 320 tok），放在 system 之后 |
| 父级摘要 | 只带 1 个父节点 summary |
| 邻居摘要 | 只带半径内最多 12 个物体 / 8 个 NPC / 6 个同级节点 |
| 事件摘要 | 本层 8 条 + 更高层 6 条 |
| 响应缓存 | `llm_cache` 表按 `(messages, task, max_tokens, temperature)` 哈希缓存，重跑零成本 |
| 截断保护 | `tokens.clamp_text` + `evolution._fit_budget` |
| 截断重试 | `finish_reason=length` 且 JSON 不可解析时自动用更大预算重试 |
| JSON 修复 | 去代码块 / 去尾逗号 / 单引号 / 前后夹带正文 |

估算器：CJK ≈ 1 tok/字，ASCII ≈ 0.28 tok/字，其余 ≈ 0.5。

## 4. 演化：同时看邻接对象与更高 LOD

`evolution.py` 为每个 LOD 配了独立时钟：

| LOD | 周期 | 关注点 |
| --- | --- | --- |
| chunk | 6 游戏小时 | 天气、野兽、小规模变化（可能改单格地形） |
| zone | 48 小时 | 聚落、旅人、地方政治 |
| region | 168 小时 | 经济、矿业、战争、迁徙 |
| world | 720 小时 | 纪元、魔力潮汐、帝国兴衰 |

一次局部演化调用会同时收到：

1. **父级摘要**（`scope_desc`）
2. **本层近期历史**（`events` 表按 lod + node_id 检索）
3. **更高 LOD 的演化结果**（`_higher_digest`：世界/区域/子区域事件按祖先链重合过滤）
4. **相邻对象现状**（`objects_near` / `npcs_near`，带 `#id` 便于模型引用）

模型只返回**小增量**（`changes`），由 `_apply_change` 白名单落地：
`new_object / new_npc / npc(hp,move,mood) / object(destroy) / tile / nation`。
这样模型无法破坏数据库结构。

为保证连贯，`max_llm_calls_per_wait` 限制一次等待的调用数；局部 chunk 每次最多演化 `max_chunk_scopes_per_advance` 个。

## 5. 玩家侧

* 无图形界面。`render.render_view` 输出 16x16 文本网格（`@` 是主角，汉字是 NPC）。
* "点击格子" = `inspect <x> <y>`，首次查看时才让 LLM 生成该格描述并**立即落库**。
* 移动 / 战斗 / 掉落 / 升级全部是 Python 逻辑（`entities.py`），只有战报描写走 LLM。
* 故事线与 NPC 对话实时生成（`narrative.py`），对话历史存 `dialogue` 表，
  每次只回灌最近 6 轮，避免上下文膨胀。
* 探索即推进时间：`move` 1 小时，`wait n` n 小时，时间推进触发演化。

## 6. 数据库

见 `db.py` 的 `SCHEMA`：`meta / worlds / nodes / tiles / objects / npcs / nations /
events / dialogue / story / player / llm_cache`。
所有 LLM 产物（描述、演化结果、对话、任务、事件）都落库，重开游戏可继续。

## 7. 确定性

地形矩阵用 `rng.py` 的哈希值噪声生成，`(seed, x, y)` 决定，可复现。
模型输出非法字符会被剔除，行不足则用噪声补齐，矩阵退化（`..,,..,,`）则整体改用
父级 LLM 决定的 `biome_mix` 程序化生成 —— 保证任何情况下地图都可用。
