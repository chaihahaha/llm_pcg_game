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

## 7. 地形布局的表示（为什么用矩形而不是字符矩阵）

最初的方案是让模型直接输出 16x16 = 256 个单字符代码。实测该 27B 量化模型**几乎必然退化**
（`=.=.=.=.`、`mm..mm..`、整行同一个字符），既浪费约 40 秒生成时间，又要靠后处理丢弃。

改为让模型输出 **3-6 个地形矩形**：

```json
"patches": [{"terrain":"m","x":0,"y":0,"w":9,"h":16},
            {"terrain":"!","x":9,"y":6,"w":7,"h":6}]
```

* 模型只需表达"哪里有湖、哪里有废墟"，这正是它擅长的（小结构 JSON，无需数 256 个字符）。
* 服务端按顺序栅格化（后者覆盖前者），**未被覆盖的格子用 `(seed, x, y)` 的哈希值噪声
  按父级 `biome_mix` 填充**。
* 兼容旧格式：若模型仍返回 `rows` 且通过合法性/多样性检查，则直接采用。

这样既保留了"地图由大模型决定"，又让结果**永远可用且可复现**。

## 8. 一致性：世界不会"失忆"

长跑 QA（走远 → 时间流逝 → 返回）暴露的核心风险是**记忆**，而不只是生成质量。
系统里有四层记忆，且每层都会被写回提示词：

| 层 | 存储 | 写回方式 |
| --- | --- | --- |
| 玩家故事线 | `story` 表全部历史 | `saga()`：历次任务标题/结局/目标 |
| 场景往事 | `nodes.data.history` | 每次演化必须更新并承接的滚动摘要 |
| NPC 记忆 | `npc_memory` 表 | 对话提示词 + 演化邻居摘要；超限时由 LLM 压缩而非丢弃 |
| 关系图 | `relations` 表 | 世界/区域/NPC 的摘要都带上相关的边 |

另外的保护：

* **软删除**：物体被毁只置 `alive=0` 并写残迹描述，行与历史保留；新物体落在有残迹的格子上
  会**原地转化**那一行，而不是叠出第二个实体。
* **单调生死**：`alive` 一旦为 0 不再复活；HP 夹在 `[0, hp_max]`。
* **单步位移上限**：NPC 一次演化最多移动 ±2/次、≤3/步，禁止瞬移。
* **身份不可变**：`name/race/role/appearance` 不允许被演化改写。
* **事件时间**：追赶式演化的事件按比例散布在流逝的时间窗内，而不是全挤在同一时刻。
* **已毁标记**：邻接摘要里已毁物体标 `[已毁]`，模型不会再宣告一次它的毁灭。

## 9. 自由动作与运行时 monkey patch

### 9.1 为什么不是裸 `eval`/`exec`

`eval("lambda ctx: ...")` 就是 monkey patch 的标准做法——本项目也确实是这么做的
（`sandbox.compile_hook`）。但**裸** `eval`/`exec` 会把整台机器的能力交给模型：
`import os`、`open('data/world.db','w')`、`while True`、`[0]*10**9`，
而且坏补丁会写进存档，下次读档继续炸。

所以分两层，都经过 AST 校验、都在没有 builtins 的命名空间里执行：

| 层 | 机制 | 能力 |
| --- | --- | --- |
| 表达式 | `eval(compile(ast, mode="eval"))` | 一个 `lambda ctx: ...`，装在 hook 点上 |
| 语句 | `exec(compile(ast, mode="exec"))` | 赋值 / `if` / `for` / `def` / 调用 api |

守卫：无 `import`/`while`/`with`/`class`/`global`/`try`/`raise`；无下划线开头的属性
（堵住 `__class__`/`__globals__`）；只允许纯函数白名单；整数常量 ≤1e6、禁 `**`
（防内存炸弹）；语句数与字符数有上限。没有 `while`、`for` 只能遍历我们给的数据，
所以不存在死循环，不需要超时线程。**这是内容沙箱，不是对抗恶意本地代码的安全边界。**

### 9.2 三个 patch 面

1. **rule**：引擎运行时读取的旋钮（`npc_speech_max_chars`、`move_cost_hours`、
   `damage_multiplier`、`extra_passable_terrain`、`forbidden_words`、`evolution_bias`…）。
2. **hook**：装在真实决策点上的 lambda —— `can_enter` / `on_enter` / `damage` /
   `speech` / `move_cost` / `evolve_bias`。返回 `None` 表示沿用默认。
3. **code**：语句级补丁，通过注入的 `api`（`set_rule` / `add_hook` /
   `register_action` / `log`）修改游戏，可定义自己的辅助函数。

补丁存在 `patches` 表，读档时重新编译并**重新执行** code patch，
因此"诅咒"跨会话生效；编译失败的补丁会被自动停用（`enabled=0`）并报错，绝不带走存档。

### 9.3 自由动作

`do <任意文本>` → 模型输出**动作程序**：

```json
{"feasible": true, "narrative": "...", "cost_hours": 2,
 "effects": [{"op":"set_tile","dx":0,"dy":0,"terrain":"cave","name":"竖井口"}],
 "new_action": {"name":"dig","title":"向下挖掘","effects":[...]},
 "patch": {"kind":"code","source":"api[\"set_rule\"](\"npc_speech_max_chars\",12)", "reason":"..."}}
```

`effects` 由 `effects.py` 唯一的解释器落地（坐标必须离玩家 ≤3 格，地形取自固定词表，
数值夹紧），模型无法直接触碰数据库。可复用的动作写进 `actions` 表，
下次 `do dig` 直接回放，**不花 LLM 调用**。

### 9.4 人类可读的反馈

- `rules` 打印所有被改过的规则、钩子和 code patch 源码；
- `actions` 列出学会的动作与使用次数；
- 补丁被拒绝时明确说明原因（哪条沙箱规则），而不是静默失败。

## 10. 键名容错

本地模型同一提示词会给出 `name` / `world_name` / `region_name` / `area_name`，
甚至 `"  name"`（键里带前导空格）。处理：
`parse_json_loose` 递归 strip 所有对象键；`world._field` 再按显式别名 + `_<key>` 后缀匹配。
代价极低，远好于重新生成一次。
