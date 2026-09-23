# PITFALL — 踩坑记录

## 1. 推理模型默认输出 thinking，吃光 max_tokens

**现象**：`max_tokens=32` 时 `content` 为空，全部预算被 `reasoning_content` 占用。
**原因**：Qwen3 系列默认开启思考，返回里 thinking 与最终答案分开。
**修复**：所有请求带 `chat_template_kwargs: {"enable_thinking": false}`（见 `llm.py::_HTTPBackend.chat`）。

## 2. `sqlite3.OperationalError: 18 values for 19 columns`

**现象**：`add_npc` 插入报占位符数量不符。
**原因**：多行字符串拼接 `VALUES(?,...,?,1,?,?,?)` 时手数占位符数错了（`alive` 用字面量 1）。
**修复**：拆行写、显式核对；`alive` 单独用字面量，占位符 18 个对应 19 列。

## 3. `hash_int()` 缺少关键字参数 `mod`

**现象**：`TypeError: hash_int() missing 1 required keyword-only argument: 'mod'`。
**原因**：`rng.hash_int(*parts, mod=...)` 的 `mod` 是 keyword-only，调用时漏了。
**修复**：所有调用都显式传 `mod=`。

## 4. 地形矩阵退化：`=.=.=.=` / `..,,..,,`

**现象**：真实模型生成的 16 字符行退化为周期重复串；第一版还把图例写成 `~=water .=grass`，
模型把 `=` 当成了地形字符。
**原因**：(a) 图例 `sym=name` 有歧义；(b) 长重复字段在低熵采样下容易自我复制。
**修复**：
* 图例改为 `~ water | - shallow | . grass | ...`（符号与名称用空格分隔，配对用 `|`）。
* 明确"字符必须紧密相连、不要分隔符"并给出示例行。
* chunk 任务温度降到 0.65。
* `world.py::_clean_rows` 剔除非法字符、短行用噪声补齐；
  `_rows_are_sane` 检测退化（不同字符 < 3 或过半数行周期重复），退化则整体改用
  父级 `biome_mix` 驱动的程序化地形。

## 5. 回复被 `finish_reason=length` 截断 → JSON 解析失败

**现象**：`world` 任务偶尔返回 `None`，世界名变成默认值。
**原因**：`max_tokens` 太小，模型话没说完。
**修复**：`LLMClient.chat` 检测 `finish_reason == "length"`，
若 JSON 不可解析则用更大预算重试一次；`json()` 另有一轮"只输出 JSON"的修复往返。

## 6. python 输出重定向到文件后看不到进度

**现象**：`python3 main.py ... > log 2>&1` 后日志长时间为 0 字节。
**原因**：stdout 被块缓冲，进程结束才 flush。
**修复**：跑长任务用 `python3 -u`（或设 `PYTHONUNBUFFERED=1`）。

## 7. macOS 没有 `timeout` 命令

**现象**：`timeout 30 cmd` → `command not found: timeout`。
**修复**：macOS 默认无 `timeout`（GNU coreutils 的 `gtimeout` 也多数未装）；
改用 Python 内部超时参数（如 `urllib` 的 `timeout=`），或放进 tmux 后手动 `kill-session`。

## 8. 只读方式读 WAL 数据库看到 0 行

**现象**：`sqlite3.connect(f"file:{p}?mode=ro", uri=True)` 读正在写入的库，所有表计数为 0，
但 `-wal` 文件已很大。
**原因**：WAL 模式下只读连接未能加载/回放 WAL（或只看到已 checkpoint 的主库）。
**修复**：用普通读写连接读取状态即可。

## 9. 玩家可能出生在被水/山围死的格子里

**现象**：`auto` 探索出现"实际移动 0 格"。
**修复**：`WorldManager.find_spawn` 螺旋搜索一个可通行且**至少两个正交方向可走**的格子。

## 10. Mock 与真实模型的缓存互相污染

**现象**：跑完单测后启动真实模型，世界瞬间"生成"完毕，但内容全是 Mock 的；
`worlds` 表 0 行、缓存里 11 条记录时间戳几乎相同。
**原因**：缓存 key 只含 messages/task/tokens/temperature，**不含后端身份**；
单测用 Mock 后端写进同一个 `data/llm_cache.db`，真实运行直接命中。
**修复**：
* `LLMClient._key` 加入 `backend_name` 与 `model_name`。
* 测试用 `test_cfg()` 把 `llm.cache_path` 指到 `data/test_llm_cache.db`，与生产缓存隔离。

## 11. 缓存放在存档库里，删档即失忆

**现象**：删除 `world.db` 重开世界，所有 LLM 调用（每次约 45 s）全部重跑。
**修复**：把缓存拆到独立的 `CacheStore`（默认 `data/llm_cache.db`），
与存档解耦；重开世界/重跑测试都零成本。

## 12. zsh 下 `rm -f a*` 遇到无匹配会让整条命令中止

**现象**：`rm -f data/llm_cache.db data/test_*.db*` 中后一个 glob 无匹配，
zsh 报 `no matches found` 并**中止整行**，结果前面的文件也没删掉。
**原因**：zsh 默认 `nomatch` 行为（bash 会保留字面量）。
**修复**：逐个写明确路径，或对 glob 加引号、用 `setopt NULL_GLOB`。

## 13. 模型键名漂移：`name` / `world_name` / `area_name`

**现象**：同一提示词，模型这次返回 `name`，下次返回 `world_name`、`region_name`、`area_name`，
导致世界名落到默认值"无名之地"。
**修复**：`world.py::_field` 先按显式别名匹配，再按 `_<key>` 后缀匹配
（找 `name` 时 `area_name` 也命中），比重新生成一次（45 s）便宜得多。

## 14. 让模型输出 256 个字符几乎必然退化

**现象**：要求 16 行 × 16 字符的地形矩阵，模型稳定产出 `=.=.=.=`、`mm..mm..`、
整行同一个字符等退化结果。
**原因**：长且要求"逐字符精确"的字段，在小量化模型 + JSON 语法约束下极易自我复制。
**修复**：改让模型输出 **3-6 个地形矩形**（`patches`），服务端栅格化并用确定性噪声补空。
模型表达"哪里有湖/林/废墟"很擅长，且省下大量 token 与时间。
仍保留对 `rows` 的兼容与合法性/多样性校验（`_rows_are_sane`）。

## 15. JSON 键里带空格：`{"  name": ...}`

**现象**：区域名/地名落到默认值，检查缓存发现键是 `"  name"`（两个前导空格）。
**原因**：模型在键名前后输出了空白。
**修复**：`parse_json_loose` 解析成功后递归 `_clean_keys` 去掉所有对象键的首尾空白；
`world._field` 再做别名 + 后缀匹配。

## 16. 测试输出污染 unittest 结果

**现象**：REPL 里的 `print` 混进测试输出。
**修复**：测试中用 `contextlib.redirect_stdout(io.StringIO())` 包住 `game.execute`。

## 17. `HTTPError` 被当成"连不上"，静默降级为 Mock（最危险的一个）

**现象**：长跑 QA 时，本地模型明明健康（`/v1/chat/completions` 200），但日志里几十次调用在 2 分钟内跑完，
内容全是 Mock 编的，且**没有任何提示**。
**原因**：`urllib.error.HTTPError` 继承自 `URLError`，而 `URLError` 继承自 `OSError`；
`_looks_like_connection_error` 里 `isinstance(exc, OSError)` 直接返回 True。
于是任何 4xx/5xx（例如 prompt+max_tokens 超出上下文返回 400）都会触发"不可达"分支，
把后端永久换成 Mock——**用虚构内容替代模型输出，且看起来完全正常**。
**修复**：
* `HTTPError` 一律不算连接失败（服务器既然回答了就说明可达）；
* HTTP 400 视为"上下文超限"，自动把 `max_tokens` 减半重试；
* 真降级时用 `ERROR` 级别日志，并置 `LLMClient.degraded`，`stats` 里高亮警告；
* QA 脚本在开跑前 `ping()`，结束时若发生降级直接判定为失败。

## 18. 演化 delta 的坐标系不明确

**现象**：模型说"在 (5,7) 新增物体"，物体却出现在世界坐标 (5,7)——离本层十万八千里。
**原因**：chunk 的提示词里既有"左上角世界坐标"，又有"features 坐标 0-15"，
演化响应里的 `x/y` 到底用哪个系没有明说；`_in_scope` 按世界坐标校验，于是大部分 delta 被丢弃。
**修复**：演化提示词顶部加 `[[SCOPE:x,y,w,h]]` 并明写"本层世界坐标范围 x∈[..], y∈[..]，
**所有 x/y 一律用世界坐标**，不要用 0-15 局部坐标"。

## 19. 世界只有玩家脚下在演化（其余区域被冻结）

**现象**："走远再回来"，A 区的一切和离开时一模一样——没有演化痕迹。
**原因**：`_due_scopes` 只收集玩家附近半径内的 chunk，以及玩家所在的那一条 region/zone 祖先链。
玩家不在的地方永远不会被调度。
**修复**：`_plan` 遍历**数据库里所有已生成的节点**，按（层级，到玩家的距离）排序调度；
离场过久（≥ `catchup_steps` 个周期）时用一次"追赶"调用概括这段时间的累积变化，
预算由 `max_catchup_calls` 控制，避免一次等待就打出几十次 LLM 调用。

## 20. 演化会"改写"历史描述

**现象**：同一个物体在多次演化后描述被整段替换，玩家回来认不出；物体被删除后历史彻底丢失。
**修复**：物体改描述时保留旧文本（`新描述（原为：旧描述）`）；`destroy()` 改为软删除
（`alive=0` + `state.destroyed`），行保留，仍然可查询、可被后续演化引用。

## 21. 抓到的最危险缺陷：HTTPError 被当成断线而静默降级 Mock

见第 17 节。它的可怕之处不是崩溃，而是**看起来一切正常**：内容全是 Mock 编的，
JSON 合法、结构完整、剧情连贯，只有读日志才能发现。任何"自动降级"都必须
带上显式、醒目的标记，并且测试/QA 必须断言后端身份。

## 22. 长跑时误判墙钟时间，把正常生成当成卡死

**现象**：以为某个 LLM 调用卡了 4 分钟，实际只过了 60 秒（`ps -o etime` 读数被误判）。
**教训**：先用 `date` + `ps -o lstart` 对齐真实时间，再下"卡死"的结论；
耗时任务的进度改为写状态文件（`--status`），而不是靠感觉。
