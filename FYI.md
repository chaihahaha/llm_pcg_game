# FYI — 外部知识与参考

## 本地模型服务

* 地址：`http://127.0.0.1:8080`（OpenAI 兼容：`/v1/models`、`/v1/chat/completions`）
* 模型：`qwen3.8_27b/Qwen3.8-27B-UD-IQ3_XXS.gguf`（llama.cpp，Q4_K_S，约 12 GB）
* `n_ctx = 70144`，`n_ctx_train = 140000`，`n_vocab = 248320`
* 推理速度实测：**约 13–14 tok/s**（M4 级硬件），prompt 处理约 37 tok/s
* 这是**推理模型**：默认会先输出 `reasoning_content`。本项目所有请求都带
  `chat_template_kwargs: {"enable_thinking": false}` 关闭思考，否则 `max_tokens` 会被思考内容吃光。

### 请求示例

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8_27b/Qwen3.8-27B-UD-IQ3_XXS.gguf",
       "messages":[{"role":"user","content":"只回复：PONG"}],
       "max_tokens":64,"temperature":0.3,
       "chat_template_kwargs":{"enable_thinking":false}}'
```

### 使用经验

* `response_format: {"type":"json_object"}` 可用，能显著减少解析失败。
* 温度过高（≥0.85）时，长字段（如 16x16 字符矩阵）容易**退化重复**；
  本项目的 chunk 生成用 0.65，并在 `world.py::_rows_are_sane` 做退化检测与程序化回退。
* prompt 前缀一致时 llama.cpp 会命中 KV cache（返回的 `timings.cache_n` 可见）。
  本项目固定 system prompt 以复用缓存。
* `finish_reason: "length"` 表示被 `max_tokens` 截断，需处理（本项目会自动加大预算重试一次）。

## 术语

* **LOD** (Level of Detail)：分层细节。本项目指"世界 → 区域 → 子区域 → 地块 → 格子 → 物体"的粒度树。
* **PCG** (Procedural Content Generation)：程序化内容生成。
* **roguelike**：回合制、程序化生成地图、永久死亡（本项目未启用永久死亡）。
