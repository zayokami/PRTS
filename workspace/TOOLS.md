# TOOLS.md

> 当前可用的工具汇总。Agent 启动后会自动刷新本文件(P4 阶段)。

## 内置 skill(workspace/skills/*.py)

P0 阶段:无。范例见 `skills/_examples/weather.py`。

## 外部 MCP 工具

P4 阶段接入,默认开启:

- `filesystem` —— 读写本地文件(限制在 workspace 目录内)
- (后续可加 `github` / `brave-search` / `prts-vector` 等)

## Rust 守护

- `prts-watcher` —— 文件变更 + cron 触发(P6)
- `prts-vector` —— 向量检索(P7)
- `prts-audio` —— 语音(P9)
