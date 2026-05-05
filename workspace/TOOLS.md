# TOOLS.md

> 当前可用的工具汇总(给博士看的静态目录)。运行时实际工具表见
> `GET /agent/v1/skills`(已注册的 LLM 工具) 与 `GET /agent/v1/mcp/servers`(MCP server 状态)。

## 内置 skill(workspace/skills/*.py)

P0 阶段:无。范例见 `skills/_examples/weather.py`。

## 外部 MCP 工具

P4 阶段:Agent 启动时读取 `workspace/mcp.json`,把声明的每个 MCP server 拉起来(stdio 子进程),把它们暴露的工具注册到 LLM 可见的工具表里。

**命名规则**:`<server_name>__<tool_name>`,前缀固定带,LLM 看到的名字就是这个。例如 `filesystem` server 暴露的 `read_text_file` 工具实际名是 `filesystem__read_text_file`。

**配置位置**:`workspace/mcp.json`,Claude Desktop 风格。可用变量:

- `${WORKSPACE_DIR}` — 当前 workspace 的绝对路径(POSIX)
- `${env:VAR_NAME}` — 环境变量(未定义则替换为空字符串并打 warning)

**默认 seed**:首启时本仓库附带的 `mcp.json` 预置一个 `filesystem` server 但 `disabled: true`,不会真去 `npx` 拉东西。要启用时把 `disabled` 改成 `false`(或删掉这一行)再重启 Agent。

**看状态**:`GET /agent/v1/mcp/servers`(经 gateway 是 `GET /mcp/servers`),返回每个 server 的 `status`(`ready` / `error` / `disabled`)、报错信息、注册到 registry 的工具名列表。

**热重载**:`POST /agent/v1/mcp/reload`(经 gateway 是 `POST /mcp/reload`),停止所有现有 MCP server,重新加载 `workspace/mcp.json` 并启动。改配置后无需重启 Agent。

**健康检查**:`POST /agent/v1/mcp/health-check`(经 gateway 是 `POST /mcp/health-check`),探测每个 MCP server 的连接是否仍然存活。默认自动重启失败的 server;传 `?auto_restart=false` 可只检查不重启。返回每个 server 的 `healthy` / `unhealthy` / `restarting` / `disabled` / `error` 状态。

**已知 server**:

- `filesystem` —— 官方 [`@modelcontextprotocol/server-filesystem`](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem),读写本地文件(限制在传入的目录内)。seed 默认指向 `${WORKSPACE_DIR}`。
- `prts-vector` —— 自研向量检索 MCP server,基于 sqlite-vec。提供 `prts-vector__upsert`(写入向量)和 `prts-vector__search`(L2 最近邻搜索)两个工具。
  - **编译**: `cargo build --bin prts-vector`
  - **启用**: 把 `workspace/mcp.json` 中 `prts-vector.disabled` 改为 `false`,并确保 `command` 指向 binary(已编译时可直接用 `prts-vector`,或用绝对路径 `target/debug/prts-vector`)
  - **维度**: 默认 1536(对应 OpenAI text-embedding-3-small),可在 `args` 中改 `--dim`
  - **数据库**: 默认 `${WORKSPACE_DIR}/vector.db`,自动创建
- `prts-workspace` —— 自研 Workspace MCP server,暴露 `~/.prts/workspace/*.md` 给 MCP client。提供 `prts-workspace__list_documents`(列文档)、`prts-workspace__read_document`(读文档)、`prts-workspace__write_document`(写文档)和 `prts-workspace__search_documents`(搜索文档)四个工具。安装方式:已在 uv workspace 中,执行 `uv sync --all-packages` 后 `prts-workspace` 命令即加入 PATH。seed 默认 `disabled: true`。

## P8 —— 长上下文管理与记忆管理

PRTS Agent 采用**三层记忆架构**管理长对话上下文:

### 三层记忆

1. **短期上下文** (Layer 1): 最近 20 条消息,保证多轮工具调用连贯性
2. **中期记忆** (Layer 2): 每 10 轮自动总结一次,生成"记忆卡片"注入 system prompt
3. **长期向量** (Layer 3): 自动 embedding + sqlite-vec 召回,跨 session 关联

### Token 自报告

P8 起 LLM 客户端解析响应中的 `usage` 字段,让 Agent 知道实际消耗:
- OpenAI: `chunk.usage` 或 `chunk.choices[0].usage`
- Anthropic: `message_stop` 事件中的 `usage`
- 其他: 通过 `LLM_CONTEXT_LIMIT` 环境变量覆盖

### 动态预算

根据历史 usage 模式自动调整 headroom:
- 平稳对话: 85% headroom
- 中等增长: 75% headroom
- 工具密集: 65% headroom

### 重要性评分

截断时不是简单丢弃旧消息,而是基于内容重要性:
- 用户决策、关键事实: 高优先级保留
- 闲聊、重复确认: 低优先级丢弃
- 工具错误: 额外加分(调试需要)

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PRTS_CONTEXT_MODE` | `legacy`(旧模式) / `smart`(三层记忆) | `smart` |
| `PRTS_SUMMARY_INTERVAL` | 自动摘要触发轮数,0 关闭 | `10` |
| `LLM_CONTEXT_LIMIT` | 覆盖模型上下文限制 | 自动检测 |

## Web Dashboard

前端通过 WebSocket (`/ws/chat`) 与 Gateway 实时通信:

- **自动重连** —— 连接断开后指数退避重试(1s→2s→4s…最大30s),带重连次数指示
- **心跳保活** —— 每 30s 前端发 `ping`,Gateway 回 `pong`,防止中间件静默断连
- **平滑滚动** —— 新消息自动平滑滚动到底部
- **状态指示** —— 顶部状态栏显示 ● 连接中/就绪/生成中/重连中/已断开,并带颜色区分
- **会话恢复** —— 刷新页面后通过 `localStorage` 中的 `session_id` 恢复同一会话

## Telegram Bot (P5)

Gateway 内置 grammY bot,支持以下消息类型:

- **文本** (`message:text`) —— 直接转发给 Agent
- **图片** (`message:photo`) —— 下载到 `workspace/telegram/media/{chatId}/`,内容附加 `[文件: /path/to/photo.jpg]`
- **语音** (`message:voice`) —— 下载为 `.ogg`,内容标记 `[语音消息]`
- **音频** (`message:audio`) —— 下载音乐文件,保留标题/艺术家信息
- **文件** (`message:document`) —— 文本文件(`.txt` / `.md` / `.py` 等)自动内联内容预览,二进制文件仅保留路径
- **视频** (`message:video`) / **视频笔记** (`message:video_note`) —— 下载并附加路径

所有媒体下载共用 `PRTS_WORKSPACE_DIR` 作为根目录(默认 `~/.prts/workspace`)。

环境变量:
- `TELEGRAM_BOT_TOKEN` —— bot token(不设置则禁用)
- `BOT_MODE` —— `polling`(默认) 或 `webhook`

## Rust 守护

- `prts-watcher` —— 文件变更 + cron 触发(P6)
- `prts-vector` —— 向量检索 MCP server(P7)
- `prts-audio` —— 语音(P9)
