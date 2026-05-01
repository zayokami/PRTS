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

**已知 server**:

- `filesystem` —— 官方 [`@modelcontextprotocol/server-filesystem`](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem),读写本地文件(限制在传入的目录内)。seed 默认指向 `${WORKSPACE_DIR}`。
- (后续可加 `github` / `brave-search` / `prts-vector` 等)

## Rust 守护

- `prts-watcher` —— 文件变更 + cron 触发(P6)
- `prts-vector` —— 向量检索(P7)
- `prts-audio` —— 语音(P9)
