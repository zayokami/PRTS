# mcp-servers/prts-workspace

> 自研 MCP server,把 PRTS 的 workspace markdown(默认 `~/.prts/workspace/*.md`)
> 暴露给 MCP client。**P4 阶段实装**。

## 提供的工具

| 工具(prefixed) | 说明 |
|---|---|
| `prts-workspace__list_documents` | 列 workspace 内所有 `.md`(可选 `prefix` 过滤) |
| `prts-workspace__read_document` | 读取 `.md`,返回 `{"ok": true, "content": "..."}` |
| `prts-workspace__write_document` | 覆盖写入 `.md`,缺父目录自动建 |
| `prts-workspace__search_documents` | 文件名 + 内容大小写不敏感搜索,每条带最多 3 行片段 |

返回值统一 JSON 字符串(MCP `text` content),便于 LLM 直接解析。

## 安全

- `..` 越界、绝对路径 → 在 `_safe_path` 阶段抛 `ValueError`,工具返回 `{"ok": false, "error": ...}`,**不会**落到 workspace 之外。
- workspace 不存在时 `list_documents` 返回空,不报错(冷启动友好)。

## 安装

已是 uv workspace 成员(根 `pyproject.toml` 的 `[tool.uv.workspace]`)。
项目根执行:

```bash
uv sync --all-packages
```

之后:

- `prts-workspace` 命令进 `.venv/Scripts/`(Win)或 `.venv/bin/`(POSIX)
- Agent 的 `MCPManager._resolve_command` 会自动从 venv scripts 兜底找到它,
  即使该目录不在系统 PATH 上

## Agent 接入

`workspace/mcp.json` 默认已 seed 一个 `prts-workspace` entry(`disabled: true`)。
启用方式:

```json
{
  "mcpServers": {
    "prts-workspace": {
      "command": "prts-workspace",
      "env": { "PRTS_WORKSPACE_DIR": "${WORKSPACE_DIR}" },
      "disabled": false
    }
  }
}
```

`${WORKSPACE_DIR}` 由 `mcp/config.py::expand_variables` 展开成当前 workspace 绝对路径。

启动后 `GET /agent/v1/mcp/servers` 应看到 status=ready,4 个工具注册到 registry。

## 直接 stdio 启动(调试用)

```bash
PRTS_WORKSPACE_DIR=~/.prts/workspace prts-workspace
```

或不安装直接跑源码:

```bash
python mcp-servers/prts-workspace/src/prts_workspace_mcp/server.py
```

stdin/stdout 走 MCP JSON-RPC,可用 `mcp` CLI 或自家 `MCPManager` 接。

## smoke 测试

```bash
.venv/Scripts/python.exe scripts/smoke_p4_workspace.py
```

覆盖:list / read / write / search 全工具 + 越界路径拒绝 +
`mcp.json` 风格的 `command` 名 lookup(回归 P4 Bug 1)。
