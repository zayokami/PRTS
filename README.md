# PRTS

> PRTS (Public Random Tactical System) — 本地优先的个人 AI 助理，运行于 Windows 工作站，通过 Web Dashboard 与 Telegram Bot 等渠道接受指令。

## 架构

```
┌─────────────┐     WebSocket     ┌───────────┐      HTTP/SSE      ┌─────────┐
│  Dashboard  │ ◄──────────────► │  Gateway  │ ◄────────────────► │  Agent  │
│  (React)    │   /ws/chat        │ (Fastify) │   /agent/v1/*      │(FastAPI)│
│  :5173(dev) │                   │  :4787    │                    │  :4788  │
│  或由 Gateway │                   │           │                    │         │
│  生产托管    │                   │           │                    │         │
└─────────────┘                   └─────┬─────┘                    └────┬────┘
                                        │                               │
                                        │  Telegram Bot (grammy)         │
                                        │  Webhook / Polling             │
                                        ▼                               ▼
                                  ┌───────────┐                  ┌───────────┐
                                  │ Telegram  │                  │  SQLite   │
                                  │  Users    │                  │ (aiosqlite)│
                                  └───────────┘                  └───────────┘
                                                                 ┌───────────┐
                                                                 │  MCP      │
                                                                 │  Servers  │
                                                                 └───────────┘
```

**三进程架构**：
- **Agent** (`apps/agent/`, Python/FastAPI) — LLM 对话核心，工具执行，三层记忆（短期/中期摘要/长期向量），SQLite 持久化
- **Gateway** (`apps/gateway/`, TypeScript/Fastify) — WebSocket 网关，Telegram Bot 适配器，Dashboard 静态文件托管
- **Dashboard** (`apps/dashboard/`, React/Vite) — ChatGPT 风格聊天界面，侧边栏会话历史管理

## 快速开始

### 前置条件

- Python 3.12+
- Node.js 20+ / pnpm 9+
- Rust (可选，构建 `prts-vector` 向量记忆服务)

### 安装

```bash
# 克隆仓库
git clone https://github.com/zayokami/PRTS.git
cd PRTS

# 安装前端依赖
pnpm install

# 配置环境变量
cp .env.example apps/agent/.env
# 编辑 apps/agent/.env 填入 LLM_API_KEY
```

### 配置

`apps/agent/.env` 最小配置：

```env
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-pro
LLM_API_KEY=sk-...

AGENT_PORT=4788
GATEWAY_PORT=4787
DASHBOARD_PORT=5173
```

支持 OpenAI / DeepSeek / Anthropic / Ollama 等任意 OpenAI 兼容端点。

### 启动

**开发模式**（三进程并发，热重载）：

```bash
pnpm dev
```

Dashboard: http://localhost:5173 → Gateway (4787) → Agent (4788)

**生产模式**：

```bash
# 构建前端
pnpm build

# 启动 Agent
python -m uv run --project apps/agent prts-agent

# 启动 Gateway（托管 Dashboard 静态文件）
node apps/gateway/dist/main.js
```

Dashboard: http://localhost:4787/

### 单实例锁

Agent 和 Gateway 默认单实例运行。已有一个实例在运行时会拒绝启动。

设置 `PRTS_ALLOW_MULTIPLE=1` 可跳过检查（多实例调试用）。

## 功能

### 核心对话
- 流式 SSE 响应，支持 tool calling
- DeepSeek V4 `reasoning_content` 多轮对话传回
- 上下文窗口动态预算管理（`DynamicBudget`）
- Per-session `asyncio.Lock` 防止并发对话交叉写入

### 三层记忆
- **短期**：最近 N 条消息（默认 20），保证多轮连贯
- **中期**：对话摘要（`DialogueSummarizer`），每 10 轮自动触发
- **长期**：向量召回（`prts-vector` MCP），语义检索历史对话

### 工具系统
- `@skill` 注册用户脚本为 LLM 工具
- `@task` 注册定时任务（cron 调度）
- MCP 协议集成外部工具（filesystem / vector / 自定义）
- 工具钩子：pre/post/failure hooks，权限引擎（glob 规则），prompt injection 检测
- 结构化 `ToolResult`（`is_error` / `error_type`），JSON Schema 严格验证

### 会话管理
- `GET /sessions` — 会话列表（分页，按 `updated_at` 降序）
- `DELETE /sessions/:id` — 删除会话及其所有消息和摘要
- `PATCH /sessions/:id` — 重命名会话
- `GET /sessions/:id/history` — 会话历史
- 自动生成标题（首次对话取用户消息前 30 字符）

### Dashboard
- ChatGPT 风格界面（侧边栏 + 消息气泡 + 输入框）
- 侧边栏会话历史：切换 / 删除 / 重命名（双击编辑）
- WebSocket 指数退避重连 + 心跳 ping/pong
- lucide-react 图标库，Noto Sans SC 字体

### Telegram Bot
- 文本 / 图片 / 语音 / 文件 / 视频 消息支持
- Polling / Webhook 双模式
- 多媒体文件下载与文本预览

### 安全
- Session ID 格式验证（`^[\w-]{1,64}$`）
- 敏感端点 localhost-only（`/events/*`、`/mcp/reload`）
- 请求体大小限制（content max 100KB）
- Per-IP 滑动窗口限流（120 req/60s）
- MCP 配置 Pydantic 严格验证

## 项目结构

```
PRTS/
├── apps/
│   ├── agent/              # Python FastAPI Agent
│   │   ├── src/prts_agent/
│   │   │   ├── api/        # HTTP 路由
│   │   │   ├── llm/        # LLM 客户端 (OpenAI/Anthropic)
│   │   │   ├── loop/       # AgentLoop, ContextManager, Budget
│   │   │   ├── memory/     # SQLite, 摘要, 重要性评分
│   │   │   ├── mcp/        # MCP 客户端
│   │   │   ├── tools/      # 工具注册, hooks, 权限, 验证
│   │   │   ├── evals/      # 评估框架
│   │   │   └── main.py     # FastAPI 入口
│   │   └── pyproject.toml
│   ├── dashboard/          # React + Vite 前端
│   │   ├── src/App.tsx     # 主界面
│   │   └── package.json
│   └── gateway/            # TypeScript Fastify 网关
│       ├── src/main.ts     # WebSocket + Telegram + 静态托管
│       └── package.json
├── packages/
│   └── prts-sdk/           # Rust SDK (prts-vector, prts-workspace)
├── workspace/              # 用户工作区 (~/.prts/workspace)
│   ├── AGENTS.md           # Agent system prompt
│   ├── TOOLS.md            # 工具说明
│   ├── mcp.json            # MCP 服务配置
│   └── skills/             # 用户 .py 脚本
└── scripts/                # 开发脚本
```

## API 速查

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/agent/v1/converse` | 对话（SSE 流式） |
| `GET` | `/agent/v1/sessions` | 会话列表 |
| `GET` | `/agent/v1/sessions/:id/history` | 会话历史 |
| `PATCH` | `/agent/v1/sessions/:id` | 重命名会话 |
| `DELETE` | `/agent/v1/sessions/:id` | 删除会话 |
| `GET` | `/agent/v1/skills` | 已注册工具列表 |
| `GET` | `/agent/v1/sessions/:id/summaries` | 会话摘要 |
| `GET` | `/agent/v1/mcp/servers` | MCP 服务状态 |
| `POST` | `/agent/v1/mcp/reload` | 热重载 MCP |
| `POST` | `/agent/v1/mcp/health-check` | MCP 健康检查 |
| `GET` | `/health` | 健康检查 |

WebSocket: `ws://<host>/ws/chat?session_id=<id>`

## 重要安全提醒

`workspace/skills/*.py` **等于本机执行权限**。暂无沙箱机制。**不要运行陌生人的 .py 文件。**

## 协议

MIT License

## 免责声明

本项目和《明日方舟》中的 PRTS（Primitive Rhodes Island Terminal Service）无关，纯属重名。
