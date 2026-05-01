# PRTS

PRTS is a "PRTS Rebuild Terminal Service".

## 这是什么

- **Web Dashboard + Telegram Bot** 入口,后端 Agent 通过 LLM + MCP 工具执行真实任务并回写结果
- **Python 是 PRTS 的脚本语言**:在 `workspace/skills/*.py` 里写 `@skill` / `@task`,PRTS 自动加载并暴露给 LLM
- **三语言 monorepo**:TS 网关 + Python Agent + Rust 库
- **本地优先**:Markdown workspace + SQLite 落盘,可 git 版本控制;0 Docker 依赖

## 架构

```
[ Browser / Telegram ]
        ↓ WS / HTTP
[ TS Gateway :4787 ]            (Fastify + grammY)
        ↓ HTTP REST + SSE
[ Python Agent :4788 ]          (FastAPI + asyncio)
        ↓ MCP (stdio/HTTP)      ↓ 子进程 / IPC
[ MCP Servers ]                 [ Rust crates: vector / watcher / audio ]
```

## 快速开始

### 工具链

- Node.js >= 20、pnpm >= 9
- Python 3.12 + [uv](https://github.com/astral-sh/uv)(若 `uv` 不在 PATH,可用 `python -m uv` 替代)
- Rust 1.85+(rustup)

### 安装

```bash
pnpm install
python -m uv sync
cargo build
```

### 配置

```bash
cp .env.example .env
# 编辑 .env,至少填上 LLM_API_KEY
```

### 运行

```bash
# 一键起三个进程
pnpm dev
```

或者分别起:

```bash
pnpm --filter dashboard dev    # http://localhost:5173
pnpm --filter gateway dev      # http://localhost:4787
python -m uv run --project apps/agent prts-agent  # http://localhost:4788
```

健康检查:

```bash
curl http://localhost:4787/health
curl http://localhost:4788/health
```

> 注意:Windows 默认把 `8751-8850` 端口段排除给 Hyper-V/WSL,因此本项目避开 8787/8788
> 改用 **4787 / 4788**(可在 `.env` 改)。

## 仓库结构

```
PRTS/
├── apps/
│   ├── dashboard/      Web UI (React + Vite + TS)
│   ├── gateway/        TS Gateway (Fastify + grammY)
│   └── agent/          Python Agent (FastAPI)
├── packages/
│   └── prts-sdk/       Python SDK,提供 prts.* 脚本 API
├── crates/             Rust workspace
│   ├── prts-vector/    向量检索(sqlite-vec)
│   ├── prts-watcher/   文件 / cron 守护
│   └── prts-audio/     音频(P9)
├── mcp-servers/        自研 MCP server
├── workspace/          种子模板,首次启动复制到 ~/.prts/workspace
│   └── skills/         ★ 用户写 @skill / @task 的地方
└── scripts/            dev.sh / dev.ps1 一键启动
```

## 阶段进度

- [x] **P0** 仓库骨架(本阶段)
- [ ] **P1** 最小聊天闭环(Web ↔ Gateway ↔ Agent ↔ LLM)
- [ ] **P2** Markdown workspace + SQLite 持久化
- [ ] **P3** prts-sdk + Skills 装饰器
- [ ] **P4** MCP 外部接入
- [ ] **P5** Telegram Bot
- [ ] **P6** @task + Rust watcher
- [ ] **P7** 向量检索 + memory
- [ ] **P8** PRTS 主题化
- [ ] **P9** 语音(可选)

## 重要安全提醒

`workspace/skills/*.py` **等价于本机执行权限**。Agent 进程内 import,无沙箱。**不要跑陌生人的 .py文件！！！！！！！！！**

## 协议

MIT
