#!/usr/bin/env bash
# PRTS dev runner — 一键起 dashboard / gateway / agent 三个进程
# 跨平台:macOS / Linux / Windows Git Bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 加载 .env(如果存在)
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# uv 不在 PATH 时回退 python -m uv
if command -v uv >/dev/null 2>&1; then
  UV_CMD="uv"
else
  UV_CMD="python -m uv"
fi

echo "[dev] dashboard: http://localhost:${DASHBOARD_PORT:-5173}"
echo "[dev] gateway  : http://localhost:${GATEWAY_PORT:-4787}"
echo "[dev] agent    : http://localhost:${AGENT_PORT:-4788}"

exec npx -y concurrently \
  -n dashboard,gateway,agent \
  -c blue,green,yellow \
  "pnpm --filter dashboard dev" \
  "pnpm --filter gateway dev" \
  "$UV_CMD run --project apps/agent prts-agent"
