# PRTS dev runner — 一键起 dashboard / gateway / agent(Windows PowerShell 原生)
$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

# 加载 .env(简易解析,只支持 KEY=VALUE)
if (Test-Path .env) {
    Get-Content .env | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*?)\s*=\s*(.*)\s*$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
}

# uv 不在 PATH 时回退 python -m uv
$uvCmd = if (Get-Command uv -ErrorAction SilentlyContinue) { "uv" } else { "python -m uv" }

$dashboardPort = if ($env:DASHBOARD_PORT) { $env:DASHBOARD_PORT } else { "5173" }
$gatewayPort   = if ($env:GATEWAY_PORT)   { $env:GATEWAY_PORT   } else { "4787" }
$agentPort     = if ($env:AGENT_PORT)     { $env:AGENT_PORT     } else { "4788" }

Write-Host "[dev] dashboard: http://localhost:$dashboardPort" -ForegroundColor Blue
Write-Host "[dev] gateway  : http://localhost:$gatewayPort"   -ForegroundColor Green
Write-Host "[dev] agent    : http://localhost:$agentPort"     -ForegroundColor Yellow

npx -y concurrently `
    -n dashboard,gateway,agent `
    -c blue,green,yellow `
    "pnpm --filter dashboard dev" `
    "pnpm --filter gateway dev" `
    "$uvCmd run --project apps/agent prts-agent"
