# HEARTBEAT.md

> Agent 周期性自检 / 主动行为节奏。**P6 阶段接通 cron 后生效**。

## 默认节奏(草案)

- 工作日早 08:00 —— morning brief(天气、日程、待办)
- 工作日晚 22:00 —— evening recap(今日完成 / 明日预告)
- 每小时整点 —— 静默心跳,只更新内部状态

> 真正的实现在 `workspace/skills/*.py` 里用 `@task(cron=...)` 写。
