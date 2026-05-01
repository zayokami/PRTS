"""演示 prts.client.notify —— 反向给当前会话推消息。

LLM 调 ``ping`` 工具后,前端会收到一条 notify 帧,UI 上以"通知"样式渲染。
"""

from prts import client, skill


@skill(description="发一条 notify 给当前会话,常用来做活检 / 演示反向通知。")
async def ping(message: str = "PRTS 在线") -> dict[str, str]:
    await client.notify(message, kind="info")
    return {"ok": "delivered", "message": message}
