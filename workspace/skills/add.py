"""默认 skill —— 加法。

PRTS Agent 启动时会自动加载本目录下所有非 `_` 开头的 .py;
LLM 看到 `add` 工具后,可以在被问"算 2+3"之类时直接调用。
"""

from prts import skill


@skill(description="把两个整数相加,返回结果。适合算术问答。")
async def add(a: int, b: int) -> int:
    return a + b


@skill(description="把两个数(整数或小数)相乘。")
async def multiply(a: float, b: float) -> float:
    return a * b
