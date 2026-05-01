"""范例 skill —— 演示 @skill 装饰器写法。

文件名以 ``_`` 开头(在 _examples 目录里),PRTS 加载器会跳过,不会真的注册。
拷贝到上一级目录并改名,即可被加载。
"""

from prts import skill


@skill(description="把两个整数相加,返回结果(范例 skill)")
async def add(a: int, b: int) -> int:
    return a + b


@skill(description="返回当前 PRTS 版本(范例 skill)")
async def version() -> str:
    return "PRTS 0.1.0 (P0)"
