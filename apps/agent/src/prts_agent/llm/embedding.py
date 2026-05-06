"""文本 embedding 客户端。

P7:把自然语言文本转成 float[] 向量,供 sqlite-vec 存储和检索。

默认用 OpenAI 兼容的 ``/v1/embeddings`` 端点(任意 provider:OpenAI/DeepSeek/
Ollama/...)。向量维度由模型决定(如 text-embedding-3-small 是 1536)。
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "text-embedding-3-small"


class EmbeddingClient:
    """轻量 embedding 客户端,可选 LRU+TTL 缓存避免重复 API 调用。

    不依赖 ``openai`` Python SDK(避免版本耦合),直接用 ``httpx`` 发 POST,
    只解析 ``data[0].embedding`` 字段。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("EMBEDDING_BASE_URL", "")).rstrip("/")
        if not self._base_url:
            # 默认复用 LLM 端点(绝大多数 OpenAI 兼容 provider 的 embedding 和
            # chat 在同一个 base_url 下)
            self._base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key or os.getenv("EMBEDDING_API_KEY", "") or os.getenv("LLM_API_KEY", "")
        self._model = model or os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)
        self._client = httpx.AsyncClient(timeout=30.0)
        self._cache = cache  # 可选;None 时禁用缓存

    @property
    def cache_stats(self) -> dict[str, Any] | None:
        """返回缓存统计。如果未启用缓存返回 None。"""
        if self._cache is None:
            return None
        return self._cache.stats

    async def close(self) -> None:
        """关闭底层 HTTP 连接池。Agent shutdown / reload 时调用,防连接泄漏。"""
        await self._client.aclose()

    async def embed(self, text: str) -> list[float]:
        """把单条文本转成 float 向量。启用缓存时会先查缓存。"""
        # ---- 1. 查缓存 ----
        if self._cache is not None:
            cached = await self._cache.get(text)
            if cached is not None:
                return cached

        # ---- 2. 发请求 ----
        url = f"{self._base_url}/embeddings"
        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        try:
            resp = await self._client.post(
                url,
                headers=headers,
                json={"input": text, "model": self._model},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                # Provider 不支持 embedding API (如 DeepSeek),静默跳过
                logger.warning(
                    "embedding API not available at %s (404). "
                    "Vector memory will be disabled.",
                    url,
                )
                return []
            raise
        data: dict[str, Any] = resp.json()
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)

        # ---- 3. 写缓存 ----
        if self._cache is not None:
            await self._cache.set(text, embedding)

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding(比单条循环快,单次 HTTP 往返)。

        启用缓存时会先批量查缓存,只请求未命中的文本。
        """
        if not texts:
            return []

        # ---- 1. 批量查缓存 ----
        if self._cache is not None:
            results, misses = await self._cache.get_batch(texts)
            if not misses:
                # 全部命中
                # results 中 None 已经被替换了,但 type checker 不知道,
                # 下面做一次断言
                assert all(r is not None for r in results)
                return results  # type: ignore[return-value]
        else:
            results: list[list[float] | None] = [None] * len(texts)
            misses = list(texts)

        # ---- 2. 只请求未命中的 ----
        url = f"{self._base_url}/embeddings"
        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        resp = await self._client.post(
            url,
            headers=headers,
            json={"input": misses, "model": self._model},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        items = data["data"]
        # 按 index 排序,因为 provider 可能乱序返回
        items.sort(key=lambda x: x["index"])
        fresh_vectors = [item["embedding"] for item in items]

        # ---- 3. 写缓存并回填结果 ----
        if self._cache is not None:
            await self._cache.set_batch(misses, fresh_vectors)

        # 把 fresh_vectors 回填到 results 中 None 的位置
        miss_idx = 0
        for i, r in enumerate(results):
            if r is None:
                results[i] = fresh_vectors[miss_idx]
                miss_idx += 1

        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]


def build_embedding_client() -> EmbeddingClient | None:
    """工厂函数。如果环境变量完全没配 embedding 相关信息,返回 None,
    让调用方知道"向量检索未启用"。

    默认启用 LRU+TTL 缓存(max_entries=1000, ttl=3600s)。
    """
    has_url = os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_BASE_URL")
    has_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
    if not has_url and not has_key:
        # 开发/测试环境可能没配 key,embedding 功能静默禁用
        logger.warning(
            "embedding client not configured (set EMBEDDING_BASE_URL/LLM_BASE_URL "
            "or EMBEDDING_API_KEY/LLM_API_KEY to enable vector memory)"
        )
        return None

    # 从环境变量读取缓存配置
    cache_max = int(os.getenv("EMBEDDING_CACHE_MAX", "1000"))
    cache_ttl = float(os.getenv("EMBEDDING_CACHE_TTL", "3600"))
    cache = EmbeddingCache(max_entries=cache_max, ttl_seconds=cache_ttl)

    return EmbeddingClient(cache=cache)
