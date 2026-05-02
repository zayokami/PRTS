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

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "text-embedding-3-small"


class EmbeddingClient:
    """轻量 embedding 客户端。

    不依赖 ``openai`` Python SDK(避免版本耦合),直接用 ``httpx`` 发 POST,
    只解析 ``data[0].embedding`` 字段。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = (base_url or os.getenv("EMBEDDING_BASE_URL", "")).rstrip("/")
        if not self._base_url:
            # 默认复用 LLM 端点(绝大多数 OpenAI 兼容 provider 的 embedding 和
            # chat 在同一个 base_url 下)
            self._base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self._api_key = api_key or os.getenv("EMBEDDING_API_KEY", "") or os.getenv("LLM_API_KEY", "")
        self._model = model or os.getenv("EMBEDDING_MODEL", _DEFAULT_MODEL)
        self._client = httpx.AsyncClient(timeout=30.0)

    async def embed(self, text: str) -> list[float]:
        """把单条文本转成 float 向量。"""
        url = f"{self._base_url}/embeddings"
        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        resp = await self._client.post(
            url,
            headers=headers,
            json={"input": text, "model": self._model},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量 embedding(比单条循环快,单次 HTTP 往返)。"""
        if not texts:
            return []
        url = f"{self._base_url}/embeddings"
        headers: dict[str, str] = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"

        resp = await self._client.post(
            url,
            headers=headers,
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        items = data["data"]
        # 按 index 排序,因为 provider 可能乱序返回
        items.sort(key=lambda x: x["index"])
        return [item["embedding"] for item in items]


def build_embedding_client() -> EmbeddingClient | None:
    """工厂函数。如果环境变量完全没配 embedding 相关信息,返回 None,
    让调用方知道"向量检索未启用"。
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
    return EmbeddingClient()
