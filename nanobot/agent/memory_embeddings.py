"""Optional provider embeddings; Markdown and the claim index own validity."""

from __future__ import annotations

import json
import math
from array import array
from typing import TYPE_CHECKING

from loguru import logger
from openai import APIError, AsyncOpenAI, DefaultAsyncHttpxClient, omit

from nanobot.agent.memory_index import QueryEmbedding
from nanobot.config.network import NetworkConfig
from nanobot.config.schema import MemoryEmbeddingConfig, ProviderConfig, ProvidersConfig
from nanobot.providers.registry import find_by_name

if TYPE_CHECKING:
    from nanobot.agent.memory_claims import ClaimStore


class InvalidEmbeddingResponseError(ValueError):
    """The external service returned unusable vectors or input indexes."""


class MemoryEmbeddings:
    def __init__(
        self, claims: ClaimStore, config: MemoryEmbeddingConfig,
        providers: ProvidersConfig, network: NetworkConfig,
    ) -> None:
        provider = getattr(providers, config.provider, None)
        if not isinstance(provider, ProviderConfig):
            raise ValueError(f"记忆向量配置无效：providers.{config.provider} 不存在")
        if not provider.api_key:
            raise ValueError(f"记忆向量配置无效：请设置 providers.{config.provider}.apiKey")
        spec = find_by_name(config.provider)
        base = provider.api_base or (spec.default_api_base if spec else None)
        if not base and config.provider == "openai":
            base = "https://api.openai.com/v1"
        if not base:
            raise ValueError(f"记忆向量配置无效：请设置 providers.{config.provider}.apiBase")
        self.claims = claims
        self.config = config
        self.provider = provider
        self.api_base = base
        self.proxy = network.resolve_proxy(provider.use_proxy)
        self.fingerprint = json.dumps(
            [config.provider, config.model, config.dimensions], separators=(",", ":"),
        )

    def _provider_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key=self.provider.api_key, base_url=self.api_base,
            default_headers=self.provider.extra_headers, default_query=self.provider.extra_query,
            http_client=DefaultAsyncHttpxClient(proxy=self.proxy, trust_env=False),
            max_retries=0, timeout=30, _strict_response_validation=True,
        )

    async def _embed(self, texts: list[str]) -> list[array[float]] | None:
        try:
            async with self._provider_client() as client:
                response = await client.embeddings.create(
                    model=self.config.model, input=texts, encoding_format="float",
                    dimensions=self.config.dimensions if self.config.dimensions is not None else omit,
                )
            items = sorted(response.data, key=lambda item: item.index)
            if [item.index for item in items] != list(range(len(texts))):
                raise InvalidEmbeddingResponseError("向量响应与输入条数不一致")
            dimensions = self.config.dimensions or len(items[0].embedding)
            vectors: list[array[float]] = []
            for item in items:
                if len(item.embedding) != dimensions:
                    raise InvalidEmbeddingResponseError("向量响应维度不一致")
                norm = math.sqrt(sum(value * value for value in item.embedding))
                if not math.isfinite(norm) or norm == 0:
                    raise InvalidEmbeddingResponseError("向量响应必须包含有限的非零向量")
                vectors.append(array("f", (value / norm for value in item.embedding)))
            return vectors
        except (APIError, json.JSONDecodeError, InvalidEmbeddingResponseError) as exc:
            # Do not log request text, headers, or provider response bodies.
            logger.warning("记忆向量服务暂不可用（{}），保留关键词检索", type(exc).__name__)
            return None

    async def query(self, text: str) -> QueryEmbedding | None:
        vectors = await self._embed([text])
        return (self.fingerprint, vectors[0]) if vectors is not None else None

    async def backfill(self) -> int:
        with self.claims.lock:
            self.claims.sync_index()
            pending = self.claims.index.pending_vectors(self.fingerprint, self.config.batch_size)
        if not pending:
            return 0
        vectors = await self._embed([text for _, _, text in pending])
        if vectors is None:
            return 0
        with self.claims.lock:
            self.claims.sync_index()
            return self.claims.index.save_vectors(self.fingerprint, pending, vectors)
