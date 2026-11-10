"""Provider calls are mocked; lifecycle, SQLite, ranking and runtime context are real."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import threading
from array import array
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APIConnectionError, omit
from openai.types import CreateEmbeddingResponse

from nanobot.agent import memory_embeddings
from nanobot.agent.dream import run_dream
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_documents import ClaimDraft
from nanobot.agent.memory_recall import MemoryRuntimeContext
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.memory_claims import MemorySearchTool
from nanobot.bus.events import OutboundMessage
from nanobot.config.network import NetworkConfig
from nanobot.config.schema import MemoryConfig, MemoryEmbeddingConfig, ProvidersConfig

NOW = datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc)


def embedding_response(vectors):
    return CreateEmbeddingResponse.model_validate({
        "data": [{"object": "embedding", "index": index, "embedding": vector}
                 for index, vector in reversed(list(enumerate(vectors)))],
        "model": "test-model", "object": "list",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    })


@pytest.fixture
def api(monkeypatch):
    state = SimpleNamespace(vectors={}, default=[3.0, 4.0])

    async def embed(**kwargs):
        return embedding_response([state.vectors.get(text, state.default)
                                   for text in kwargs["input"]])

    state.create = AsyncMock(side_effect=embed)
    client = AsyncMock()
    client.__aenter__.return_value = SimpleNamespace(
        embeddings=SimpleNamespace(create=state.create),
    )
    state.client = MagicMock(return_value=client)
    state.http = MagicMock()
    state.warning = MagicMock()
    monkeypatch.setattr(memory_embeddings, "AsyncOpenAI", state.client)
    monkeypatch.setattr(memory_embeddings, "DefaultAsyncHttpxClient", state.http)
    monkeypatch.setattr(memory_embeddings.logger, "warning", state.warning)
    return state


def make_store(tmp_path, *, batch_size=2, provider="openai", model="test-model", dimensions=2):
    return MemoryStore(
        tmp_path, clock=lambda: NOW,
        config=MemoryConfig(embedding=MemoryEmbeddingConfig(
            enabled=True, provider=provider, model=model, dimensions=dimensions,
            batch_size=batch_size,
        )),
        providers=ProvidersConfig.model_validate({provider: {
            "apiKey": "embedding-test-key", "apiBase": "https://vectors.example/v1",
        }}),
    )


@pytest.fixture
def store(tmp_path, api):
    return make_store(tmp_path)


def vector_rows(store):
    with store.claims.index.connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM vectors ORDER BY claim_id")]


def agent_for(store):
    return SimpleNamespace(
        context=SimpleNamespace(memory=store),
        sessions=SimpleNamespace(state=SimpleNamespace(prune_dream_sessions=AsyncMock())),
        process_direct=AsyncMock(return_value=OutboundMessage(
            channel="cli", chat_id="direct", content="完成", metadata={"_stop_reason": "completed"},
        )),
        dream_runtime=lambda: None,
    )


async def test_memory_embedding_disabled_makes_zero_api_calls(tmp_path, api):
    store = MemoryStore(tmp_path)
    store.claims.remember(ClaimDraft(text="喜欢手冲咖啡", kind="preference"))
    assert "喜欢手冲咖啡" in await MemorySearchTool(store).execute("咖啡")
    block = await MemoryRuntimeContext(store, store.claims).provide(RequestContext(
        channel="cli", chat_id="direct", original_user_text="咖啡",
    ))
    assert block is not None and "喜欢手冲咖啡" in block.content
    await run_dream(agent_for(store), trigger="manual")
    api.client.assert_not_called()
    api.create.assert_not_awaited()
    assert vector_rows(store) == []


@pytest.mark.parametrize("trigger", ["manual", "scheduled"])
@pytest.mark.parametrize("history", [False, True])
async def test_dream_backfills_a_bounded_batch_even_without_history(store, api, trigger, history):
    for number in range(3):
        store.claims.remember(ClaimDraft(text=f"有效事实 {number}", kind="fact"))
    store.claims.remember(ClaimDraft(
        text="已经过期的情况", kind="circumstance", expires_at="2030-05-01",
    ))
    if history:
        store.append_history("本批没有新记忆需要提取")
    api.create.assert_not_awaited()
    agent = agent_for(store)
    result = await run_dream(agent, trigger=trigger)
    assert result.status == ("completed" if history else "empty")
    assert len(api.create.call_args.kwargs["input"]) == 2
    assert api.create.await_count == 1
    assert len(vector_rows(store)) == 2
    assert agent.process_direct.await_count == int(history)
    await run_dream(agent, trigger=trigger)
    assert api.create.await_count == 2
    assert len(api.create.call_args.kwargs["input"]) == 1
    await run_dream(agent, trigger=trigger)
    assert api.create.await_count == 2
    rows = vector_rows(store)
    assert len(rows) == 3
    for row in rows:
        value = array("f")
        value.frombytes(row["vec"])
        assert row["dim"] == 2 and len(row["vec"]) == 8
        assert math.isclose(sum(item * item for item in value), 1, abs_tol=1e-6)


async def test_embedding_provider_uses_named_connection_headers_proxy_and_dimensions(tmp_path, api):
    providers = ProvidersConfig.model_validate({"private": {
        "apiKey": "test-key", "apiBase": "https://private.example/v1",
        "extraHeaders": {"X-Tenant": "test-tenant"}, "extraQuery": {"api-version": "v1"},
        "useProxy": True,
    }})
    store = MemoryStore(
        tmp_path, config=MemoryConfig(embedding=MemoryEmbeddingConfig(
            enabled=True, provider="private", model="private-embedding", dimensions=2,
        )), providers=providers, network=NetworkConfig(proxy="http://proxy.example:8080"),
    )
    await MemorySearchTool(store).execute("测试查询")
    options = api.client.call_args.kwargs
    assert options["api_key"] == "test-key"
    assert options["base_url"] == "https://private.example/v1"
    assert options["default_headers"] == {"X-Tenant": "test-tenant"}
    assert options["default_query"] == {"api-version": "v1"}
    assert options["max_retries"] == 0
    api.http.assert_called_once_with(proxy="http://proxy.example:8080", trust_env=False)
    assert api.create.call_args.kwargs == {
        "model": "private-embedding", "input": ["测试查询"], "dimensions": 2,
        "encoding_format": "float",
    }


async def test_enabling_default_embeddings_uses_provider_model_dimensions(tmp_path, api):
    store = MemoryStore(
        tmp_path, config=MemoryConfig(embedding=MemoryEmbeddingConfig(enabled=True)),
        providers=ProvidersConfig.model_validate({"openai": {"apiKey": "test-key"}}),
    )
    store.claims.remember(ClaimDraft(text="默认模型处理的事实", kind="fact"))
    assert await store.backfill_embeddings() == 1
    assert api.client.call_args.kwargs["base_url"] == "https://api.openai.com/v1"
    assert api.create.call_args.kwargs["model"] == "text-embedding-3-small"
    assert api.create.call_args.kwargs["dimensions"] is omit
    assert vector_rows(store)[0]["dim"] == 2


@pytest.mark.parametrize(("provider", "providers", "message"), [
    ("missing", ProvidersConfig(), "不存在"),
    ("openai", ProvidersConfig(), "apiKey"),
    ("private", ProvidersConfig.model_validate({"private": {"apiKey": "test"}}), "apiBase"),
])
def test_embedding_configuration_errors_are_explicit(tmp_path, api, provider, providers, message):
    with pytest.raises(ValueError, match=message):
        MemoryStore(tmp_path, providers=providers, config=MemoryConfig(
            embedding=MemoryEmbeddingConfig(enabled=True, provider=provider),
        ))
    api.create.assert_not_awaited()


async def test_embedding_search_fuses_lexical_and_semantic_results_in_worker(store, api, monkeypatch):
    api.vectors.update({
        "喜欢手冲咖啡": [0.8, 0.6], "早上只喝拿铁": [1.0, 0.0], "咖啡滤纸放在柜子": [0.0, 1.0],
        "咖啡": [1.0, 0.0], "morning drink": [1.0, 0.0],
    })
    for text in ("喜欢手冲咖啡", "早上只喝拿铁", "咖啡滤纸放在柜子"):
        store.claims.remember(ClaimDraft(text=text, kind="fact"))
    await store.backfill_embeddings()
    await store.backfill_embeddings()
    api.create.reset_mock()
    main_thread = threading.get_ident()
    workers = []
    scan = store.claims.index._vector_hits

    def record_scan(*args, **kwargs):
        workers.append(threading.get_ident())
        return scan(*args, **kwargs)

    monkeypatch.setattr(store.claims.index, "_vector_hits", record_scan)
    result = json.loads(await MemorySearchTool(store).execute("咖啡"))["results"]
    assert result[0]["text"] == "喜欢手冲咖啡"
    assert {hit["text"] for hit in result} == set(api.vectors) - {"咖啡", "morning drink"}
    api.create.assert_awaited_once()
    assert api.create.call_args.kwargs["input"] == ["咖啡"]
    assert workers and all(worker != main_thread for worker in workers)
    result = json.loads(await MemorySearchTool(store).execute("morning drink"))["results"]
    assert result[0]["text"] == "早上只喝拿铁"


async def test_semantic_runtime_recall_filters_scope_before_top_k(store, api):
    store.config.recall_limit = 1
    api.default = [1.0, 0.0]
    api.vectors["早餐只喝拿铁"] = [0.8, 0.6]
    for index in range(6):
        store.claims.remember(ClaimDraft(
            text=f"未提及人物的资料 {index}", kind="fact", person=f"人物 {index}", quote="已确认",
        ))
        store.claims.remember(ClaimDraft(text=f"稳定核心 {index}", kind="fact", core=True))
    store.claims.remember(ClaimDraft(text="已完成项目的决定", kind="decision", project="旧项目"))
    store.claims.update_project("旧项目", "completed")
    store.claims.remember(ClaimDraft(text="早餐只喝拿铁", kind="preference"))
    while await store.backfill_embeddings():
        pass
    api.create.reset_mock()
    block = await MemoryRuntimeContext(store, store.claims).provide(RequestContext(
        channel="cli", chat_id="direct", original_user_text="早饭饮品",
    ))
    assert block is not None and "早餐只喝拿铁" in block.content
    assert "未提及人物" not in block.content and "稳定核心" not in block.content
    assert "已完成项目" not in block.content
    api.create.assert_awaited_once()


@pytest.mark.parametrize("error", [
    APIConnectionError(request=httpx.Request("POST", "https://test/v1")),
    json.JSONDecodeError("Invalid JSON response", "not-json", 0),
])
async def test_embedding_api_failure_warns_and_preserves_lexical_search(store, api, error):
    store.claims.remember(ClaimDraft(text="喜欢手冲咖啡", kind="preference"))
    api.create.side_effect = error
    result = json.loads(await MemorySearchTool(store).execute("咖啡"))
    assert result["results"][0]["text"] == "喜欢手冲咖啡"
    assert await store.backfill_embeddings() == 0
    assert api.warning.call_count == 2 and not vector_rows(store)


@pytest.mark.parametrize("vectors", [[], [[0.0, 0.0]], [[1.0, 0.0, 0.0]]])
async def test_invalid_embedding_response_falls_back_at_api_boundary(store, api, vectors):
    store.claims.remember(ClaimDraft(text="有效的咖啡偏好", kind="preference"))
    api.create.side_effect = None
    api.create.return_value = embedding_response(vectors)
    assert "有效的咖啡偏好" in await MemorySearchTool(store).execute("咖啡")
    api.warning.assert_called_once()


async def test_internal_vector_query_errors_propagate(store, api, monkeypatch):
    def broken(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store.claims.index, "_vector_hits", broken)
    with pytest.raises(sqlite3.OperationalError, match="disk I/O error"):
        await MemorySearchTool(store).execute("咖啡")
    api.warning.assert_not_called()


async def test_vector_cache_invalidates_edited_text_but_retains_unchanged_neighbors(store, api):
    first = store.claims.remember(ClaimDraft(text="旧的数据库选择", kind="fact")).claim
    second = store.claims.remember(ClaimDraft(text="邻居事实保持不变", kind="fact")).claim
    await store.backfill_embeddings()
    before = {row["claim_id"]: row for row in vector_rows(store)}
    path = store.workspace / first.source_file
    path.write_text(path.read_text().replace("旧的数据库选择", "新的数据库选择"))
    store.claims.sync_index()
    assert vector_rows(store) == [before[second.claim_id]]
    await store.backfill_embeddings()
    assert api.create.call_args.kwargs["input"] == ["新的数据库选择"]
    after = {row["claim_id"]: row for row in vector_rows(store)}
    assert before[first.claim_id]["content_hash"] != after[first.claim_id]["content_hash"]
    assert before[second.claim_id] == after[second.claim_id]


async def test_vector_lifecycle_forgets_supersedes_expires_and_prunes_all_fingerprints(store, api):
    old = store.claims.remember(ClaimDraft(text="采用旧数据库", kind="fact")).claim
    secret = store.claims.remember(ClaimDraft(text="需要遗忘的地点", kind="fact")).claim
    await store.backfill_embeddings()
    alternative = make_store(store.workspace, model="other-model")
    await alternative.backfill_embeddings()
    assert len(vector_rows(store)) == 4
    store.claims.remember(ClaimDraft(
        text="采用新数据库", kind="fact", supersedes=(old.claim_id,),
    ))
    store.forget_memory(claim_ids=[secret.claim_id])
    assert vector_rows(store) == []
    expiring = store.claims.remember(ClaimDraft(
        text="暂时有效的安排", kind="circumstance", expires_at=(NOW + timedelta(hours=1)).isoformat(),
    )).claim
    await store.backfill_embeddings()
    store.claims.expire_due(now=NOW + timedelta(hours=2))
    assert expiring.claim_id not in {row["claim_id"] for row in vector_rows(store)}
    assert old.claim_id not in {row["claim_id"] for row in vector_rows(store)}


@pytest.mark.parametrize("operation", ["backfill", "search"])
async def test_forget_during_embedding_api_wait_never_restores_vectors_or_results(store, api, operation):
    saved = store.claims.remember(ClaimDraft(text="秘密地点", kind="fact")).claim
    if operation == "search":
        await store.backfill_embeddings()
    started = asyncio.Event()
    resume = asyncio.Event()

    async def wait_for_api(**kwargs):
        started.set()
        await resume.wait()
        return embedding_response([[1.0, 0.0] for _ in kwargs["input"]])

    api.create.side_effect = wait_for_api
    task = asyncio.create_task(
        store.backfill_embeddings() if operation == "backfill"
        else MemorySearchTool(store).execute("秘密地点")
    )
    await started.wait()
    try:
        await asyncio.wait_for(asyncio.to_thread(
            store.forget_memory, claim_ids=[saved.claim_id],
        ), timeout=2)
    finally:
        resume.set()
        result = await task
    assert vector_rows(store) == []
    if operation == "backfill":
        assert result == 0
    else:
        assert json.loads(result)["results"] == []


async def test_schema_one_upgrades_from_markdown_without_rewriting_metadata(store, api):
    claim = store.claims.remember(ClaimDraft(
        text="保留来源的事实", kind="fact", quote="这是原话",
    )).claim
    forgotten = store.claims.remember(ClaimDraft(text="不应重建的秘密", kind="fact")).claim
    path = store.workspace / claim.source_file
    original = path.read_bytes()
    store.forget_memory(claim_ids=[forgotten.claim_id])
    path.write_bytes(original)
    with store.claims.index.connect() as conn:
        conn.execute("DROP TABLE vectors")
        conn.execute("PRAGMA user_version=1")
    assert store.claims.get(claim.claim_id) == claim
    redacted = store.claims.get(forgotten.claim_id)
    assert redacted is not None and redacted.status == "retracted"
    assert redacted.text == "" and redacted.quote is None
    assert path.read_bytes() == original
    with store.claims.index.connect() as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert await store.backfill_embeddings() == 1


async def test_embedding_fingerprint_separates_model_provider_and_dimensions(store, api):
    store.claims.remember(ClaimDraft(text="需要索引的事实", kind="fact"))
    await store.backfill_embeddings()
    fingerprints = {store.embeddings.fingerprint}
    for provider, model, dimensions in (
        ("openai", "next-model", 2), ("private", "test-model", 2), ("openai", "test-model", 3),
    ):
        other = make_store(store.workspace, provider=provider, model=model, dimensions=dimensions)
        api.default = [1.0] * dimensions
        assert await other.backfill_embeddings() == 1
        fingerprints.add(other.embeddings.fingerprint)
    assert len(fingerprints) == 4
    assert {row["fingerprint"] for row in vector_rows(store)} == fingerprints
