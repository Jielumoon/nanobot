import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import Consolidator, MemoryStore
from nanobot.agent.memory_documents import ClaimDraft
from nanobot.agent.memory_dreams import DreamBatch
from nanobot.agent.tools.context import RequestContext, ToolContext, request_context
from nanobot.agent.tools.loader import ToolLoader
from nanobot.agent.tools.memory_claims import (
    MemoryForgetTool,
    MemoryRememberTool,
    MemorySearchTool,
)
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.schema import Config, MemoryConfig
from nanobot.providers.base import GenerationSettings
from nanobot.utils.llm_runtime import LLMRuntime

_MEMORY_TOOLS = {
    "memory_remember", "memory_search", "memory_forget", "memory_project", "memory_explain",
}


def _store(tmp_path, **config) -> MemoryStore:
    return MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=True, **config))


def _loaded_tools(memory: MemoryStore | None, tmp_path) -> set[str]:
    registry = ToolRegistry()
    ctx = ToolContext(config=Config().tools, workspace=str(tmp_path), memory=memory)
    ToolLoader().load(ctx, registry)
    return set(registry.tool_names)


def test_memory_tools_load_only_when_claims_are_enabled(tmp_path):
    assert not _MEMORY_TOOLS & _loaded_tools(None, tmp_path)
    disabled = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=False))
    assert not _MEMORY_TOOLS & _loaded_tools(disabled, tmp_path)
    assert _MEMORY_TOOLS <= _loaded_tools(MemoryStore(tmp_path), tmp_path)
    assert _MEMORY_TOOLS <= _loaded_tools(_store(tmp_path), tmp_path)


async def test_remember_tool_records_turn_provenance(tmp_path):
    store = _store(tmp_path)
    tool = MemoryRememberTool(store)
    ctx = RequestContext(
        channel="cli", chat_id="direct", message_id="m-1", session_key="cli:direct",
    )

    with request_context(ctx):
        result = json.loads(await tool.execute(
            text="只喝手冲咖啡", kind="preference", core=True, quote="我只喝手冲",
        ))

    assert result["status"] == "saved"
    claim = store.claims.get(result["claim"]["claim_id"])
    assert (claim.session_key, claim.message_id, claim.source_file) == (
        "cli:direct", "m-1", "USER.md",
    )


async def test_remember_tool_rejects_unknown_kind(tmp_path):
    result = await MemoryRememberTool(_store(tmp_path)).execute(text="x", kind="mood")

    assert result.is_error


async def test_forget_tool_redacts_history_journal(tmp_path):
    store = _store(tmp_path)
    remembered = store.claims.remember(ClaimDraft(
        text="养了一只叫 Luna 的猫", kind="fact", quote="我家猫叫 Luna",
    ))
    store.append_history("- [permanent] 养了一只叫 Luna 的猫 (user: \"我家猫叫 Luna\")")
    store.append_history("- [durable] 部署在 1 核 1G 机器")

    result = json.loads(
        await MemoryForgetTool(store).execute(claim_ids=[remembered.claim.claim_id]),
    )

    assert result["status"] == "forgotten"
    assert result["history_entries_redacted"] == 1
    contents = [entry["content"] for entry in store.read_unprocessed_history(0)]
    assert "Luna" not in contents[0] and "[已遗忘]" in contents[0]
    assert contents[1] == "- [durable] 部署在 1 核 1G 机器"
    assert "Luna" not in store.read_memory()


async def test_forget_tool_reports_ambiguous_query(tmp_path):
    store = _store(tmp_path)
    store.claims.remember(ClaimDraft(text="喜欢手冲咖啡", kind="preference", core=True))
    store.claims.remember(ClaimDraft(text="咖啡豆每月买一次", kind="fact"))

    result = json.loads(await MemoryForgetTool(store).execute(query="咖啡"))

    assert result["status"] == "ambiguous"
    assert len(result["candidates"]) == 2
    assert "手冲咖啡" in store.read_user()


async def test_search_tool_returns_sources_and_claim_ids(tmp_path):
    store = _store(tmp_path)
    remembered = store.claims.remember(ClaimDraft(text="住在上海", kind="fact"))

    result = json.loads(await MemorySearchTool(store).execute(query="上海"))

    assert result["results"] == [{
        "source": f"memory/bank/facts.md:{remembered.claim.source_line}",
        "claim_id": remembered.claim.claim_id,
        "kind": "fact",
        "text": "住在上海",
        "status": "active",
    }]


def _long_memory(store: MemoryStore) -> None:
    filler = "\n".join(f"- 项目笔记 {index}: 例行记录" for index in range(40))
    store.write_memory(f"# Memory\n\n## 项目\n{filler}\n- 生产数据库是 PostgreSQL 16\n")


def test_context_keeps_full_memory_below_injection_limit(tmp_path):
    builder = ContextBuilder(
        tmp_path,
        memory_config=MemoryConfig(claims_enabled=True, inject_max_chars=10_000),
    )
    builder.memory.write_memory("# Memory\n\n- 生产数据库是 PostgreSQL 16\n")

    prompt = builder.build_system_prompt()

    assert "# Memory\n\n- 生产数据库是 PostgreSQL 16\n" in prompt
    assert "MEMORY.md truncated" not in prompt


async def test_context_truncates_long_memory_and_injects_commitments_in_message(tmp_path):
    builder = ContextBuilder(
        tmp_path,
        memory_config=MemoryConfig(claims_enabled=True, inject_max_chars=200),
    )
    _long_memory(builder.memory)
    builder.memory.claims.remember(ClaimDraft(
        text="周五前提交周报", kind="commitment", due_at="2099-01-01",
    ))

    prompt = builder.build_system_prompt()

    assert "项目笔记 0" in prompt
    assert "PostgreSQL" not in prompt
    assert "MEMORY.md truncated" in prompt
    block = await MemorySearchTool(builder.memory).runtime_context_provider()(
        RequestContext(channel="cli", chat_id="direct"),
    )
    assert block is not None
    assert "周五前提交周报 (due 2099-01-01)" in block.content
    assert "周五前提交周报" not in prompt
    assert "<!-- claim:" not in prompt


def test_context_ignores_injection_limit_without_claims(tmp_path):
    builder = ContextBuilder(
        tmp_path, memory_config=MemoryConfig(claims_enabled=False, inject_max_chars=200),
    )
    _long_memory(builder.memory)

    assert "PostgreSQL" in builder.build_system_prompt()


def test_context_strips_claim_markers_and_explains_memory_tools(tmp_path):
    enabled = ContextBuilder(tmp_path, memory_config=MemoryConfig(claims_enabled=True))
    enabled.memory.claims.remember(ClaimDraft(text="只喝手冲咖啡", kind="preference", core=True))

    prompt = enabled.build_system_prompt()

    assert "- 只喝手冲咖啡" in prompt
    assert "<!-- claim:" not in prompt
    assert "memory_remember" in prompt
    disabled = ContextBuilder(tmp_path, memory_config=MemoryConfig(claims_enabled=False))
    assert "memory_remember" not in disabled.build_system_prompt()


async def test_recall_adds_matching_tail_entries_for_long_memory(tmp_path):
    store = _store(tmp_path, inject_max_chars=200)
    _long_memory(store)
    provider = MemorySearchTool(store).runtime_context_provider()
    assert provider is not None

    block = await provider(RequestContext(
        channel="cli", chat_id="direct", original_user_text="我们的生产数据库用的什么版本 PostgreSQL？",
    ))

    assert block is not None and block.source == "memory_recall"
    assert "PostgreSQL 16" in block.content
    assert "项目笔记 0" not in block.content


async def test_recall_searches_bank_with_short_memory_and_no_injection_limit(tmp_path):
    store = _store(tmp_path)
    store.write_memory("# Memory\n- 住在北京\n")
    store.claims.remember(ClaimDraft(text="只喝手冲咖啡", kind="preference"))
    block = await MemorySearchTool(store).runtime_context_provider()(
        RequestContext(channel="cli", chat_id="direct", original_user_text="咖啡"),
    )
    assert block is not None and "只喝手冲咖啡" in block.content


def test_dream_gets_claim_tools_and_protects_managed_files(tmp_path):
    store = _store(tmp_path, daily_notes=True)

    tools = store.build_dream_tools(DreamBatch("", 0, {}))

    assert _MEMORY_TOOLS <= set(tools.tool_names)
    write = tools.get("write_file")
    assert write is not None
    assert store.daily.dir not in write._extra_write_allowed_dirs


def test_dream_prompt_uses_redacted_history_without_forgotten_content(tmp_path):
    store = _store(tmp_path, daily_notes=True)
    claim = store.claims.remember(ClaimDraft(text="养了一只猫", kind="fact"))
    store.append_history("养了一只猫")
    store.forget_memory(claim_ids=[claim.claim.claim_id])
    store.append_history("- [permanent] 只喝手冲咖啡")

    result = store.build_dream_prompt()

    assert result is not None
    prompt = result.prompt
    assert "## Structured memory" in prompt
    assert "memory/daily/YYYY-MM-DD.md" in prompt
    assert "养了一只猫" not in prompt
    assert "[h:2:1]" in prompt
    assert "只喝手冲咖啡" in prompt


def test_dream_prompt_unchanged_when_structured_memory_is_off(tmp_path):
    store = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=False, daily_notes=False))
    store.append_history("- [permanent] 只喝手冲咖啡")

    result = store.build_dream_prompt()

    assert result is not None
    assert result.prompt == (
        f"{MemoryStore.default_dream_prompt(claims_enabled=False)}\n\n## Conversation History\n"
        f"[h:1] [{store.read_unprocessed_history(0)[0]['timestamp']}] (conversation summary)\n"
        "[h:1:1] - [permanent] 只喝手冲咖啡"
    )


def test_dream_content_diff_includes_daily_notes(tmp_path):
    store = MemoryStore(tmp_path, config=MemoryConfig(daily_notes=True))
    store.git.init()

    store.append_daily_summary("- [ephemeral] 正在实现记忆第一版", session_key="cli:direct")

    assert "memory/daily/" in store.dream_content_diff()


@pytest.mark.parametrize("daily_notes", [False, True])
async def test_archive_summary_reaches_daily_note_only_when_enabled(tmp_path, daily_notes):
    store = MemoryStore(tmp_path, config=MemoryConfig(daily_notes=daily_notes))
    provider = MagicMock(aclose=AsyncMock())
    provider.chat_stream_with_retry = AsyncMock(return_value=MagicMock(
        content="- [ephemeral] 正在实现记忆第一版", finish_reason="stop",
    ))
    provider.generation = GenerationSettings(max_tokens=100)
    runtime = LLMRuntime.capture(provider, "test-model", context_window_tokens=1000)
    consolidator = Consolidator(
        store=store,
        sessions=MagicMock(),
        build_messages=MagicMock(return_value=[]),
        get_tool_definitions=MagicMock(return_value=[]),
    )
    messages = [{"role": "user", "content": "继续"}, {"role": "assistant", "content": "好"}]

    await consolidator.archiver.archive(
        messages,
        runtime=runtime,
        session_key="cli:direct",
        history=[{"role": "system", "content": "system"}, *messages],
        request_tools=[],
    )

    note = tmp_path / "memory" / "daily"
    if daily_notes:
        (path,) = note.glob("*.md")
        assert "- 正在实现记忆第一版" in path.read_text(encoding="utf-8")
    else:
        assert not note.exists()


async def test_context_keeps_overdue_commitment_without_core_memory(tmp_path):
    from datetime import datetime, timezone

    builder = ContextBuilder(
        tmp_path, timezone="Asia/Shanghai",
        clock=lambda: datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc),
        memory_config=MemoryConfig(claims_enabled=True),
    )
    tool = MemoryRememberTool(builder.memory)
    result = json.loads(await tool.execute(text="完成周报", kind="commitment", due_at="2030-05-03"))
    assert result["claim"]["due_at"] == "2030-05-03T15:59:59Z"
    await tool.execute(text="临时饮食偏好", kind="preference", core=True, expires_at="2030-05-02")
    prompt = builder.build_system_prompt()
    block = await MemorySearchTool(builder.memory).runtime_context_provider()(
        RequestContext(channel="cli", chat_id="direct"),
    )
    assert block is not None
    assert "完成周报" in block.content and "已逾期" in block.content
    assert "完成周报" not in prompt
    assert "临时饮食偏好" not in prompt
    assert not builder.memory.memory_file.exists()


async def test_search_can_inspect_expired_history(tmp_path):
    from datetime import datetime, timezone

    store = MemoryStore(
        tmp_path, clock=lambda: datetime(2030, 5, 3, tzinfo=timezone.utc),
        config=MemoryConfig(claims_enabled=True),
    )
    store.claims.remember(ClaimDraft(text="临时工位在三楼", kind="circumstance", expires_at="2030-05-01"))
    store.claims.expire_due()
    tool = MemorySearchTool(store)
    assert json.loads(await tool.execute(query="工位"))["results"] == []
    hits = json.loads(await tool.execute(query="工位", include_history=True))["results"]
    assert hits[0]["text"] == "临时工位在三楼" and hits[0]["status"] == "expired"


@pytest.mark.parametrize("default_memory", [False, True])
def test_context_omits_empty_memory_section_without_commitments(tmp_path, default_memory):
    from nanobot.utils.helpers import load_bundled_template

    builder = ContextBuilder(tmp_path, memory_config=MemoryConfig(claims_enabled=True))
    content = load_bundled_template("memory/MEMORY.md") if default_memory else ""
    builder.memory.write_memory(content)
    assert builder.build_system_prompt() == builder.build_system_prompt(include_memory=False)


async def test_forget_rejects_one_character_query_even_with_id(tmp_path):
    store = _store(tmp_path)
    saved = store.claims.remember(ClaimDraft(text="养了一只猫", kind="fact"))
    result = await MemoryForgetTool(store).execute(claim_ids=[saved.claim.claim_id], query="猫")
    assert result.is_error
    assert store.claims.get(saved.claim.claim_id).status == "active"
