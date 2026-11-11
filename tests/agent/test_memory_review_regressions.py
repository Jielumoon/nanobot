"""User paths from the final memory review, including interrupted file operations."""

from __future__ import annotations

import asyncio
import json
import tracemalloc
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.dream import run_dream
from nanobot.agent.memory import Consolidator, MemoryStore
from nanobot.agent.memory_documents import ClaimDraft, digest
from nanobot.agent.memory_recall import MemoryRuntimeContext
from nanobot.agent.tools.memory_claims import MemoryForgetTool, MemoryRememberTool
from nanobot.bus.events import OutboundMessage
from nanobot.config.schema import MemoryConfig
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.utils.llm_runtime import LLMRuntime

NOW = datetime(2030, 5, 3, 12, tzinfo=timezone.utc)


@pytest.fixture
def builder(tmp_path):
    return ContextBuilder(tmp_path, clock=lambda: NOW)


async def test_forget_claim_revokes_paraphrased_history_evidence_and_restored_mirror(builder):
    store = builder.memory
    claims = store.claims
    summary = "- 用户常住沪上\n- 用户养了金鱼"
    cursor = store.append_history(summary, session_key="cli:source")
    store.append_daily_summary(summary, cursor=cursor)
    batch = store.build_dream_prompt()
    tools = store.build_dream_tools(batch)
    arguments = {"text": "常住上海", "kind": "fact", "source": f"h:{cursor}:1"}
    saved = json.loads(await tools.execute("memory_remember", arguments))["claim"]
    originals = claims.documents.snapshot()
    result = json.loads(await MemoryForgetTool(store).execute(claim_ids=[saved["claim_id"]]))
    assert result["history_entries_redacted"] == 1
    assert "沪上" not in store.history_file.read_text()
    assert "用户养了金鱼" in store.history_file.read_text()
    assert "已被遗忘" in await tools.execute("memory_remember", arguments)
    remaining = json.loads(await tools.execute("memory_remember", {
        "text": "养金鱼", "kind": "fact", "source": f"h:{cursor}:2",
    }))
    assert remaining["status"] == "saved"
    assert ("history", f"{cursor}:{digest(summary.splitlines()[0])}") in claims.documents.tombstones()
    claims.documents.apply(originals)
    claims.db_path.unlink()
    store.reconcile_files()
    assert "沪上" not in "\n".join(claims.documents.snapshot().values())
    assert not claims.search("沪上", include_history=True)
    # A fresh statement is a new source, not a semantic ban on the same words.
    fresh = json.loads(await MemoryRememberTool(store).execute("常住上海", "fact"))
    assert fresh["status"] == "saved" and fresh["claim"]["claim_id"] != saved["claim_id"]


@pytest.mark.parametrize("source", ["history", "daily"])
async def test_forget_summary_text_retracts_derived_claim_closure(builder, source):
    store = builder.memory
    summary = "- 用户常住沪上\n- 用户养了金鱼"
    cursor = store.append_history(summary)
    store.append_history("- 常住上海")
    batch = store.build_dream_prompt()
    tools = store.build_dream_tools(batch)
    saved = [json.loads(await tools.execute("memory_remember", arguments))["claim"] for arguments in (
        {"text": "常住上海", "kind": "fact", "source": f"h:{cursor}:1"},
        {"text": "住所位于黄浦江畔", "kind": "fact", "source": "h:2:1"},
    )]
    query = "沪上"
    if source == "daily":
        store.append_daily_summary(summary, cursor=cursor)
        daily = store.daily.path_for()
        # A user-edited mirror retains its original history provenance.
        daily.write_text(daily.read_text().replace("沪上", "申城"))
        query = "申城"

    result = json.loads(await MemoryForgetTool(store).execute(query=query, all_matches=True))

    assert result["status"] == "forgotten"
    assert {store.claims.get(claim["claim_id"]).status for claim in saved} == {"retracted"}
    assert not store.claims.search("上海", include_history=True)
    assert not store.claims.search("黄浦江", include_history=True)
    assert "沪上" not in store.history_file.read_text()
    assert "用户养了金鱼" in store.history_file.read_text()


async def test_forget_response_does_not_echo_removed_copy_into_current_chat(builder):
    store = builder.memory
    saved = store.claims.remember(ClaimDraft(text="门禁口令是蓝鲸松林", kind="fact")).claim
    store.daily.append("## 备注", ["- 门禁口令是蓝鲸松林"])
    response = await MemoryForgetTool(store).execute(claim_ids=[saved.claim_id])
    result = json.loads(response)
    assert result["status"] == "forgotten" and result["removed"]
    assert "蓝鲸松林" not in response
    assert result["removed"][0]["source"].startswith("memory/daily/")


async def test_forget_preupgrade_summary_claim_revokes_entry_without_fabricating_line(builder):
    store = builder.memory
    store.append_history("- 用户常住沪上\n- 不相关的金鱼记录")
    batch = store.build_dream_prompt()
    # The prior release persisted only this entry reference, with no line hashes.
    old = store.claims.remember(
        ClaimDraft(text="常住上海", kind="fact"), source_ref="history:1", source_kind="summary",
    ).claim
    await MemoryForgetTool(store).execute(claim_ids=[old.claim_id])
    result = await store.build_dream_tools(batch).execute("memory_remember", {
        "text": "常住上海", "kind": "fact", "source": "h:1:1",
    })
    assert "已被遗忘" in result
    assert "沪上" not in store.build_dream_prompt().prompt
    assert "不相关的金鱼记录" in store.history_file.read_text()


def test_forget_closes_transitive_quote_targets_before_removing_copies(builder):
    store = builder.memory
    claims = store.claims
    first = claims.remember(ClaimDraft(text="旧办公室代号蓝鲸", kind="fact")).claim
    second = claims.remember(ClaimDraft(
        text="办公室地址为山岚路", quote=first.text, kind="fact",
    ), source_ref="session:cli:one#2").claim
    third = claims.remember(ClaimDraft(
        text="办公门牌为四十二号", quote=second.text, kind="fact",
    )).claim
    store.daily.append("## 笔记", [f"- {second.text}", f"- {third.text}", "- 其他信息保留"])
    store.append_history(f"- {third.text}\n- 其他历史保留")
    result, _ = store.forget_memory(claim_ids=[first.claim_id])
    assert {claim.claim_id for claim in result.retracted} == {
        first.claim_id, second.claim_id, third.claim_id,
    }
    assert not claims.search("办公室", include_history=True)
    assert not claims.search("四十二", include_history=True)
    assert "其他信息保留" in store.daily.path_for().read_text()
    assert "其他历史保留" in store.history_file.read_text()


async def test_compact_dream_correction_keeps_daily_mirror_historical(builder):
    store = builder.memory
    summary = "- [durable] 早餐饮品选择拿铁"
    provider = MagicMock()
    provider.generation = GenerationSettings(max_tokens=512)
    provider.chat_stream_with_retry = AsyncMock(return_value=LLMResponse(content=summary))
    runtime = LLMRuntime.capture(provider, "test", context_window_tokens=8000)
    consolidator = Consolidator(
        store=store, sessions=MagicMock(), build_messages=builder.build_messages,
        get_tool_definitions=lambda: [],
    )
    transcript = [{"role": "user", "content": "我早餐选拿铁"}]
    await consolidator.archiver.archive(
        transcript, runtime=runtime, session_key="cli:test", history=transcript,
        request_tools=[], previous_summary=None,
    )
    batch = store.build_dream_prompt()
    tools = store.build_dream_tools(batch)
    old = json.loads(await tools.execute("memory_remember", {
        "text": "早餐饮品选择拿铁", "kind": "preference", "source": "h:1:1",
    }))["claim"]["claim_id"]
    new = json.loads(await MemoryRememberTool(store).execute(
        "早餐饮品改为手冲", "preference", supersedes=[old],
    ))["claim"]["claim_id"]
    messages = builder.build_messages([], "早餐饮品")
    assert "早餐饮品改为手冲" in messages[-1]["content"]
    assert "拿铁" not in json.dumps(messages, ensure_ascii=False)
    assert {hit.claim_id for hit in store.claims.search("早餐饮品")} == {new}
    archive = store.claims.search("拿铁", include_history=True)
    assert {hit.status for hit in archive} == {"superseded", "historical"}
    assert "<!-- history:1:" in store.daily.path_for().read_text()


def test_old_workspace_projection_rebuild_classifies_summary_without_editing_file(builder):
    claims = builder.memory.claims
    old_daily = "# 2029-01-01\n\n## 10:00 会话摘要 · cli:old\n- 旧地点在杭州\n"
    claims.documents.apply({"memory/daily/2029-01-01.md": old_daily})
    claims.sync_index()
    with claims.index.connect() as conn:
        # Version 2 indexed unmarked compaction mirrors as current ordinary lines.
        conn.execute("UPDATE chunks SET current=1")
        conn.execute("PRAGMA user_version=2")
    assert not claims.search("杭州")
    assert claims.search("杭州", include_history=True)[0].status == "historical"
    assert (builder.workspace / "memory/daily/2029-01-01.md").read_text() == old_daily


async def test_forget_sensitive_heading_survives_restore_and_hashtag_claim_is_valid(builder):
    store = builder.memory
    store.write_user("# 家庭住址：蓝鲸路42号\n\n## 爱好\n- 喜欢画画\n")
    original = store.read_user()
    response = json.loads(await MemoryForgetTool(store).execute(query="蓝鲸路", all_matches=True))
    assert response["status"] == "forgotten"
    store.write_user(original)
    assert "蓝鲸路" not in builder.build_system_prompt()
    store.reconcile_files()
    assert "蓝鲸路" not in store.read_user() and "喜欢画画" in store.read_user()
    saved = json.loads(await MemoryRememberTool(store).execute("#42 工单已经完成", "fact"))
    assert saved["status"] == "saved"
    hit = store.claims.search("工单")[0]
    assert hit.claim_id == saved["claim"]["claim_id"] and hit.text.startswith("#42")
    assert json.loads(await MemoryForgetTool(store).execute(
        claim_ids=[hit.claim_id],
    ))["status"] == "forgotten"


def test_person_gating_and_budget_cover_commitment_project_and_recall(builder):
    store = builder.memory
    claims = store.claims
    for number in range(8):
        claims.remember(ClaimDraft(
            text=f"小林私人事项{number}：" + "细节" * 40,
            kind="commitment" if number < 3 else "progress",
            person="小林", project="蓝桥" if number > 0 else None, quote="明确原话",
        ))
    runtime = MemoryRuntimeContext(store, claims)
    assert runtime.build("你好") is None
    assert runtime.build("蓝桥 私人事项") is None
    block = runtime.build("小林 私人事项")
    assert block is not None
    person_lines = [line for line in block.content.splitlines() if line.startswith("- ")]
    assert person_lines and sum(len(line) + 1 for line in person_lines) <= 600
    claims.update_project("蓝桥", "completed")
    block = runtime.build("小林")
    assert block is not None and "私人事项0" in block.content and "私人事项1" not in block.content


def test_person_page_legacy_commitment_respects_mention_gate(builder):
    claims = builder.memory.claims
    saved = claims.remember(ClaimDraft(
        text="小林将于周五就诊", kind="commitment", person="小林", quote="周五去就诊",
    )).claim
    path = builder.workspace / saved.source_file
    path.write_text(path.read_text().replace(f'"person_id":"{saved.person_id}"', '"person_id":null'))
    runtime = MemoryRuntimeContext(builder.memory, claims)
    assert runtime.build("你好") is None
    assert saved.text in runtime.build("小林").content
    path.write_text("\n".join(
        line for line in path.read_text().splitlines() if "<!-- entity:" not in line
    ))
    assert runtime.build("小林") is None


@pytest.mark.parametrize("entity", ["person", "project"])
@pytest.mark.parametrize("action", ["expire", "supersede"])
def test_reconcile_finishes_archive_when_source_cleanup_failed(builder, monkeypatch, entity, action):
    from nanobot.agent import memory_documents

    store = builder.memory
    claims = store.claims
    old = claims.remember(ClaimDraft.model_validate({
        "text": "旧安排是周五交付", "kind": "commitment", entity: "蓝桥",
        "quote": "原话支持", "expires_at": (NOW + timedelta(days=1)).isoformat(),
    })).claim
    source = store.workspace / old.source_file
    marker = f"<!-- claim:{old.claim_id} "
    write = memory_documents.atomic_write_text

    def interrupt(path, content):
        if path == source and marker not in content:
            raise OSError("源文件暂时不可写")
        return write(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(memory_documents, "atomic_write_text", interrupt)
        with pytest.raises(OSError, match="源文件"):
            if action == "expire":
                store.clock = lambda: NOW + timedelta(days=2)
                claims.expire_due()
            else:
                claims.remember(ClaimDraft.model_validate({
                    "text": "新安排改为周一交付", "kind": "commitment", entity: "蓝桥",
                    "quote": "新的原话", "supersedes": [old.claim_id],
                }))
    assert marker in source.read_text()
    assert old.claim_id in store.daily.path_for().read_text()
    claims.db_path.unlink()
    assert not claims.search("周五")
    store.reconcile_files()
    assert marker not in source.read_text()
    archived = claims.search("周五", include_history=True)
    assert len(archived) == 1 and archived[0].status == (
        "expired" if action == "expire" else "superseded"
    )


def test_prompt_build_syncs_once_and_reuses_read_core(builder, monkeypatch):
    claims = builder.memory.claims
    claims.remember(ClaimDraft(text="喜欢画画", kind="preference", core=True))
    claims.remember(ClaimDraft(text="完成报告", kind="commitment"))
    builder.build_messages([], "报告")  # Warm parser and generation caches.
    sync = MagicMock(wraps=claims.index.sync)
    reads = []
    read = Path.read_text

    def counted(path, *args, **kwargs):
        if path.is_relative_to(builder.workspace):
            reads.append(path.relative_to(builder.workspace).as_posix())
        return read(path, *args, **kwargs)

    monkeypatch.setattr(claims.index, "sync", sync)
    monkeypatch.setattr(Path, "read_text", counted)
    messages = builder.build_messages([], "报告")
    assert "喜欢画画" in messages[0]["content"] and "完成报告" in messages[-1]["content"]
    sync.assert_called_once()
    assert reads.count("USER.md") == 1
    assert reads.count("memory/bank/facts.md") == 1
    assert reads.count("memory/ledger.md") == 1


def test_lexical_search_keeps_bounded_candidates_before_dedup_and_limit(builder):
    claims = builder.memory.claims
    claims.documents.apply({"memory/bank/facts.md": "\n".join(
        f"- 检索共同词 条目{number:05d} " + "正文" * 140 for number in range(5000)
    )})
    claims.sync_index()
    tracemalloc.start()
    hits = claims.index.search(
        "检索共同词", now="2030-05-03T12:00:00Z", limit=5, min_score=0,
        exclude=(), include_history=False, deduplicate=True,
    )
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(hits) == 5 and [hit.line for hit in hits] == [1, 2, 3, 4, 5]
    assert peak < 512 * 1024


async def test_restore_rejects_active_dream_without_changing_cursor(builder):
    store = builder.memory
    store.append_history("新摘要")
    entered, resume = asyncio.Event(), asyncio.Event()

    async def process(*_args, **_kwargs):
        entered.set()
        await resume.wait()
        return OutboundMessage(channel="cli", chat_id="test", content="完成",
                               metadata={"_stop_reason": "completed"})

    agent = SimpleNamespace(
        context=SimpleNamespace(memory=store), process_direct=process, dream_runtime=lambda: None,
        sessions=SimpleNamespace(state=SimpleNamespace(prune_dream_sessions=AsyncMock())),
    )
    task = asyncio.create_task(run_dream(agent, trigger="manual"))
    await entered.wait()
    with pytest.raises(ValueError, match="正在整理或恢复"):
        store.restore_memory("selected")
    assert store.get_last_dream_cursor() == 0
    resume.set()
    assert (await task).status == "completed"
    assert store.get_last_dream_cursor() == 1


@pytest.mark.parametrize("structured", [True, False])
def test_dream_maintenance_instructions_match_available_tool_permissions(tmp_path, structured):
    store = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=structured))
    store.append_history("用户纠正了一个事实")
    batch = store.build_dream_prompt()
    tools = store.build_dream_tools(batch)
    if structured:
        assert "Routine maintenance must never call `memory_forget`" in batch.prompt
        assert "runtime archives expired and superseded claims" in batch.prompt
        assert "memory_remember" in tools.tool_names
    else:
        assert "replace the older conflicting fact in place" in batch.prompt
        assert "make focused edits, and verify the results" in batch.prompt
        assert "write_file" in tools.tool_names
