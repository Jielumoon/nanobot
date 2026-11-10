"""Dream user paths: shared execution, live evidence, private notes, and safe restore."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.dream import dream_running, run_dream
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_claims import ClaimStore
from nanobot.agent.memory_documents import ClaimDraft
from nanobot.agent.memory_dreams import DreamBatch
from nanobot.agent.memory_recall import MemoryRuntimeContext
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.cli.gateway_runtime import _run_scheduled_dream
from nanobot.command.builtin import cmd_dream
from nanobot.command.router import CommandContext
from nanobot.config.schema import MemoryConfig, MemoryPeopleConfig
from nanobot.utils.gitstore import GitStore, GitStoreError

NOW = datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        tmp_path, clock=lambda: NOW, timezone="Asia/Shanghai",
        config=MemoryConfig(
            claims_enabled=True, daily_notes=True, people=MemoryPeopleConfig(enabled=True),
        ),
    )


@pytest.fixture
def claims(store: MemoryStore) -> ClaimStore:
    assert store.claims is not None
    return store.claims


def response(reason: str = "completed") -> OutboundMessage:
    return OutboundMessage(
        channel="cli", chat_id="direct", content="完成", metadata={"_stop_reason": reason},
    )


def agent_for(store: MemoryStore, process=None):
    return SimpleNamespace(
        context=SimpleNamespace(memory=store),
        sessions=SimpleNamespace(state=SimpleNamespace(prune_dream_sessions=AsyncMock())),
        process_direct=process or AsyncMock(return_value=response()),
        dream_runtime=lambda: None,
        bus=SimpleNamespace(publish_outbound=AsyncMock()),
    )


@pytest.mark.parametrize("entrypoint", ["manual", "scheduled"])
async def test_dream_entrypoints_share_evidence_audit_and_cursor(
    store: MemoryStore, claims: ClaimStore, entrypoint: str,
) -> None:
    store.append_history('- [durable] 已完成迁移预演 (user: "迁移预演完成了")',
                         session_key="cli:source")

    async def process(prompt, **kwargs):
        assert "[h:1:1]" in prompt
        assert kwargs["ephemeral"]
        tools = kwargs["tools"]
        await tools.execute("memory_reflect", {"section": "反思", "lines": ["迁移进展可归档"]})
        result = json.loads(await tools.execute("memory_remember", {
            "text": "已完成迁移预演", "kind": "progress", "project": "迁移",
            "source": "h:1:1", "quote": "迁移预演完成了",
        }))
        assert result["status"] == "saved"
        await tools.execute("memory_project", {"project": "迁移", "status": "completed"})
        return response()

    agent = agent_for(store, process)
    if entrypoint == "manual":
        msg = InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/dream")
        ctx = CommandContext(msg=msg, session=None, key="cli:direct", raw="/dream", args="",
                             loop=agent)
        before = asyncio.all_tasks()
        immediate = await cmd_dream(ctx)
        await asyncio.gather(*(asyncio.all_tasks() - before))
        assert immediate.content == "正在整理 Dream 记忆……"
        assert "记忆整理完成" in agent.bus.publish_outbound.call_args.args[0].content
    else:
        mcp = SimpleNamespace(connect=AsyncMock())
        await _run_scheduled_dream(agent, mcp)
        mcp.connect.assert_awaited_once()

    assert store.get_last_dream_cursor() == 1
    agent.sessions.state.prune_dream_sessions.assert_awaited_once()
    saved = claims.search("迁移预演")[0]
    claim = claims.get(saved.claim_id)
    assert claim is not None
    assert claim.evidence.source_ref == "history:1"
    assert claim.evidence.session_key == "cli:source"
    assert claim.evidence.source_timestamp == "2030-05-04 00:30"
    assert claim.evidence.source_kind == "summary"
    events = claims.documents.events()
    assert {event.actor for event in events} == {"dream"}
    run_ids = {event.run_id for event in events}
    assert len(run_ids) == 1 and None not in run_ids
    assert any(event.action == "update" and event.target_type == "entity" for event in events)
    audit = store.dreams.path().read_text()
    assert store.dreams.path().name == "2030-05-04.md"
    assert "## 反思" in audit and "## 变更审计" in audit
    assert claim.claim_id in audit and next(iter(run_ids)) in audit
    content_commit = store.git.log(message_prefix="dream:")[0]
    assert content_commit.sha in audit
    assert not dream_running(store.workspace)


async def test_dream_single_flight_allows_forgetting_during_model_wait(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    saved = claims.remember(ClaimDraft(text="养了一只猫", kind="fact")).claim
    store.append_history("养了一只猫")
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def process(_prompt, **kwargs):
        entered.set()
        await resume.wait()
        result = await kwargs["tools"].execute("memory_remember", {
            "text": "养了一只猫", "kind": "fact", "source": "h:1",
        })
        assert "已被遗忘" in result
        return response()

    first = asyncio.create_task(run_dream(agent_for(store, process), trigger="scheduled"))
    await entered.wait()
    other_store = MemoryStore(store.workspace, config=store.config, clock=lambda: NOW)
    second = agent_for(other_store)
    busy = await run_dream(second, trigger="manual")
    assert busy.status == "busy"
    second.process_direct.assert_not_awaited()
    try:
        forgotten, _ = await asyncio.wait_for(asyncio.to_thread(
            store.forget_memory, claim_ids=[saved.claim_id],
        ), timeout=2)
        assert forgotten.retracted
    finally:
        resume.set()
        await first
    assert not claims.search("养了一只猫")
    assert store.get_last_dream_cursor() == 1


async def test_simultaneous_manual_dreams_share_one_run_and_completion(store: MemoryStore) -> None:
    store.append_history("需要整理的摘要")
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def process(_prompt, **_kwargs):
        entered.set()
        await resume.wait()
        return response()

    agent = agent_for(store, AsyncMock(side_effect=process))
    msg = InboundMessage(channel="cli", sender_id="u", chat_id="direct", content="/dream")
    ctx = CommandContext(msg=msg, session=None, key="cli:direct", raw="/dream", args="", loop=agent)
    before = asyncio.all_tasks()
    first = await cmd_dream(ctx)
    second = await cmd_dream(ctx)
    await entered.wait()
    resume.set()
    await asyncio.gather(*(asyncio.all_tasks() - before))

    assert first.content == second.content == "正在整理 Dream 记忆……"
    agent.process_direct.assert_awaited_once()
    agent.bus.publish_outbound.assert_awaited_once()
    assert "记忆整理完成" in agent.bus.publish_outbound.call_args.args[0].content


@pytest.mark.parametrize("failure", ["exception", "max_iterations"])
async def test_dream_failure_audits_but_does_not_advance(
    store: MemoryStore, failure: str,
) -> None:
    store.append_history("有待处理的摘要")

    async def process(_prompt, **_kwargs):
        if failure == "exception":
            raise RuntimeError("模型调用失败")
        return response(failure)

    agent = agent_for(store, process)
    if failure == "exception":
        with pytest.raises(RuntimeError, match="模型调用失败"):
            await run_dream(agent, trigger="manual")
        assert "状态=失败" in store.dreams.path().read_text()
    else:
        result = await run_dream(agent, trigger="manual")
        assert result.status == "incomplete"
        assert "状态=未完成" in store.dreams.path().read_text()
    assert store.get_last_dream_cursor() == 0
    assert not dream_running(store.workspace)
    agent.sessions.state.prune_dream_sessions.assert_awaited_once()


async def test_dream_commit_failure_keeps_batch_pending(store: MemoryStore, monkeypatch) -> None:
    store.append_history("有待整理的摘要")
    original = store.git.auto_commit

    def fail_audit(message: str):
        if message.startswith("dream-audit:"):
            raise GitStoreError("磁盘写入失败")
        return original(message)

    monkeypatch.setattr(store.git, "auto_commit", fail_audit)
    with pytest.raises(GitStoreError, match="磁盘写入失败"):
        await run_dream(agent_for(store), trigger="manual")
    assert store.get_last_dream_cursor() == 0
    assert not dream_running(store.workspace)
    monkeypatch.setattr(store.git, "auto_commit", original)
    assert (await run_dream(agent_for(store), trigger="manual")).status == "completed"
    assert store.get_last_dream_cursor() == 1


async def test_dream_without_history_expires_and_reconciles(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    expired = claims.remember(ClaimDraft(
        text="临时出差地点", kind="circumstance", expires_at="2030-05-01",
    )).claim
    forgotten = claims.remember(ClaimDraft(text="秘密地点", kind="fact")).claim
    path = store.workspace / str(forgotten.source_file)
    raw = path.read_text()
    store.forget_memory(claim_ids=[forgotten.claim_id])
    path.write_text(raw)
    agent = agent_for(store)

    result = await run_dream(agent, trigger="scheduled")

    assert result.status == "empty"
    agent.process_direct.assert_not_awaited()
    assert "秘密地点" not in path.read_text()
    archived = claims.get(expired.claim_id)
    assert archived is not None and archived.status == "expired"
    assert archived.source_file == "memory/daily/2030-05-04.md"
    assert not claims.search("临时出差地点")
    audit = store.dreams.path().read_text()
    assert expired.claim_id in audit and "过期" in audit
    assert forgotten.claim_id not in audit  # User's prior mutation is outside this run.
    assert store.get_last_dream_cursor() == 0


async def test_dream_handles_validate_batch_quote_and_live_history(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    store.append_history('- 养了一只猫\n- 小林负责设计 (user: "小林负责设计")',
                         session_key="cli:source")
    store.append_history("下一批的事实")
    batch = store.build_dream_prompt(max_entries=1)
    assert batch is not None
    tools = store.build_dream_tools(batch)
    for source in (None, "h:2", "h:1:3"):
        result = await tools.execute("memory_remember", {
            "text": "事实", "kind": "fact", **({"source": source} if source else {}),
        })
        assert "来源不在本批次" in result
    for quote in (None, "小林喜欢旅行"):
        result = await tools.execute("memory_remember", {
            "text": "小林负责设计", "kind": "fact", "person": "小林", "source": "h:1:2",
            **({"quote": quote} if quote else {}),
        })
        assert result.is_error

    raw_history = store.history_file.read_text()
    store.forget_memory(query="养了一只猫", all_matches=True)
    store.history_file.write_text(raw_history)  # Even a manually restored source stays invalid.
    for source in ("h:1", "h:1:1"):
        result = await tools.execute("memory_remember", {
            "text": "养了一只猫", "kind": "fact", "source": source,
        })
        assert "已被遗忘" in result
    stale_reflection = await tools.execute("memory_reflect", {
        "section": "待确认", "lines": ["还记得养猫的事情"],
    })
    assert "已被遗忘" in stale_reflection
    saved = json.loads(await tools.execute("memory_remember", {
        "text": "小林负责设计", "kind": "fact", "person": "小林", "source": "[h:1:2]",
        "quote": "小林负责设计",
    }))["claim"]
    assert claims.get(saved["claim_id"]).evidence.source_kind == "summary"
    store.history_file.write_text(raw_history.replace("小林负责设计", "小林负责开发"))
    rejected = await tools.execute("memory_remember", {
        "text": "小林负责设计", "kind": "fact", "source": "h:1:2",
    })
    assert "已被遗忘或修改" in rejected


async def test_dream_notes_are_private_bounded_and_cannot_edit_audit(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    store.append_history("近期观察")
    batch = store.build_dream_prompt()
    assert batch is not None
    tools = store.build_dream_tools(batch)
    result = await tools.execute("memory_reflect", {
        "section": "待确认", "lines": [f"秘密观察{index}" + "内容" * 200 for index in range(8)],
    })
    assert "已写入" in result
    assert 1000 < len(store.dreams.pending()) <= 1800
    assert "秘密观察" in store.build_dream_prompt().prompt
    builder = ContextBuilder(store.workspace, memory_config=store.config)
    assert "秘密观察" not in builder.build_system_prompt()
    assert not claims.search("秘密观察")
    assert MemoryRuntimeContext(store, claims).build("秘密观察") is None
    for path in ("dreams/2030-05-04.md", "memory/MEMORY.md", "memory/daily/2030-05-04.md"):
        rejected = await tools.execute("write_file", {"path": path, "content": "伪造审计与记忆"})
        assert rejected.is_error
    rejected = await tools.execute("memory_reflect", {"section": "变更审计", "lines": ["伪造"]})
    assert rejected.is_error
    assert "伪造" not in store.dreams.path().read_text()


def test_dream_audit_owns_only_its_run_and_forgotten_bodies_stay_hidden(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    store.git.init()
    unrelated = claims.remember(ClaimDraft(text="本轮用户记忆", kind="fact")).claim
    kept = claims.remember(ClaimDraft(text="需要保留的进展", kind="progress"),
                           actor="dream", run_id="run-a").claim
    forgotten = claims.remember(ClaimDraft(text="已经遗忘的地点", kind="fact"),
                                actor="dream", run_id="run-a").claim
    store.forget_memory(claim_ids=[forgotten.claim_id], actor="dream", run_id="run-a")
    store.finish_dream(run_id="run-a", status="completed", cursor=1, prefix="dream: test")
    audit = store.dreams.path().read_text()
    assert kept.text in audit and kept.claim_id in audit
    assert forgotten.claim_id in audit and "已遗忘" in audit
    assert forgotten.text not in audit
    assert unrelated.claim_id not in audit and unrelated.text not in audit

    store.forget_memory(claim_ids=[kept.claim_id])
    assert kept.text not in store.dreams.path().read_text()
    store.dreams.path().write_text(audit)
    store.reconcile_files()
    assert kept.claim_id not in store.dreams.path().read_text()


def test_dream_only_forgetting_survives_restored_notes(store: MemoryStore) -> None:
    store.dreams.append("待确认", ["- 尚未证实的旅行计划", "- 保留其他问题"])
    raw = store.dreams.path().read_text()
    result, _ = store.forget_memory(query="旅行计划", all_matches=True)
    assert result.removed_lines
    assert "旅行计划" not in store.dreams.path().read_text()
    store.dreams.path().write_text(raw)
    assert "旅行计划" not in store.dreams.pending()
    store.reconcile_files()
    assert "旅行计划" not in store.dreams.path().read_text()
    assert "保留其他问题" in store.dreams.path().read_text()


def test_memory_restore_reconciles_full_snapshot_before_commit(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    store.git.init()
    claim = claims.remember(ClaimDraft(
        text="小林负责秘密项目", kind="decision", person="小林", project="秘密项目",
        quote="小林负责秘密项目",
    )).claim
    store.append_history("- 小林负责秘密项目\n- 保留的事实")
    raw_history = store.history_file.read_text()
    store.dreams.append("反思", ["- 小林负责秘密项目"])
    store.write_soul("# 原来的行为规则")
    store.git.auto_commit("dream: first")
    store.write_soul("# 新的行为规则")
    selected = store.git.auto_commit("dream: selected")
    assert selected is not None
    later = claims.remember(ClaimDraft(text="之后才创建的页面", kind="experience")).claim
    store.git.auto_commit("dream: later")
    store.forget_memory(person="小林", all_matches=True)
    ledger = claims.documents.ledger.read_text()
    store.history_file.write_text(raw_history)

    restored = store.restore_memory(selected)

    assert restored is not None
    assert store.read_soul() == "# 原来的行为规则"
    assert not (store.memory_dir / "people/小林.md").exists()
    assert not (store.workspace / str(later.source_file)).exists()
    assert "小林负责秘密项目" not in store.dreams.path().read_text()
    assert "小林负责秘密项目" not in store.history_file.read_text()
    assert "保留的事实" in store.read_unprocessed_history(0)[0]["content"]
    assert claims.documents.ledger.read_text() == ledger
    claims.db_path.unlink()
    assert not claims.search("秘密项目")
    from dulwich.repo import Repo

    with Repo(str(store.workspace)) as repo:
        blob = repo[repo[repo.head()].tree][b"dreams"]
        tree = repo[blob[1]]
        note = repo[tree[b"2030-05-04.md"][1]].data.decode()
        assert "小林负责秘密项目" not in note
    assert claims.get(claim.claim_id).status == "retracted"


def test_git_store_upgrades_existing_ignore_and_never_tracks_databases(tmp_path: Path) -> None:
    old = GitStore(tmp_path, tracked_files=["memory/MEMORY.md"])
    old.init()
    upgraded = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=True))
    upgraded.dreams.append("反思", ["- 今天的观察"])
    assert not upgraded.git.init()
    for name in ("claims.sqlite3", "claims.sqlite3-wal", "claims.sqlite3-shm"):
        (upgraded.memory_dir / name).write_bytes(b"cache")
    sha = upgraded.git.auto_commit("dream: upgraded")
    assert sha is not None
    from dulwich.ignore import IgnoreFilterManager
    from dulwich.repo import Repo

    with Repo(str(tmp_path)) as repo:
        ignored = IgnoreFilterManager.from_repo(repo)
        for name in ("claims.sqlite3", "claims.sqlite3-wal", "claims.sqlite3-shm"):
            assert ignored.is_ignored(f"memory/{name}")
        paths = [path.decode() for path in repo.open_index().paths()]
    assert any(path.startswith("dreams/") for path in paths)
    assert not any("sqlite" in path for path in paths)


def test_dream_entry_budget_is_shared_across_lines(store: MemoryStore) -> None:
    store.append_history("\n".join(str(index) * 200 for index in range(10)))
    batch = store.build_dream_prompt()
    assert isinstance(batch, DreamBatch)
    history = batch.prompt.split("## Conversation History\n", 1)[1]
    assert len(history) < 1200
    assert "[h:1:5]" in history and "[h:1:6]" not in history
