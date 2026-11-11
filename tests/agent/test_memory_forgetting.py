import asyncio
import json
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.memory import MemoryArchiver, MemoryStore
from nanobot.agent.memory_documents import ClaimDraft
from nanobot.agent.tools.memory_claims import MemoryForgetTool
from nanobot.config.schema import MemoryConfig
from nanobot.providers.base import GenerationSettings, LLMProvider, LLMResponse
from nanobot.session.manager import SessionManager
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.rotating_output import RotatingTextOutput


@pytest.mark.parametrize("response", ["summary", "error", "empty", "session-only"])
async def test_archive_discards_results_after_forget_including_raw_fallback(tmp_path, response):
    store = MemoryStore(tmp_path, config=MemoryConfig(daily_notes=True))
    saved = (
        store.claims.remember(ClaimDraft(text="secret archive phrase", kind="fact"))
        if response != "session-only" else None
    )
    started, release = asyncio.Event(), asyncio.Event()
    provider = MagicMock(spec=LLMProvider)
    provider.generation = GenerationSettings(max_tokens=100)

    async def chat(**kwargs):
        started.set()
        await release.wait()
        if response == "error":
            raise RuntimeError("provider unavailable")
        return LLMResponse(content="secret archive phrase" if response in {"summary", "session-only"} else "")

    provider.chat_stream_with_retry = AsyncMock(side_effect=chat)
    archiver = MemoryArchiver(store, MagicMock(), MagicMock())
    messages = [{"role": "user", "content": "secret archive phrase"}]
    task = asyncio.create_task(archiver.archive(
        messages, runtime=LLMRuntime.capture(provider, "test", context_window_tokens=1000),
        session_key="cli:chat", history=messages, request_tools=[],
    ))
    await started.wait()
    if saved is None:
        store.forget_memory(query="secret archive phrase")
    else:
        store.forget_memory(claim_ids=[saved.claim.claim_id])
    release.set()
    assert await task is None
    assert store.read_unprocessed_history(0) == []
    assert list(store.daily.dir.glob("*.md")) == []


@pytest.mark.parametrize("fts", [True, False])
def test_forget_removes_claim_and_fts_bytes_after_vacuum(tmp_path, fts):
    store = MemoryStore(tmp_path)
    store.claims.index.want_fts = fts
    secret = "silverpinepassword"
    saved = store.claims.remember(ClaimDraft(text=secret, kind="fact", quote=secret))
    assert secret.encode() in store.claims.db_path.read_bytes()
    store.forget_memory(claim_ids=[saved.claim.claim_id])
    assert secret.encode() not in store.claims.db_path.read_bytes()
    with store.claims.index.connect() as conn:
        assert conn.execute("PRAGMA secure_delete").fetchone()[0] == 1
        assert not conn.execute("PRAGMA freelist_count").fetchone()[0]
        if fts:
            assert not conn.execute("SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?", (secret,)).fetchall()


def test_history_source_closure_supplies_session_redaction_terms(tmp_path):
    store = MemoryStore(tmp_path)
    store.append_history("原始秘密是银杉海盐")
    batch = store.build_dream_prompt()
    saved = store.remember_from_history(
        ClaimDraft(text="用户有一个保密口令", kind="fact"), batch.sources["h:1:1"], run_id=batch.run_id,
    )
    result, _ = store.forget_memory(claim_ids=[saved.claim.claim_id])
    assert "原始秘密是银杉海盐" in result.redact_terms


async def test_forget_cleans_all_supported_local_files_and_reopens_active_log(tmp_path, monkeypatch):
    workspace, data, home = (tmp_path / name for name in ("workspace", "data", "home"))
    sessions = SessionManager(workspace, sessions_root=data / "sessions")
    monkeypatch.setattr("nanobot.agent.memory_forgetting.get_data_dir", lambda: data)
    monkeypatch.setattr("nanobot.agent.memory_forgetting.get_cli_history_path", lambda: home / "history/cli_history")
    store = MemoryStore(workspace)
    secret = "银杉海盐"
    text_paths = [
        workspace / "memory/HISTORY.md", workspace / "memory/HISTORY.md.bak",
        workspace / "memory/HISTORY.md.bak.2", home / "history/cli_history",
        data / "logs/gateway.log", data / "logs/gateway.log.1", data / "logs/gateway.log.5",
        data / "logs/api.instance.log", data / "logs/api.instance.log.1", data / "logs/api.instance.log.5",
    ]
    for path in text_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"删除 {secret}\n保留行\n", encoding="utf-8")
    jsonl_paths = [
        sessions.sessions_dir / "legacy.jsonl", workspace / "sessions/legacy.jsonl",
        sessions.sessions_dir / "jsonl-backup/legacy.jsonl",
        data / "webui/websocket_chat.jsonl", data / "webui/websocket_chat.segments/000001.jsonl",
    ]
    kept = {"role": "user", "content": "保留\u2028记录"}
    for path in jsonl_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"role": "user", "content": secret}) + "\n" +
                        json.dumps({"tool_calls": [{"function": {"arguments": json.dumps({"query": secret})}}]}) + "\n" +
                        json.dumps(kept, ensure_ascii=False) + "\n", encoding="utf-8")
    checkpoints = [path.with_suffix(".checkpoint.json") for path in jsonl_paths[:3]]
    for path in checkpoints:
        path.write_text(json.dumps({"checkpoint": {"content": secret}, "version": 1}, indent=2), encoding="utf-8")
    output = RotatingTextOutput(data / "logs/gateway.log", max_bytes=10000, backup_count=1)
    output.write(f"正在写入 {secret}\n")
    try:
        result = json.loads(await MemoryForgetTool(store, sessions=sessions).execute(query=secret))
        assert result["status"] == "forgotten"
        assert "gateway/api 日志及轮转副本" in result["completed"]
        output.write("后续日志\n")
        for path in text_paths:
            assert secret not in path.read_text(encoding="utf-8")
            assert "保留行" in path.read_text(encoding="utf-8")
        assert (data / "logs/gateway.log").read_text(encoding="utf-8").endswith("后续日志\n")
        for path in jsonl_paths:
            assert [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line] == [kept]
        for path in checkpoints:
            assert json.loads(path.read_text()) == {"checkpoint": {"content": "[已遗忘]"}, "version": 1}
    finally:
        output.close()
        await sessions.state.aclose()


async def test_file_cleanup_error_is_reported_as_partial(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    saved = store.claims.remember(ClaimDraft(text="secret phrase", kind="fact"))
    path = tmp_path / "sessions/broken.jsonl"
    path.parent.mkdir()
    path.write_text('{"content":')
    result = await MemoryForgetTool(store).execute(claim_ids=[saved.claim.claim_id])
    assert result.is_error
    payload = json.loads(result)
    assert payload["status"] == "partial" and payload["completed"] and payload["incomplete"]
    assert path.read_text() == '{"content":'
    assert store.claims.get(saved.claim.claim_id).status == "retracted"



async def test_index_cleanup_failure_is_not_reported_as_success(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    saved = store.claims.remember(ClaimDraft(text="secret phrase", kind="fact"))

    def busy():
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store.claims.index, "purge_deleted", busy)
    result = await MemoryForgetTool(store).execute(claim_ids=[saved.claim.claim_id])
    assert result.is_error and "部分完成" in result
    assert store.claims.get(saved.claim.claim_id).status == "retracted"
