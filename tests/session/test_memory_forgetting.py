import asyncio
import json
import sqlite3
import threading

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.agent.tools.context import RequestContext, ToolContext, request_context
from nanobot.agent.tools.memory_claims import MemoryForgetTool
from nanobot.bus.events import InboundMessage
from nanobot.config.schema import Config
from nanobot.providers.base import ProviderConversationState
from nanobot.session.display_history import HistoryFact, HistoryKind
from nanobot.session.history_graph import expand_graph_fields, history_record_size
from nanobot.session.manager import SessionManager
from nanobot.session.recovery import restore_runtime_checkpoint
from nanobot.session.sqlite_store import SessionConflictError
from nanobot.webui.transcript import build_webui_thread_response


def _provider():
    return ProviderConversationState(
        kind="test", provider="test", model="test", version=1, payload={"opaque": "server-context"},
    )


@pytest.mark.parametrize("transient", [False, True])
async def test_forget_changes_only_matching_session_generations(tmp_path, transient):
    sessions = SessionManager(tmp_path)
    state = sessions.state
    try:
        create = state.register_transient if transient else state.get
        matched = await create("websocket:matched")
        clean = await create("websocket:clean")
        for draft, content in ((matched, "银杉海盐密语"), (clean, "保留的会话")):
            draft.add_message("user", content)
            draft.provider_state = _provider()
            await state.prepare_input(draft)
            if not transient:
                await state.append_history(draft.key, [HistoryFact(HistoryKind.INPUT, {"text": content})],
                                           expected_generation=draft.generation)
        if not transient:
            history_before = {
                draft.key: await state.project_history(draft.key, lambda history: history.revision)
                for draft in (matched, clean)
            }

        await state.redact_memory(["银杉海盐密语"])
        rewritten = await state.get(matched.key)
        untouched = await state.get(clean.key)
        assert rewritten.generation != matched.generation
        assert rewritten.revision == matched.revision + 1
        assert rewritten.messages[0]["content"] == "[已遗忘]"
        assert rewritten.provider_state is None
        assert untouched.generation == clean.generation
        assert untouched.revision == clean.revision
        assert untouched.provider_state is None

        # An unrelated in-flight turn must still be able to commit its answer.
        clean.add_message("assistant", "继续完成这一轮")
        await state.finish_turn(clean)
        assert clean.messages[-1]["content"] == "继续完成这一轮"
        assert clean.provider_state is None
        if not transient:
            assert await state.project_history(clean.key, lambda history: history.revision) == history_before[clean.key]
            assert await state.project_history(matched.key, lambda history: history.revision) == (
                rewritten.generation, history_before[matched.key][1] + 1,
            )
        with pytest.raises(SessionConflictError):
            await state.finish_turn(matched)
    finally:
        await state.aclose()


async def test_forget_scrubs_current_turn_replay_restart_and_database_bytes(tmp_path):
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    state = sessions.state
    secret = "银杉海盐密语"
    try:
        current = await state.get("websocket:chat")
        current.add_message("user", secret)
        current.last_archived = 1
        current.metadata.update(title=secret, _last_summary={"text": secret})
        current.provider_state = _provider()
        await state.prepare_input(current)
        child = await state.create_child(
            current.key, "subagent:child", metadata={"label": secret},
            content=secret, policy=current.policy,
        )
        child.metadata.update({
            "_last_summary": {"text": "保留的摘要"},
            "runtime_checkpoint": {"assistant_message": {"role": "assistant", "content": secret}},
        })
        child.provider_state = _provider()
        await state.prepare_input(child)
        await state.queue_followup(child.key, InboundMessage(
            channel="websocket", sender_id="user", chat_id="chat", content=secret,
        ))
        stale_child = await state.get(child.key)
        clean = await state.get("cli:clean")
        clean.add_message("user", "保留的会话")
        clean.provider_state = _provider()
        await state.finish_turn(clean)

        graph = {"nodes": [{"path": f"/{secret}.md", "name": secret, "tokens": 7, "state": "loaded"}], "edges": []}
        await state.append_history(current.key, [
            HistoryFact(HistoryKind.INPUT, {"text": secret}),
            HistoryFact(HistoryKind.MODEL_REQUEST, {
                "request_id": "request", "context_graph": {
                    "kind": "context_graph_parts", "stage": graph,
                    "system": {"nodes": [], "edges": []}, "reads": {"nodes": [], "edges": []},
                },
                "context_snapshot": {"graph": graph, "estimated_tokens": 7, "context_window_tokens": 1000},
            }),
            HistoryFact(HistoryKind.ANSWER, {"text": secret}),
            HistoryFact(HistoryKind.COMPLETION),
        ], expected_generation=current.generation)
        with sqlite3.connect(sessions._store.path) as db:
            assert "graph-columns-v1:" in db.execute(
                "SELECT record FROM history_entries WHERE kind='model_request'",
            ).fetchone()[0]

        memory = MemoryStore(workspace)
        tool = MemoryForgetTool.create(ToolContext(
            config=Config().tools, workspace=str(workspace), memory=memory, sessions=sessions,
        ))
        with request_context(RequestContext(channel="websocket", chat_id="chat", session_key=current.key)):
            result = json.loads(await tool.execute(query=secret))
        # The only copies were in sessions; the tool must still schedule cleanup.
        assert result["status"] == "forgotten" and result["pending"]
        assert result["retracted"] == [] and result["history_entries_redacted"] == 0
        assert secret not in json.dumps(result, ensure_ascii=False)
        current.add_message("assistant", secret)
        await state.finish_turn(current)
        assert current.messages[-1]["content"] == "[已遗忘]"
        assert current.metadata["_last_summary"]["text"] == "[已遗忘]"
        assert current.last_archived == 1
        assert secret not in json.dumps(current.get_history(), ensure_ascii=False)
        assert state.peek(child.key) is None
        restored_child = await state.get(child.key)
        assert restored_child.metadata["_last_summary"]["text"] == "保留的摘要"
        assert restored_child.metadata["runtime_checkpoint"]["assistant_message"]["content"] == "[已遗忘]"
        assert restored_child.metadata["pending_user_followups"][0]["content"] == "[已遗忘]"
        assert restored_child.provider_state is None
        assert (await state.get(clean.key)).provider_state is None
        with pytest.raises(SessionConflictError):
            await state.finish_turn(stale_child)
        with pytest.raises(SessionConflictError):
            await state.checkpoint_view(stale_child, {"assistant_message": {"content": secret}})
        with pytest.raises(SessionConflictError):
            await state.append_history(child.key, [HistoryFact(HistoryKind.ANSWER, {"text": secret})],
                                       expected_generation=stale_child.generation)

        payload = await state.project_history(current.key, lambda history: build_webui_thread_response(
            current.key, history=history, include_model_requests=True,
        ))
        assert secret not in json.dumps(payload, ensure_ascii=False)
        with sqlite3.connect(sessions._store.path) as db:
            records = [row[0] for row in db.execute("SELECT record FROM history_entries ORDER BY position")]
            decoded = expand_graph_fields(HistoryFact.decode(records[1]).content)
            assert decoded["context_graph"]["nodes"][0]["name"] == "[已遗忘]"
            assert decoded["context_snapshot"]["graph"]["nodes"][0]["path"] == "[已遗忘]"
            assert db.execute("SELECT byte_count FROM history_turns").fetchone()[0] == sum(map(history_record_size, records))
            assert db.execute("SELECT record_count, user_count FROM history_turns").fetchone() == (4, 1)
            assert secret not in "\n".join(db.iterdump())
        for path in sessions.sessions_dir.glob("sessions.sqlite3*"):
            assert secret.encode() not in path.read_bytes()
    finally:
        await state.aclose()
    reopened = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    try:
        for key in ("websocket:chat", "subagent:child", "cli:clean"):
            session = await reopened.state.get(key)
            assert secret not in json.dumps([session.messages, session.metadata], ensure_ascii=False)
            assert session.provider_state is None
    finally:
        await reopened.state.aclose()


async def test_forget_preserves_call_ids_and_decodes_tool_arguments(tmp_path):
    sessions = SessionManager(tmp_path)
    state = sessions.state
    try:
        draft = await state.get("cli:chat")
        draft.messages = [
            {"role": "user", "content": "secret"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "secret-call", "type": "function", "function": {
                    "name": "secret_tool", "arguments": json.dumps({"name": "secret", "keep": 3}),
                },
            }]},
            {"role": "tool", "tool_call_id": "secret-call", "name": "secret_tool", "content": "secret result"},
        ]
        state.request_memory_forget(draft.key, ["secret"])
        await state.finish_turn(draft)
        call = draft.messages[1]["tool_calls"][0]
        assert call["id"] == draft.messages[2]["tool_call_id"] == "secret-call"
        assert call["function"]["name"] == draft.messages[2]["name"] == "secret_tool"
        assert call["type"] == "function" and draft.messages[2]["role"] == "tool"
        assert json.loads(call["function"]["arguments"]) == {"name": "[已遗忘]", "keep": 3}
        assert draft.messages[2]["content"] == "[已遗忘]"
        assert draft.get_history()[-1]["tool_call_id"] == "secret-call"
    finally:
        await state.aclose()


async def test_forget_runs_after_cancelled_checkpoint_is_restored(tmp_path):
    sessions = SessionManager(tmp_path)
    state = sessions.state
    try:
        draft = await state.get("cli:chat")
        draft.add_message("user", "secret")
        await state.prepare_input(draft)
        await state.checkpoint_view(draft, {
            "phase": "final_response", "assistant_message": {"role": "assistant", "content": "secret"},
            "completed_tool_results": [], "pending_tool_calls": [],
        })
        state.request_memory_forget(draft.key, ["secret"])
        recovered = await state.get(draft.key)
        assert restore_runtime_checkpoint(recovered)
        await state.finish_turn(recovered)
        assert [m["content"] for m in recovered.messages] == ["[已遗忘]", "[已遗忘]"]
    finally:
        await state.aclose()


async def test_cancellation_during_finish_still_drains_forget(tmp_path, monkeypatch):
    sessions = SessionManager(tmp_path)
    state = sessions.state
    started, release = threading.Event(), threading.Event()
    original = sessions._store.save

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    try:
        draft = await state.get("cli:chat")
        draft.add_message("user", "secret")
        state.request_memory_forget(draft.key, ["secret"])
        monkeypatch.setattr(sessions._store, "save", blocked)
        finishing = asyncio.create_task(state.finish_turn(draft))
        assert await asyncio.to_thread(started.wait, 2)
        finishing.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await finishing
        assert (await state.get(draft.key)).messages[0]["content"] == "[已遗忘]"
    finally:
        release.set()
        await state.aclose()


async def test_cancellation_during_tool_still_registers_session_cleanup(tmp_path, monkeypatch):
    sessions = SessionManager(tmp_path)
    state = sessions.state
    store = MemoryStore(tmp_path)
    started, release = threading.Event(), threading.Event()
    original = store.forget_memory

    def blocked(**kwargs):
        started.set()
        assert release.wait(3)
        return original(**kwargs)

    try:
        draft = await state.get("cli:chat")
        draft.add_message("user", "secret")
        await state.prepare_input(draft)
        monkeypatch.setattr(store, "forget_memory", blocked)
        with request_context(RequestContext(channel="cli", chat_id="chat", session_key=draft.key)):
            task = asyncio.create_task(MemoryForgetTool(store, sessions=sessions).execute(query="secret"))
            assert await asyncio.to_thread(started.wait, 2)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        recovered = await state.get(draft.key)
        await state.finish_turn(recovered)
        assert recovered.messages[0]["content"] == "[已遗忘]"
    finally:
        release.set()
        await state.aclose()


async def test_busy_wal_is_reported_as_incomplete(tmp_path, monkeypatch):
    sessions = SessionManager(tmp_path)
    state = sessions.state
    try:
        draft = await state.get("cli:chat")
        draft.add_message("user", "secret")
        await state.finish_turn(draft)
        original = sessions._store._open_connection

        def quick_connection(**kwargs):
            connection = original(**kwargs)
            connection.execute("PRAGMA busy_timeout=1")
            return connection

        monkeypatch.setattr(sessions._store, "_open_connection", quick_connection)
        with sqlite3.connect(sessions._store.path) as reader:
            reader.execute("BEGIN")
            reader.execute("SELECT * FROM messages").fetchall()
            with pytest.raises(sqlite3.OperationalError, match="WAL 清理未完成"):
                await state.redact_memory(["secret"])
        # The cleanup transaction committed, but physical cleanup failed visibly.
        assert (await state.get(draft.key)).messages[0]["content"] == "[已遗忘]"
        await state.redact_memory(["secret"])
        assert b"secret" not in sessions._store.path.read_bytes()
    finally:
        await state.aclose()


async def test_delete_during_forget_does_not_recreate_session(tmp_path, monkeypatch):
    sessions = SessionManager(tmp_path)
    state = sessions.state
    started, release = threading.Event(), threading.Event()
    original = sessions._store.vacuum

    def blocked():
        started.set()
        assert release.wait(3)
        original()

    try:
        draft = await state.get("cli:chat")
        draft.add_message("user", "secret")
        state.request_memory_forget(draft.key, ["secret"])
        monkeypatch.setattr(sessions._store, "vacuum", blocked)
        finishing = asyncio.create_task(state.finish_turn(draft))
        assert await asyncio.to_thread(started.wait, 2)
        deleting = asyncio.create_task(state.delete(draft.key))
        await asyncio.sleep(0)
        release.set()
        await deleting
        with pytest.raises(SessionConflictError, match="已被删除"):
            await finishing
        assert await state.read(draft.key) is None
    finally:
        release.set()
        await state.aclose()
