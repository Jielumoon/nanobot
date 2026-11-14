from __future__ import annotations

import asyncio
import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nanobot.agent.context import TranscriptInput
from nanobot.bus.events import InboundMessage
from nanobot.providers.base import LLMResponse, LLMUsage
from nanobot.session.manager import Session
from nanobot.session.state import SessionState


def _make_loop(tmp_path):
    """Create a minimal AgentLoop with mocked dependencies."""
    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus

    bus = MessageBus()
    provider = MagicMock(aclose=AsyncMock())
    provider.get_default_model.return_value = "test-model"
    workspace = tmp_path


    with patch("nanobot.agent.loop.ContextBuilder"), \
         patch("nanobot.agent.loop.SessionManager") as mock_sessions, \
         patch("nanobot.agent.loop.SubagentManager") as mock_sub_mgr:
        mock_sessions.return_value.sessions_dir = tmp_path / "sessions"
        mock_sessions.return_value.state = MagicMock(spec=SessionState)
        mock_sessions.return_value.state.get = AsyncMock(side_effect=lambda key: Session(key))
        mock_sessions.return_value.state.peek.return_value = None
        mock_sub_mgr.return_value.close = AsyncMock()
        loop = AgentLoop(bus=bus, provider=provider, workspace=workspace)
    return loop, bus


class TestRestartCommand:

    @pytest.mark.asyncio
    async def test_restart_sends_message_and_calls_execv(self, tmp_path):
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext
        from nanobot.utils.restart import (
            RESTART_NOTIFY_CHANNEL_ENV,
            RESTART_NOTIFY_CHAT_ID_ENV,
            RESTART_STARTED_AT_ENV,
        )

        loop, _bus = _make_loop(tmp_path)
        loop.restart_mode = "exec"
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        async def _fast_sleep(_delay: float) -> None:
            return None

        scheduled: list[asyncio.Task] = []

        def _capture_task(coro):
            task = asyncio.create_task(coro)
            scheduled.append(task)
            return task

        fake_asyncio = SimpleNamespace(
            sleep=_fast_sleep,
            create_task=_capture_task,
        )

        with patch.dict(os.environ, {}, clear=False), \
             patch("nanobot.command.builtin.asyncio", new=fake_asyncio), \
             patch("nanobot.command.builtin.os.execv") as mock_execv:
            out = await cmd_restart(ctx)
            assert "正在重启" in out.content
            assert os.environ.get(RESTART_NOTIFY_CHANNEL_ENV) == "cli"
            assert os.environ.get(RESTART_NOTIFY_CHAT_ID_ENV) == "direct"
            assert os.environ.get(RESTART_STARTED_AT_ENV)

            assert scheduled
            await scheduled[0]
            mock_execv.assert_called_once()

    @pytest.mark.asyncio
    async def test_restart_windows_auto_spawns_and_exits(self, tmp_path):
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        async def _fast_sleep(_delay: float) -> None:
            return None

        scheduled: list[asyncio.Task] = []
        fake_asyncio = SimpleNamespace(
            sleep=_fast_sleep,
            create_task=lambda coro: scheduled.append(asyncio.create_task(coro)) or scheduled[-1],
        )

        with patch("nanobot.command.builtin.asyncio", new=fake_asyncio), \
             patch("nanobot.command.builtin.sys.platform", "win32"), \
             patch("nanobot.command.builtin.subprocess.CREATE_NEW_PROCESS_GROUP", 512, create=True), \
             patch("nanobot.command.builtin.subprocess.Popen") as mock_popen, \
             patch("nanobot.command.builtin.os._exit") as mock_exit, \
             patch("nanobot.command.builtin.os.execv") as mock_execv:
            await cmd_restart(ctx)
            await scheduled[0]

        mock_popen.assert_called_once_with(
            [sys.executable, "-m", "nanobot"] + sys.argv[1:],
            creationflags=512,
        )
        mock_exit.assert_called_once_with(0)
        mock_execv.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_exit_mode_does_not_spawn(self, tmp_path):
        from nanobot.command.builtin import cmd_restart
        from nanobot.command.router import CommandContext

        loop, _bus = _make_loop(tmp_path)
        loop.restart_mode = "exit"
        msg = InboundMessage(channel="cli", sender_id="user", chat_id="direct", content="/restart")
        ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/restart", loop=loop)

        async def _fast_sleep(_delay: float) -> None:
            return None

        scheduled: list[asyncio.Task] = []
        fake_asyncio = SimpleNamespace(
            sleep=_fast_sleep,
            create_task=lambda coro: scheduled.append(asyncio.create_task(coro)) or scheduled[-1],
        )

        with patch("nanobot.command.builtin.asyncio", new=fake_asyncio), \
             patch("nanobot.command.builtin.subprocess.Popen") as mock_popen, \
             patch("nanobot.command.builtin.os._exit") as mock_exit, \
             patch("nanobot.command.builtin.os.execv") as mock_execv:
            await cmd_restart(ctx)
            await scheduled[0]

        mock_exit.assert_called_once_with(0)
        mock_popen.assert_not_called()
        mock_execv.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_intercepted_in_run_loop(self, tmp_path):
        loop, bus = _make_loop(tmp_path)
        loop.restart_mode = "exec"
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/restart")

        async def _fast_sleep(_delay: float) -> None:
            return None

        scheduled: list[asyncio.Task] = []

        def _capture_task(coro):
            task = asyncio.create_task(coro)
            scheduled.append(task)
            return task

        fake_asyncio = SimpleNamespace(
            sleep=_fast_sleep,
            create_task=_capture_task,
        )

        with patch.object(loop, "_enqueue_session_message", new_callable=MagicMock) as mock_dispatch, \
             patch("nanobot.command.builtin.asyncio", new=fake_asyncio), \
             patch("nanobot.command.builtin.os.execv"):
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(loop.run())
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_not_called()
            assert "正在重启" in out.content
            assert scheduled
            await scheduled[0]

    @pytest.mark.asyncio
    async def test_status_intercepted_in_run_loop(self, tmp_path):
        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")

        with patch.object(loop, "_enqueue_session_message", new_callable=MagicMock) as mock_dispatch:
            await bus.publish_inbound(msg)

            loop._running = True
            run_task = asyncio.create_task(loop.run())
            out = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
            loop._running = False
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass

            mock_dispatch.assert_not_called()
            assert "nanobot" in out.content.lower() or "模型" in out.content

    @pytest.mark.asyncio
    async def test_run_propagates_external_cancellation(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)

        run_task = asyncio.create_task(loop.run())
        await asyncio.sleep(0)
        run_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=1.0)

    @pytest.mark.asyncio
    async def test_help_includes_restart(self, tmp_path):
        loop, bus = _make_loop(tmp_path)
        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/help")

        response = await loop._process_message(msg)

        assert response is not None
        assert "/restart" in response.content
        assert "/status" in response.content
        assert response.metadata == {"render_as": "text"}

    @pytest.mark.asyncio
    async def test_status_reports_runtime_info(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}] * 3
        session.metadata = {
            "_last_usage": LLMUsage.reported(input_tokens=0, output_tokens=0).to_dict()
        }
        loop.sessions.state.get = AsyncMock(return_value=session)
        loop._start_time = time.time() - 125
        loop.consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(20500, "tiktoken")
        )
        loop.subagents.get_running_count_by_session.return_value = 0

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")
        runtime = loop.llm_runtime()
        loop.set_runtime_model("replacement-model")
        loop.set_runtime_context_window(10)
        loop.provider.generation = SimpleNamespace(
            temperature=1.0,
            max_tokens=1,
            reasoning_effort=None,
        )

        response = await loop._process_message(msg, runtime=runtime)

        assert response is not None
        assert "模型：test-model" in response.content
        assert "Token：输入 0 / 输出 0" in response.content
        assert "上下文：20k/200k（占输入预算 10%）" in response.content
        assert "会话消息：3 条" in response.content
        assert "运行时间：2m 5s" in response.content
        assert "正在运行的任务：0 个" in response.content
        assert response.metadata == {"render_as": "text"}
        loop.consolidator.estimate_session_prompt_tokens.assert_called_once_with(
            session,
            runtime=runtime,
        )

    @pytest.mark.asyncio
    async def test_status_counts_running_dispatch_and_subagent_tasks(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}]
        loop.sessions.state.get = AsyncMock(return_value=session)
        loop.consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(1000, "tiktoken")
        )

        running_task = MagicMock()
        running_task.done.return_value = False
        finished_task = MagicMock()
        finished_task.done.return_value = True

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")
        loop._active_tasks[msg.session_key] = {running_task, finished_task}
        loop.subagents.get_running_count_by_session.return_value = 2

        response = await loop._process_message(msg)

        assert response is not None
        assert "正在运行的任务：3 个" in response.content

    @pytest.mark.asyncio
    async def test_run_agent_loop_estimates_usage_when_provider_omits_it(self, monkeypatch, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        monkeypatch.setattr(
            "nanobot.agent.runner.estimate_prompt_tokens_chain",
            lambda *_args, **_kwargs: (123, "test"),
        )
        monkeypatch.setattr(
            "nanobot.agent.runner.estimate_message_tokens",
            lambda _message: 7,
        )
        loop.provider.chat_stream_with_retry = AsyncMock(side_effect=[
            LLMResponse(content="first", usage=LLMUsage.reported(input_tokens=9, output_tokens=4)),
            LLMResponse(content="second", usage=None),
        ])

        first = await loop._run_agent_loop(
            TranscriptInput(history=[], current_message=None),
            runtime=loop.llm_runtime(),
        )
        assert first.usage == LLMUsage.reported(input_tokens=9, output_tokens=4)

        second = await loop._run_agent_loop(
            TranscriptInput(history=[], current_message=None),
            runtime=loop.llm_runtime(),
        )
        assert second.usage == LLMUsage.estimated(input_tokens=123, output_tokens=7)

    @pytest.mark.asyncio
    async def test_status_falls_back_to_session_usage_when_context_estimate_missing(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        session = MagicMock()
        session.get_history.return_value = [{"role": "user"}]
        session.metadata = {
            "_last_usage": LLMUsage.reported(input_tokens=1200, output_tokens=34).to_dict()
        }
        loop.sessions.state.get = AsyncMock(return_value=session)
        loop.consolidator.estimate_session_prompt_tokens = MagicMock(
            return_value=(0, "none")
        )
        loop.subagents.get_running_count_by_session.return_value = 0

        response = await loop._process_message(
            InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/status")
        )

        assert response is not None
        assert "Token：输入 1200 / 输出 34" in response.content
        assert "上下文：1k/200k（占输入预算 0%）" in response.content
        assert "正在运行的任务：0 个" in response.content

    @pytest.mark.asyncio
    async def test_history_shows_recent_messages(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        session = MagicMock()
        session.get_history.return_value = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "tool", "content": "tool result"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am doing well."},
        ]
        loop.sessions.state.get = AsyncMock(return_value=session)

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/history")
        response = await loop._process_message(msg)

        assert response is not None
        assert "👤 你: Hello" in response.content
        assert "🤖 助手: Hi there!" in response.content
        assert "tool result" not in response.content
        assert response.metadata == {"render_as": "text"}

    @pytest.mark.asyncio
    async def test_history_respects_count_argument(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        session = MagicMock()
        session.get_history.return_value = [
            {"role": "user", "content": f"message {i}"} for i in range(20)
        ]
        loop.sessions.state.get = AsyncMock(return_value=session)

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/history 3")
        response = await loop._process_message(msg)

        assert response is not None
        assert "最近 3 条消息" in response.content
        assert "message 19" in response.content
        assert "message 0" not in response.content

    @pytest.mark.asyncio
    async def test_history_clamps_count_and_extracts_text_blocks(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        session = MagicMock()
        session.get_history.return_value = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "visible text"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
                ],
            },
            *({"role": "assistant", "content": f"reply {i}"} for i in range(60)),
        ]
        loop.sessions.state.get = AsyncMock(return_value=session)

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/history 999")
        response = await loop._process_message(msg)

        assert response is not None
        assert "最近 50 条消息" in response.content
        assert "visible text" not in response.content
        assert "reply 59" in response.content
        assert "reply 9" not in response.content

    @pytest.mark.asyncio
    async def test_history_invalid_count_returns_usage(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/history nope")
        response = await loop._process_message(msg)

        assert response is not None
        assert response.content.startswith("用法：/history [count]")

    @pytest.mark.asyncio
    async def test_history_empty_session(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        session = MagicMock()
        session.get_history.return_value = []
        loop.sessions.state.get = AsyncMock(return_value=session)

        msg = InboundMessage(channel="telegram", sender_id="u1", chat_id="c1", content="/history")
        response = await loop._process_message(msg)

        assert response is not None
        assert "暂无对话历史。" in response.content

    @pytest.mark.asyncio
    async def test_process_direct_preserves_render_metadata(self, tmp_path):
        loop, _bus = _make_loop(tmp_path)
        session = MagicMock()
        session.get_history.return_value = []
        loop.sessions.state.get = AsyncMock(return_value=session)
        loop.subagents.get_running_count.return_value = 0
        loop.subagents.get_running_count_by_session.return_value = 0

        response = await loop.process_direct("/status", session_key="cli:test")

        assert response is not None
        assert response.metadata == {"render_as": "text"}
