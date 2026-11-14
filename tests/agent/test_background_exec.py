from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.background_exec import BackgroundExec
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.exec_session import ExecSessionManager, ExecSessionTool
from nanobot.agent.tools.shell import ExecTool
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.outbox import DeliveryOutbox
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager


def _command(source: str) -> str:
    if sys.platform == "win32":
        return "& " + subprocess.list2cmdline([sys.executable, "-c", source])
    return shlex.join([sys.executable, "-c", source])


@pytest.mark.asyncio
async def test_background_command_result_and_isolation(tmp_path):
    bus = MessageBus()
    sessions = SessionManager(tmp_path)
    session = await sessions.state.get("weixin.bot:chat")
    await sessions.state.prepare_input(session)
    jobs = BackgroundExec(sessions, bus)
    manager = ExecSessionManager(background_exec=jobs)
    shell = ExecTool(working_dir=str(tmp_path), session_manager=manager)
    status = ExecSessionTool(manager=manager)
    ctx = RequestContext(channel="weixin.bot", chat_id="chat",
                         session_key=session.key, session_generation=session.generation)
    try:
        with request_context(ctx):
            started = json.loads(await shell.execute(
                command=_command("import time,sys; time.sleep(0.1); print('x'*60000); print('done'); sys.exit(7)"),
                background=True,
            ))
        assert started["status"] == "starting"
        task_id = started["task_id"]
        # 发起工具的上下文已退出，原生命令仍可完成并产生结构化事件。
        event = await asyncio.wait_for(bus.consume_inbound(), 5)
        assert event.session_key == session.key
        assert event.metadata["background_exec_event"] == {
            "task_id": task_id, "status": "failed", "exit_code": 7, "finish_reason": "exit",
        }
        with request_context(ctx):
            result = json.loads(await status.execute(session_id=task_id))
            again = json.loads(await status.execute(session_id=task_id))
        assert result["status"] == "failed" and result["exit_code"] == 7
        assert len(result["output"]) <= 10000
        assert result["output"].endswith(f"done{os.linesep}") and result["truncated_chars"] > 0
        assert again["output"] == result["output"]
        with request_context(replace(ctx, session_key="weixin.bot:other")):
            assert "not found" in await status.execute(session_id=task_id)
    finally:
        await manager.close_all()
        await sessions.state.aclose()


@pytest.mark.asyncio
async def test_cancel_background_command(tmp_path):
    bus = MessageBus()
    sessions = SessionManager(tmp_path)
    session = await sessions.state.get("cli:cancel")
    await sessions.state.prepare_input(session)
    manager = ExecSessionManager(background_exec=BackgroundExec(sessions, bus))
    shell = ExecTool(working_dir=str(tmp_path), session_manager=manager)
    ctx = RequestContext(channel="cli", chat_id="cancel",
                         session_key=session.key, session_generation=session.generation)
    try:
        with request_context(ctx):
            started = json.loads(await shell.execute(
                command=_command("import time; time.sleep(30)"), background=True,
            ))
            task_id = started["task_id"]
            running = await manager._starting[task_id][1]
            result = json.loads(await ExecSessionTool(manager=manager).execute(
                session_id=task_id, terminate=True,
            ))
        assert running.process.returncode is not None
        assert result["status"] == "cancelled"
        assert result["finish_reason"] == "cancelled"
        assert (await bus.consume_inbound()).metadata["background_exec_event"]["status"] == "cancelled"
    finally:
        await manager.close_all()
        await sessions.state.aclose()


@pytest.mark.asyncio
async def test_background_completion_resumes_originating_conversation(tmp_path):
    bus = MessageBus()
    outbox = DeliveryOutbox(tmp_path / "delivery.sqlite3")
    bus.outbound_recorder = outbox.record
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    calls = 0
    task_id = ""

    async def chat(**kwargs):
        nonlocal calls, task_id
        calls += 1
        if calls == 1:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="start", name="exec", arguments={
                    "command": _command("import time; time.sleep(0.2); print('native-result')"),
                    "background": True,
                },
            )])
        if calls == 2:
            task_id = json.loads(kwargs["messages"][-1]["content"])["task_id"]
            return LLMResponse(content="后台已启动")
        if calls == 3:
            assert task_id in kwargs["messages"][-1]["content"]
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="read", name="exec_session", arguments={"session_id": task_id},
            )])
        result = json.loads(kwargs["messages"][-1]["content"])
        assert result["exit_code"] == 0 and result["output"] == f"native-result{os.linesep}"
        return LLMResponse(content="命令完成，已读取结果")

    provider.chat_with_retry = chat
    provider.chat_stream_with_retry = chat
    loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")
    worker = asyncio.create_task(loop.run())

    async def reply():
        while True:
            msg = await bus.consume_outbound()
            if msg.event is None and msg.content:
                return msg

    try:
        await bus.publish_inbound(InboundMessage(
            channel="weixin.bot", sender_id="user", chat_id="chat", content="后台运行命令",
        ))
        first = await asyncio.wait_for(reply(), 5)
        final = await asyncio.wait_for(reply(), 5)
        assert first.content == "后台已启动"
        assert final.content == "命令完成，已读取结果"
        assert (final.channel, final.chat_id) == ("weixin.bot", "chat")
        assert final.delivery is not None and final.delivery.run_id == task_id
        # 命令成功且事件被消费；用户回复仅进入渠道 outbox，未伪称已经送达。
        assert outbox.ready()[0].content == final.content
        session = await loop.sessions.state.read("weixin.bot:chat")
        assert sum(row.get("background_exec_id") == task_id for row in session.messages) == 1
        assert calls == 4
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
