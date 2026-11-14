from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from mcp import types
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.background_exec import BackgroundExec
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.exec_session import ExecSessionManager, ExecSessionTool
from nanobot.agent.tools.mcp import MCPToolWrapper
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.events import InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager


@pytest.mark.asyncio
async def test_discovered_t3_background_wait_resumes_original_session(tmp_path):
    server = FastMCP("本地 T3 等待检查")
    remote_started = asyncio.Event()
    remote_finished = asyncio.Event()
    waits = []

    @server.tool()
    async def t3_thread_wait(
        threadId: str, runId: str | None = None, timeoutMs: int = 60000,  # noqa: N803
    ) -> types.CallToolResult:
        waits.append((threadId, runId))
        remote_started.set()
        if len(waits) == 1:
            return types.CallToolResult(content=[types.TextContent(type="text", text="仍在运行")],
                structuredContent={"threadId": threadId, "runId": "run-a", "status": "running",
                                   "timedOut": True, "next": "Completion wakes this thread, so end your turn. Do not wait on it or poll it."})
        await remote_finished.wait()
        return types.CallToolResult(content=[types.TextContent(type="text", text="远端交付已就绪")],
            structuredContent={"threadId": threadId, "runId": "run-a", "status": "completed",
                               "timedOut": False, "resultRun": {"runId": "run-a", "status": "completed"}})

    async with create_connected_server_and_client_session(server) as client:
        tool = MCPToolWrapper(client, "t3", (await client.list_tools()).tools[0])
        bus = MessageBus()
        provider = MagicMock()
        provider.get_default_model.return_value = "test-model"
        provider.generation = GenerationSettings()
        calls = 0
        task_id = ""

        async def chat(**kwargs):
            nonlocal calls, task_id
            calls += 1
            schemas = {schema["function"]["name"]: schema for schema in kwargs["tools"]}
            if calls <= 2:
                assert tool.name not in schemas
                assert "mcp_search" in schemas
            else:
                assert schemas[tool.name] == tool.to_schema()
                assert "_nanobot_background" in schemas[tool.name]["function"]["parameters"]["properties"]
            if calls in (1, 3):
                if calls == 3:
                    found = json.loads(kwargs["messages"][-1]["content"])
                    assert [item["name"] for item in found["loaded"]] == [tool.name]
                return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id=f"wait-{calls}", name=tool.name, arguments={"threadId": "thread-a", "timeoutMs": 1000,
                    "_nanobot_background": {"timeout_s": 10}},
                )])
            if calls == 2:
                assert "not found" in kwargs["messages"][-1]["content"]
                assert waits == []
                return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id="discover", name="mcp_search",
                    arguments={"query": tool.name, "limit": 1},
                )])
            if calls == 4:
                task_id = json.loads(kwargs["messages"][-1]["content"])["task_id"]
                return LLMResponse(content="已交给后台等待")
            if calls == 5:
                assert task_id in kwargs["messages"][-1]["content"]
                return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                    id="read", name="exec_session", arguments={"session_id": task_id},
                )])
            result = json.loads(kwargs["messages"][-1]["content"])
            assert result["kind"] == "mcp" and result["status"] == "succeeded"
            assert result["mcp_result"]["structuredContent"]["resultRun"]["runId"] == "run-a"
            assert result["output"] == "远端交付已就绪"
            return LLMResponse(content="已读取远端交付并接续")

        provider.chat_with_retry = chat
        provider.chat_stream_with_retry = chat
        loop = AgentLoop(bus=bus, provider=provider, workspace=tmp_path, model="test-model")
        loop.tools.register(tool)
        worker = asyncio.create_task(loop.run())

        async def reply():
            while True:
                msg = await bus.consume_outbound()
                if msg.event is None and msg.content:
                    return msg

        try:
            await bus.publish_inbound(InboundMessage(
                channel="weixin.bot", sender_id="user", chat_id="origin", content="等待远端任务完成后接续",
            ))
            assert (await asyncio.wait_for(reply(), 5)).content == "已交给后台等待"
            await asyncio.wait_for(remote_started.wait(), 5)
            assert not remote_finished.is_set()
            remote_finished.set()
            final = await asyncio.wait_for(reply(), 5)
            assert (final.channel, final.chat_id) == ("weixin.bot", "origin")
            assert final.content == "已读取远端交付并接续"
            assert waits == [("thread-a", None), ("thread-a", "run-a")]
            assert calls == 6
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_background_mcp_outlasts_default_thirty_second_timeout(tmp_path):
    server = FastMCP("本地长等待检查")
    remote_calls = 0

    @server.tool()
    async def wait_once() -> types.CallToolResult:
        nonlocal remote_calls
        remote_calls += 1
        await asyncio.sleep(30.1)
        return types.CallToolResult(content=[types.TextContent(type="text", text="超过默认三十秒后返回")],
                                    structuredContent={"status": "running"})

    async with create_connected_server_and_client_session(
        server, read_timeout_seconds=timedelta(seconds=30),
    ) as client:
        tool = MCPToolWrapper(client, "local", (await client.list_tools()).tools[0])
        assert tool._tool_timeout == 30
        registry = ToolRegistry()
        registry.register(tool)
        bus = MessageBus()
        sessions = SessionManager(tmp_path)
        session = await sessions.state.get("cli:origin")
        await sessions.state.prepare_input(session)
        jobs = BackgroundExec(sessions, bus)
        manager = ExecSessionManager(background_exec=jobs)
        ctx = RequestContext(channel="cli", chat_id="origin", session_key=session.key,
                             session_generation=session.generation, background_exec=jobs)
        try:
            with request_context(ctx):
                started = json.loads(await registry.execute(tool.name, {
                    "_nanobot_background": {"timeout_s": 40},
                }))
            assert started["status"] == "starting"
            event = await asyncio.wait_for(bus.consume_inbound(), 40)
            assert event.session_key == session.key
            with request_context(ctx):
                result = json.loads(await ExecSessionTool(manager=manager).execute(
                    session_id=started["task_id"],
                ))
            assert result["status"] == "succeeded" and result["exit_code"] is None
            assert result["mcp_result"]["structuredContent"] == {"status": "running"}
            assert remote_calls == 1
        finally:
            await manager.close_all()
            await sessions.state.aclose()
