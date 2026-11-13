from __future__ import annotations

from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool

from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.mcp import MCPToolWrapper
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest
from nanobot.session.manager import SessionManager


async def test_model_discovers_loads_and_calls_mcp_with_session_isolation(tmp_path):
    sessions = SessionManager(tmp_path)
    provider = MagicMock(aclose=AsyncMock())
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings()
    provider.can_resume_conversation_state.return_value = False
    responses = iter([
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            "discover", "mcp_search", {"query": "find weather forecast", "limit": 1},
        )]),
        LLMResponse(content=None, tool_calls=[ToolCallRequest(
            "weather", "mcp_weather_forecast", {"city": "杭州"},
        )]),
        LLMResponse(content="晴天"),
        LLMResponse(content="继续"),
        LLMResponse(content="新会话"),
    ])
    requests = []

    async def respond(**kwargs):
        requests.append(deepcopy(kwargs["tools"]))
        return next(responses)

    provider.chat_stream_with_retry = AsyncMock(side_effect=respond)
    client = MagicMock()
    client.call_tool = AsyncMock(return_value=CallToolResult(
        content=[TextContent(type="text", text="晴天")],
    ))
    definition = MCPTool(name="forecast", description="Weather forecast by city", inputSchema={
        "type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"],
    })
    loop = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=tmp_path, model="test-model",
        session_manager=sessions,
    )
    wrapper = MCPToolWrapper(client, "weather", definition)
    loop.tools.register(wrapper)
    try:
        await loop.process_direct("查询杭州天气", session_key="cli:one")
        await loop.process_direct("继续", session_key="cli:one")
        await loop.process_direct("新会话", session_key="cli:two")
    finally:
        await loop.aclose()
    names = [{schema["function"]["name"] for schema in request} for request in requests]
    assert wrapper.name not in names[0]
    assert "mcp_search" in names[0]
    assert all(wrapper.name in names[index] for index in (1, 2, 3))
    assert wrapper.name not in names[4]
    assert wrapper.to_schema() in requests[1]
    assert {name for name in names[0] if not name.startswith("mcp_")} == {
        name for name in names[1] if not name.startswith("mcp_")
    }
    client.call_tool.assert_awaited_once_with("forecast", arguments={"city": "杭州"})
