import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.bus.events import OutboundMessage
from nanobot.command.builtin import cmd_stop
from nanobot.command.router import CommandContext


@pytest.mark.asyncio
async def test_cmd_stop_drains_pending_queue():
    mock_loop = MagicMock()
    mock_loop._cancel_active_tasks = AsyncMock(return_value=1)
    mock_loop._pending_queues = {}

    pending = asyncio.Queue()
    await pending.put("msg1")
    await pending.put("msg2")
    mock_loop._pending_queues["test-session"] = pending

    ctx = CommandContext(
        msg=MagicMock(channel="websocket", chat_id="test-chat", metadata={}),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    result = await cmd_stop(ctx)

    assert isinstance(result, OutboundMessage)
    assert "已停止 3 个任务" in result.content
    assert "test-session" not in mock_loop._pending_queues


@pytest.mark.asyncio
async def test_cmd_stop_with_empty_pending_queue():
    mock_loop = MagicMock()
    mock_loop._cancel_active_tasks = AsyncMock(return_value=2)
    mock_loop._pending_queues = {}

    pending = asyncio.Queue()
    mock_loop._pending_queues["test-session"] = pending

    ctx = CommandContext(
        msg=MagicMock(channel="websocket", chat_id="test-chat", metadata={}),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    result = await cmd_stop(ctx)

    assert "已停止 2 个任务" in result.content
    assert "test-session" not in mock_loop._pending_queues


@pytest.mark.asyncio
async def test_cmd_stop_no_pending_queue():
    mock_loop = MagicMock()
    mock_loop._cancel_active_tasks = AsyncMock(return_value=0)
    mock_loop._pending_queues = {}

    ctx = CommandContext(
        msg=MagicMock(channel="websocket", chat_id="test-chat", metadata={}),
        session=None,
        key="test-session",
        raw="/stop",
        loop=mock_loop,
    )

    result = await cmd_stop(ctx)

    assert "没有正在运行的任务" in result.content
