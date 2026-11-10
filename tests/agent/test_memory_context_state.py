"""Forgetting invalidates persisted and in-flight provider continuations."""

from __future__ import annotations

import json
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory_documents import ClaimDraft
from nanobot.agent.tools.memory_claims import MemoryForgetTool, MemorySearchTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import (
    GenerationSettings,
    LLMProvider,
    LLMResponse,
    ProviderConversationState,
    ToolCallRequest,
)
from nanobot.runtime_context import RuntimeContextBlock, wrap_runtime_context_lines
from nanobot.session.manager import SessionManager


@pytest.mark.parametrize("same_turn", [True, False], ids=["tool-loop", "next-turn"])
@pytest.mark.parametrize("core", [True, False], ids=["core", "recall"])
async def test_memory_forget_invalidates_provider_state_and_model_projection(
    tmp_path, monkeypatch, same_turn, core,
):
    provider = MagicMock(spec=LLMProvider)
    provider.provider_name = "test"
    provider.generation = GenerationSettings(max_tokens=1024)
    provider.estimate_prompt_tokens = MagicMock(return_value=(100, "test"))
    provider.can_resume_conversation_state.return_value = True
    provider.supports_native_compaction.return_value = False
    provider.aclose = AsyncMock()
    workspace = tmp_path / "workspace"
    sessions = SessionManager(workspace, sessions_root=tmp_path / "sessions")
    agent = AgentLoop(
        bus=MessageBus(), provider=provider, workspace=workspace, model="test-model",
        session_manager=sessions, tool_registry=ToolRegistry(),
    )
    for name in agent.tools.tool_names:
        agent.tools.unregister(name)
    store = agent.context.memory
    agent.tools.register(MemoryForgetTool(store))
    agent.tools.register(MemorySearchTool(store))
    saved = store.claims.remember(ClaimDraft(
        text="私人门禁口令是蓝鲸松林", kind="fact", core=core,
    )).claim
    other = RuntimeContextBlock("test-route", wrap_runtime_context_lines(["keep-route-context"]))

    async def route_context(_request):
        return other

    agent.register_runtime_context_provider(route_context)
    calls = []

    async def chat(**kwargs):
        calls.append(deepcopy(kwargs))
        state = ProviderConversationState(
            kind="test", provider="test", model="test-model", version=1,
            payload={"opaque_context": "server-owned"},
        )
        if same_turn and len(calls) == 1:
            return LLMResponse(content=None, tool_calls=[ToolCallRequest(
                id="forget", name="memory_forget", arguments={"claim_ids": [saved.claim_id]},
            )], provider_state=state)
        return LLMResponse(content="已完成", provider_state=state)

    provider.chat_stream_with_retry = AsyncMock(side_effect=chat)
    sync = MagicMock(wraps=store.claims.index.sync)
    monkeypatch.setattr(store.claims.index, "sync", sync)
    try:
        await agent.process_direct("查看门禁口令备忘", session_key="cli:memory")
        assert "蓝鲸松林" in json.dumps(calls[0]["messages"], ensure_ascii=False)
        if not same_turn:
            sync.assert_called_once()
            snapshot = sessions.read_session_snapshot("cli:memory")
            assert snapshot.provider_state.context_generation == ""
            original_messages = deepcopy(snapshot.messages)
            store.forget_memory(claim_ids=[saved.claim_id])
            assert sessions.read_session_snapshot("cli:memory").messages == original_messages
            await agent.process_direct("现在继续", session_key="cli:memory")
        assert len(calls) == 2
        assert calls[1]["provider_context"].conversation_state is None
        replay = json.dumps(calls[1]["messages"], ensure_ascii=False)
        assert "蓝鲸松林" not in replay and "keep-route-context" in replay
        snapshot = sessions.read_session_snapshot("cli:memory")
        assert snapshot.provider_state.context_generation == store.generation()
        assert store.generation()
        # The next unchanged turn may resume the newly accepted, clean state.
        await agent.process_direct("继续", session_key="cli:memory")
        assert calls[2]["provider_context"].conversation_state is not None
        assert calls[2]["provider_context"].conversation_state.context_generation == store.generation()
    finally:
        await agent.aclose()
