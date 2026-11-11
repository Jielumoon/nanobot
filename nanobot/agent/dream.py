"""Workspace-scoped Dream execution shared by scheduled and manual entrypoints."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_coordination import (
    acquire_memory_operation,
    memory_operation_running,
    release_memory_operation,
)

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop


@dataclass(frozen=True)
class DreamResult:
    status: str
    elapsed: float = 0
    cursor: int | None = None
    changed: bool = False
    sha: str | None = None
    reason: str = ""


def dream_running(workspace: Path) -> bool:
    return memory_operation_running(workspace)


async def run_dream(
    agent: AgentLoop, *, trigger: Literal["manual", "scheduled"],
    before_run: Callable[[], Awaitable[None]] | None = None,
) -> DreamResult:
    store = agent.context.memory
    workspace = store.workspace.resolve()
    if not acquire_memory_operation(workspace):
        return DreamResult("busy")
    run_id = uuid4().hex
    started = time.monotonic()
    status = "failed"
    cursor = None
    reason = ""
    prefix = "dream: manual run" if trigger == "manual" else "dream: periodic memory consolidation"

    async def silent(*_args: Any, **_kwargs: Any) -> None:
        pass

    try:
        try:
            store.git.init()
            store.maintain_memory(run_id=run_id)
            batch = store.build_dream_prompt(run_id=run_id)
            if batch is None:
                status = "empty"
            else:
                if before_run is not None:
                    await before_run()
                response = await agent.process_direct(
                    batch.prompt, session_key=f"dream:{run_id}", ephemeral=True,
                    tools=store.build_dream_tools(batch), on_progress=silent,
                    runtime=agent.dream_runtime(),
                )
                if MemoryStore.dream_run_completed(response):
                    status = "completed"
                    cursor = batch.cursor
                else:
                    status = "incomplete"
                    reason = MemoryStore.dream_incompletion_reason(response)
        finally:
            try:
                store.compact_history()
                await agent.sessions.state.prune_dream_sessions()
                await store.backfill_embeddings()
            finally:
                changed, sha = store.finish_dream(
                    run_id=run_id, status=status, cursor=cursor, prefix=prefix,
                )
        return DreamResult(status, time.monotonic() - started, cursor, changed, sha, reason)
    finally:
        release_memory_operation(workspace)
