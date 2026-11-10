"""Invalidate in-flight memory projections after a durable forget operation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.runtime_context import replay_model_memory

if TYPE_CHECKING:
    from nanobot.agent.context import ContextBuilder


class MemoryContextRefresh:
    def __init__(
        self, builder: ContextBuilder, generation: str, *, channel: str,
        workspace: Path, include_memory: bool,
    ) -> None:
        self.builder = builder
        self.generation = generation
        self.channel = channel
        self.workspace = workspace
        self.include_memory = include_memory

    def __call__(
        self, messages: list[dict[str, Any]], accepted: list[dict[str, Any]], summary: str | None,
    ) -> str | None:
        if self.builder.memory.generation() == self.generation:
            return None
        view = self.builder.memory.prompt_view()
        system = self.builder.build_system_prompt(
            channel=self.channel, workspace=self.workspace, include_memory=self.include_memory,
            memory_view=view,
            session_summary={"text": summary, "last_active": datetime.now().astimezone().isoformat()}
            if summary is not None else None,
        )
        for transcript in (messages, accepted):
            transcript[:] = [replay_model_memory(message) for message in transcript]
            transcript[0] = {"role": "system", "content": system}
        self.generation = view.generation
        return self.generation
