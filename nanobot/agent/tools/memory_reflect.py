# Tool.execute accepts heterogeneous schemas.
# pyright: reportIncompatibleMethodOverride=false

"""Dream-only notes; the model cannot write the code-owned audit section."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from nanobot.agent.memory_dreams import DreamBatch
from nanobot.agent.tools.base import Tool, ToolResult, tool_parameters
from nanobot.agent.tools.schema import ArraySchema, StringSchema, tool_parameters_schema

if TYPE_CHECKING:
    from nanobot.agent.memory import MemoryStore
    from nanobot.agent.tools.context import ToolContext


@tool_parameters(tool_parameters_schema(
    section=StringSchema("Reflection or unverified questions.", enum=("反思", "待确认")),
    lines=ArraySchema(
        StringSchema("One plain-text note, without metadata.", min_length=1, max_length=500),
        description="Append up to ten concise notes to today's private Dream file.",
        min_items=1, max_items=10,
    ),
    required=["section", "lines"],
))
class MemoryReflectTool(Tool):
    def __init__(self, store: MemoryStore, batch: DreamBatch) -> None:
        self.store = store
        self.batch = batch

    @classmethod
    def enabled(cls, ctx: ToolContext) -> bool:
        return False

    @property
    def name(self) -> str:
        return "memory_reflect"

    @property
    def description(self) -> str:
        return (
            "Write Dream reflections or pending questions before extracting claims. These notes "
            "are unverified, are not evidence, and never enter ordinary recall. Audit is code-owned."
        )

    async def execute(
        self, section: Literal["反思", "待确认"], lines: list[str], **kwargs: Any,
    ) -> str:
        if any(
            not line.strip() or "\n" in line or "\r" in line
            or "<!--" in line or "-->" in line for line in lines
        ):
            return ToolResult.error("笔记必须是非空单行文本，不能包含 HTML 注释")
        try:
            self.store.reflect(section, [line.strip() for line in lines], batch=self.batch)
        except ValueError as exc:
            return ToolResult.error(f"无法保存反思：{exc}")
        return f"已写入今日 Dream 的“{section}”节，共 {len(lines)} 条。"
