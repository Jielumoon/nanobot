"""Dream batches and private reflection notes, outside the recall corpus."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from uuid import uuid4

from nanobot.agent.memory_documents import (
    Clock,
    MemoryDocuments,
    digest,
    insert_lines,
    iso,
    line_text,
    local_day,
    marked_claim_ids,
    read_text,
)
from nanobot.utils.atomic_file import atomic_write_text


@dataclass(frozen=True)
class DreamSource:
    cursor: int
    timestamp: str
    session_key: str | None
    lines: tuple[str, ...]


@dataclass(frozen=True)
class DreamBatch:
    prompt: str
    cursor: int
    sources: dict[str, DreamSource]
    run_id: str = field(default_factory=lambda: uuid4().hex)


class DreamNotes:
    def __init__(
        self, documents: MemoryDocuments, *, lock: threading.RLock, clock: Clock, timezone: str,
    ) -> None:
        self.documents = documents
        self.directory = documents.workspace / "dreams"
        self.lock = lock
        self.clock = clock
        self.timezone = timezone

    def path(self) -> Path:
        return self.directory / f"{local_day(iso(self.clock()), self.timezone)}.md"

    def append(self, section: Literal["反思", "待确认", "变更审计"], lines: list[str]) -> None:
        with self.lock:
            path = self.path()
            content = read_text(path) or f"# {path.stem}\n"
            atomic_write_text(path, insert_lines(content, f"## {section}", lines))

    def pending(self, *, budget: int = 1800) -> str:
        with self.lock:
            tombstones = self.documents.tombstones()
            kept: list[str] = []
            used = 0
            for path in sorted(self.directory.glob("*.md"), reverse=True):
                section = ""
                for raw in read_text(path).splitlines():
                    if raw.startswith("#"):
                        section = raw
                        continue
                    text = line_text(raw)
                    if section != "## 待确认" or not text:
                        continue
                    if ("chunk", digest(text)) in tombstones or any(
                        ("claim", claim_id) in tombstones for claim_id in marked_claim_ids(raw)
                    ):
                        continue
                    line = f"- {path.stem}: {text}"
                    if used + len(line) + 1 > budget:
                        return "\n".join(kept)
                    kept.append(line)
                    used += len(line) + 1
            return "\n".join(kept)
