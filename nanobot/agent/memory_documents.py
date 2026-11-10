"""Markdown codecs and atomic file operations for the memory source of truth."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterable
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Literal, get_args
from uuid import uuid4
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator

from nanobot.utils.atomic_file import atomic_write_text

ClaimKind = Literal[
    "preference", "fact", "commitment", "progress", "experience", "circumstance", "decision",
]
ClaimStatus = Literal["active", "superseded", "retracted", "expired"]
CLAIM_KINDS: tuple[ClaimKind, ...] = get_args(ClaimKind)
Clock = Callable[[], datetime]
MARKER_RE = re.compile(r"[ \t]*<!--\s*claim:(c_[0-9a-f]{12})(?:\s+(\{.*?\}))?\s*-->")
CLAIM_REF_RE = re.compile(r"<!--\s*claim:(c_[0-9a-f]{12})(?=\s|-->|$)")
CLAIM_COMMENT_RE = re.compile(r"[ \t]*<!--\s*claim:.*?(?:-->|$)", re.MULTILINE)
ENTITY_RE = re.compile(r"<!--\s*entity:(\S+)\s+(\{.*?\})\s*-->")
BULLET_RE = re.compile(r"^(?:[-*+]|\d+\.)\s+(?:\[[ xX]\]\s+)?")
SECTIONS: dict[ClaimKind, str] = {
    "preference": "## 已记住的偏好", "fact": "## 已记住的事实",
    "commitment": "## 承诺与待办", "progress": "## 进度",
    "experience": "## 经历", "circumstance": "## 近况", "decision": "## 决定",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_time(value: str | None, tz: str, field: str) -> str | None:
    if value is None:
        return None
    try:
        raw = value.strip()
        moment = (
            datetime.combine(date.fromisoformat(raw), time(23, 59, 59))
            if len(raw) == 10 else datetime.fromisoformat(raw.replace("Z", "+00:00"))
        )
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=ZoneInfo(tz))
        return iso(moment)
    except ValueError as exc:
        raise ValueError(f"{field} 日期无效，请使用 YYYY-MM-DD 或 ISO 8601 日期时间") from exc


def local_day(value: str, tz: str = "UTC") -> str:
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return moment.astimezone(ZoneInfo(tz)).date().isoformat()


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_claim_markers(text: str) -> str:
    return ENTITY_RE.sub("", CLAIM_COMMENT_RE.sub("", text))


def marked_claim_ids(line: str) -> set[str]:
    """Keep IDs addressable even when a hand-edited comment has invalid metadata."""
    return set(CLAIM_REF_RE.findall(line))


def line_text(line: str) -> str:
    return BULLET_RE.sub("", strip_claim_markers(line).strip())


def claim_marker(claim_id: str) -> str:
    """Legacy marker, still accepted on hand-maintained pages."""
    return f" <!-- claim:{claim_id} -->"


def is_claim_id(value: str) -> bool:
    return re.fullmatch(r"c_[0-9a-f]{12}", value) is not None


class Evidence(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    quote: str | None = None
    source_ref: str | None = None
    session_key: str | None = None
    message_id: str | None = None


class ClaimMetadata(BaseModel):
    """Missing legacy metadata stays unknown; future fields survive rewrites."""

    model_config = ConfigDict(extra="allow", frozen=True)
    v: Literal[1] = 1
    kind: ClaimKind | None = None
    status: ClaimStatus = "active"
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: Evidence = Field(default_factory=Evidence)
    created_at: str | None = None
    updated_at: str | None = None
    due_at: str | None = None
    expires_at: str | None = None
    supersedes: tuple[str, ...] = ()
    person_id: str | None = None
    project_id: str | None = None

    @field_validator("created_at", "updated_at", "due_at", "expires_at")
    @classmethod
    def timestamp(cls, value: str | None, info: ValidationInfo) -> str | None:
        return parse_time(value, "UTC", str(info.field_name))

    @field_validator("supersedes")
    @classmethod
    def ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not is_claim_id(value) for value in values):
            raise ValueError("supersedes 包含无效记忆 ID")
        return tuple(dict.fromkeys(values))


class ClaimDraft(BaseModel):
    """Validated tool input. The lifecycle store consumes this contract directly."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    text: str = Field(min_length=1, max_length=500)
    kind: ClaimKind
    quote: str | None = Field(default=None, max_length=500)
    confidence: float | None = Field(default=None, ge=0, le=1)
    core: bool = False
    due_at: str | None = None
    expires_at: str | None = None
    supersedes: tuple[str, ...] = ()
    person_id: str | None = None
    project_id: str | None = None

    @field_validator("text", "quote")
    @classmethod
    def one_line(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = " ".join(value.split())
        if not text or "<!--" in text or "-->" in text:
            raise ValueError("记忆正文和引文不能为空或包含 HTML 注释")
        return text

    @field_validator("due_at", "expires_at")
    @classmethod
    def timestamp(cls, value: str | None, info: ValidationInfo) -> str | None:
        tz = info.context["timezone"] if info.context else "UTC"
        return parse_time(value, tz, str(info.field_name))

    @field_validator("supersedes")
    @classmethod
    def ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return ClaimMetadata.ids(values)


class Claim(ClaimMetadata):
    claim_id: str
    text: str
    source_file: str | None = None
    source_line: int | None = None
    is_core: bool = False

    @property
    def quote(self) -> str | None:
        return self.evidence.quote

    @property
    def session_key(self) -> str | None:
        return self.evidence.session_key

    @property
    def message_id(self) -> str | None:
        return self.evidence.message_id

    def metadata_json(self) -> str:
        data = self.model_dump(exclude={
            "claim_id", "text", "source_file", "source_line", "is_core",
        })
        # JSON strings must not terminate the surrounding HTML comment.
        encoded = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return encoded.replace("<", "\\u003c").replace(">", "\\u003e")

    def to_public(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id, "text": self.text, "kind": self.kind,
            "status": self.status, "confidence": self.confidence, "quote": self.quote,
            "source": f"{self.source_file}:{self.source_line}" if self.source_file else None,
            "supersedes": list(self.supersedes), "created_at": self.created_at,
            "due_at": self.due_at, "expires_at": self.expires_at,
            "person_id": self.person_id, "project_id": self.project_id,
        }


def encode_claim(claim: Claim, *, timezone: str = "UTC") -> str:
    citation = ""
    if claim.created_at is not None and claim.evidence.source_ref is not None:
        day = local_day(claim.created_at, timezone)[5:]
        citation = f" 〔{day} · {claim.evidence.source_ref}〕"
    return f"- {claim.text}{citation} <!-- claim:{claim.claim_id} {claim.metadata_json()} -->"


def decode_claim(
    line: str, path: str, number: int, heading: str = "", *, issues: set[str],
) -> Claim | None:
    matches = list(MARKER_RE.finditer(line))
    if not matches:
        if "<!--" in line and "claim:" in line:
            issues.add("记忆标记格式无效")
        return None
    if len(matches) != 1 or line[matches[0].end():].strip():
        issues.add("记忆标记格式无效")
        return None
    marker = matches[0]
    if marker.group(2):
        try:
            metadata = ClaimMetadata.model_validate_json(marker.group(2))
        except ValidationError:
            issues.add("记忆元数据无效")
            return None
    else:
        kind: ClaimKind | None = next(
            (kind for kind, section in SECTIONS.items() if section == heading), None,
        )
        if path == "USER.md":
            kind = "preference"
        metadata = ClaimMetadata(kind=kind)
    text = line_text(line)
    if metadata.evidence.source_ref is not None:
        citation = r" 〔\d{2}-\d{2} · " + re.escape(metadata.evidence.source_ref) + r"〕$"
        text = re.sub(citation, "", text)
    data = dict(metadata)
    data.update({
        "claim_id": marker.group(1), "text": text,
        "source_file": path, "source_line": number,
        "is_core": path in {"USER.md", "memory/MEMORY.md"},
    })
    return Claim.model_construct(**data)


class Entity(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)
    v: Literal[1] = 1
    entity_id: str = Field(default_factory=lambda: f"e_{uuid4().hex[:12]}")
    kind: Literal["person", "project"]
    display_name: str
    aliases: tuple[str, ...] = ()
    status: Literal["active", "completed", "paused"] = "active"


def encode_entity(entity: Entity) -> str:
    data = entity.model_dump_json(exclude={"entity_id"})
    data = data.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"<!-- entity:{entity.entity_id} {data} -->"


def decode_entity(text: str) -> Entity | None:
    matches = list(ENTITY_RE.finditer(text))
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("人物或项目页只能有一个实体标记")
    data = json.loads(matches[0].group(2))
    return Entity.model_validate({**data, "entity_id": matches[0].group(1)})


class LedgerEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    v: Literal[1] = 1
    event_id: str = Field(default_factory=lambda: f"ev_{uuid4().hex[:12]}")
    target_type: Literal["claim", "chunk", "history"]
    target_id: str
    action: Literal["remember", "supersede", "expire", "forget"]
    actor: str
    at: str
    run_id: str | None = None
    source_ref: str | None = None


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def redact_history_lines(content: str, cursor: int, tombstones: set[tuple[str, str]]) -> str:
    """Apply durable line tombstones without changing the rest of a summary."""
    lines: list[str] = []
    for raw in content.splitlines(keepends=True):
        body = raw.rstrip("\r\n")
        if ("history", f"{cursor}:{digest(body)}") in tombstones:
            lines.append("[已遗忘]" + raw[len(body):])
        else:
            lines.append(raw)
    return "".join(lines)


def insert_lines(content: str, heading: str, additions: Iterable[str]) -> str:
    lines = content.splitlines()
    start = next((i for i, value in enumerate(lines) if value == heading), None)
    if start is None:
        prefix = content.rstrip("\n") + ("\n\n" if content else "")
        return prefix + heading + "\n" + "\n".join(additions) + "\n"
    end = start + 1
    while end < len(lines) and not lines[end].startswith("#"):
        end += 1
    while end > start + 1 and not lines[end - 1].strip():
        end -= 1
    lines[end:end] = additions
    return "\n".join(lines) + "\n"


class MemoryDocuments:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.ledger = workspace / "memory/ledger.md"

    def files(self) -> list[Path]:
        paths = [self.workspace / "USER.md", self.workspace / "memory/MEMORY.md"]
        for folder in ("bank", "people", "projects", "daily"):
            paths.extend(sorted((self.workspace / "memory" / folder).glob("*.md")))
        return [path for path in paths if path.is_file()]

    def snapshot(self) -> dict[str, str]:
        return {
            path.relative_to(self.workspace).as_posix(): read_text(path) for path in self.files()
        }

    def events(self) -> list[LedgerEvent]:
        return [
            LedgerEvent.model_validate_json(line[2:])
            for line in read_text(self.ledger).splitlines() if line.startswith("- ")
        ]

    def tombstones(self) -> set[tuple[str, str]]:
        return {
            (event.target_type, event.target_id)
            for event in self.events() if event.action == "forget"
        }

    def append_events(self, events: Iterable[LedgerEvent]) -> None:
        additions = "".join(f"- {event.model_dump_json()}\n" for event in events)
        if additions:
            atomic_write_text(self.ledger, (read_text(self.ledger) or "# 记忆操作记录\n\n") + additions)

    def apply(self, changes: dict[str, str]) -> None:
        for relative, content in changes.items():
            atomic_write_text(self.workspace / relative, content)


class DailyNotes:
    def __init__(
        self, memory_dir: Path, *, lock: threading.RLock,
        timezone: str = "UTC", clock: Clock = utc_now,
    ) -> None:
        self.dir = memory_dir / "daily"
        self.timezone = timezone
        self.clock = clock
        self.lock = lock

    def path_for(self, day: date | None = None) -> Path:
        current = (
            day if day is not None else self.clock().astimezone(ZoneInfo(self.timezone)).date()
        )
        return self.dir / f"{current.isoformat()}.md"

    def files(self) -> list[Path]:
        return sorted(self.dir.glob("*.md"))

    def append(
        self, heading: str, lines: Iterable[str], *, day: date | None = None,
    ) -> Path | None:
        body = [line for line in lines if line.strip()]
        if not body:
            return None
        with self.lock:
            path = self.path_for(day)
            content = read_text(path) or f"# {path.stem}\n"
            atomic_write_text(path, insert_lines(content, heading, body))
        return path

    def append_summary(self, summary: str, *, session_key: str | None = None) -> Path | None:
        lines = [
            strip_claim_markers(re.sub(r"^(\s*[-*]\s*)\[[a-z]+\]\s*", r"\1", line))
            for line in summary.splitlines() if line.strip() and line.strip() != "(nothing)"
        ]
        source = f" · {session_key}" if session_key else ""
        moment = self.clock().astimezone(ZoneInfo(self.timezone))
        return self.append(f"## {moment:%H:%M} 会话摘要{source}", lines)
