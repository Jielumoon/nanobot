"""Claim ledger, daily notes, and search index for structured long-term memory.

Markdown files remain the source of truth: every active claim is one bullet in
``USER.md``, ``memory/MEMORY.md``, or a daily note, tagged with an HTML comment
marker. ``memory/claims.sqlite3`` only stores claim metadata and a derived
search index that is rebuilt from the files whenever they change.

The module uses the standard-library ``sqlite3`` driver with short-lived
connections, so nothing stays resident between calls.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
from collections.abc import Generator, Iterable, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Literal, cast, get_args
from uuid import uuid4

from loguru import logger

ClaimKind = Literal["preference", "fact", "commitment", "progress"]
ClaimStatus = Literal["active", "superseded", "retracted", "expired"]
CLAIM_KINDS: tuple[ClaimKind, ...] = get_args(ClaimKind)
CLAIM_STATUSES: tuple[ClaimStatus, ...] = get_args(ClaimStatus)

_MARKER_RE = re.compile(r"[ \t]*<!--\s*claim:(c_[0-9a-f]{12})\s*-->")
_CLAIM_ID_RE = re.compile(r"^c_[0-9a-f]{12}$")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_BULLET_RE = re.compile(r"^(?:[-*+]|\d+\.)\s+(?:\[[ xX]\]\s+)?")
_MAX_CLAIM_CHARS = 500
_MAX_QUOTE_CHARS = 500
_MAX_QUERY_TERMS = 32
_CANDIDATE_LIMIT = 200
_SNIPPET_CHARS = 300
_FORGOTTEN = "[已遗忘]"
# Frequent function words carry no topic signal for recall.
_CJK_STOP_BIGRAMS = frozenset({
    "什么", "怎么", "这个", "那个", "我们", "你们", "他们", "一下", "可以", "不是",
    "就是", "还是", "没有", "现在", "如果", "因为", "所以", "然后", "一个", "请问",
    "帮我", "告诉", "知道", "我的", "你的", "是不", "的是", "了吗", "吗？", "一些",
})
_ASCII_STOP_WORDS = frozenset({
    "the", "and", "for", "you", "are", "was", "what", "with", "this", "that", "have",
    "from", "can", "how", "please", "about", "your", "my", "me", "is", "it", "to", "of",
    "in", "on", "do", "does", "did", "a", "an", "or", "be", "we", "i",
})

# Where each claim kind lives in the Markdown source of truth.
_SECTION_BY_KIND: dict[ClaimKind, str] = {
    "preference": "## 已记住的偏好",
    "fact": "## 已记住的事实",
    "commitment": "## 承诺与待办",
    "progress": "## 工作进度",
}
_DAILY_PROGRESS_SECTION = "## 进度"
_DAILY_EXPIRED_SECTION = "## 已过期归档"


def claim_marker(claim_id: str) -> str:
    return f" <!-- claim:{claim_id} -->"


def strip_claim_markers(text: str) -> str:
    """Remove claim bookkeeping markers before Markdown reaches a model prompt."""
    return _MARKER_RE.sub("", text)


def is_claim_id(value: str) -> bool:
    return bool(_CLAIM_ID_RE.match(value))


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_expiry(value: str | None) -> str | None:
    """Normalize an expiry to UTC ISO; a bare date expires at the end of that local day."""
    if value is None or not value.strip():
        return None
    raw = value.strip()
    try:
        if len(raw) == 10:
            day = date.fromisoformat(raw)
            moment = datetime.combine(day, time(23, 59, 59)).astimezone()
        else:
            moment = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if moment.tzinfo is None:
                moment = moment.astimezone()
    except ValueError:
        raise ValueError(
            f"invalid expires_at {value!r}; use YYYY-MM-DD or an ISO 8601 date-time"
        ) from None
    return _iso(moment)


def local_day(iso_value: str) -> str:
    moment = datetime.fromisoformat(iso_value.replace("Z", "+00:00"))
    return moment.astimezone().strftime("%Y-%m-%d")


def _clean_line(text: str, limit: int) -> str:
    """Collapse to one bounded line that cannot forge or break claim markers."""
    compact = " ".join(text.replace("<!--", "").replace("-->", "").split())
    if len(compact) > limit:
        compact = compact[: limit - 1].rstrip() + "…"
    return compact


@dataclass(frozen=True, slots=True)
class Claim:
    claim_id: str
    text: str
    kind: ClaimKind
    status: ClaimStatus
    confidence: float
    quote: str | None
    session_key: str | None
    message_id: str | None
    source_file: str | None
    source_line: int | None
    supersedes_claim_id: str | None
    created_at: str
    updated_at: str
    expires_at: str | None

    def to_public(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "kind": self.kind,
            "status": self.status,
            "confidence": self.confidence,
            "quote": self.quote,
            "source": (
                f"{self.source_file}:{self.source_line}"
                if self.source_file and self.source_line
                else self.source_file
            ),
            "supersedes_claim_id": self.supersedes_claim_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class SearchHit:
    path: str
    line: int
    text: str
    claim_id: str | None
    kind: str | None
    score: float

    def to_public(self) -> dict[str, object]:
        return {
            "source": f"{self.path}:{self.line}",
            "claim_id": self.claim_id,
            "kind": self.kind,
            "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class RememberResult:
    claim: Claim
    created: bool
    superseded: list[Claim] = field(default_factory=lambda: [])


@dataclass(frozen=True, slots=True)
class LineMatch:
    path: str
    line: int
    text: str


@dataclass(frozen=True, slots=True)
class ForgetResult:
    retracted: list[Claim] = field(default_factory=lambda: [])
    removed_lines: list[LineMatch] = field(default_factory=lambda: [])
    # Filled when a keyword matched several entries and nothing was changed.
    candidates: list[LineMatch] = field(default_factory=lambda: [])
    redact_terms: list[str] = field(default_factory=lambda: [])

    @property
    def changed(self) -> bool:
        return bool(self.retracted or self.removed_lines)


def query_terms(query: str) -> list[str]:
    """Split text into search terms without a word segmenter.

    Latin words are kept whole; CJK runs contribute overlapping bigrams and
    trigrams so substring matches work for Chinese and Japanese text.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        if term not in seen:
            seen.add(term)
            terms.append(term)

    for word in _WORD_RE.findall(query.casefold()):
        position = 0
        for match in _CJK_RE.finditer(word):
            latin = word[position:match.start()]
            if len(latin) >= 2 and latin not in _ASCII_STOP_WORDS:
                add(latin)
            run = match.group()
            if len(run) == 1:
                pass
            elif len(run) == 2:
                if run not in _CJK_STOP_BIGRAMS:
                    add(run)
            else:
                for size in (3, 2):
                    for start in range(len(run) - size + 1):
                        gram = run[start:start + size]
                        if size == 2 and gram in _CJK_STOP_BIGRAMS:
                            continue
                        add(gram)
            position = match.end()
        latin = word[position:]
        if len(latin) >= 2 and latin not in _ASCII_STOP_WORDS:
            add(latin)
    return terms[:_MAX_QUERY_TERMS]


def _score(text: str, terms: Sequence[str]) -> float:
    folded = text.casefold()
    return float(sum(len(term) for term in terms if term in folded))


class DailyNotes:
    """Append-only, human-readable daily notes under ``memory/daily/``."""

    def __init__(self, memory_dir: Path) -> None:
        self.dir = memory_dir / "daily"

    def path_for(self, day: date | None = None) -> Path:
        return self.dir / f"{(day or datetime.now().date()).isoformat()}.md"

    def files(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        return sorted(path for path in self.dir.glob("*.md") if path.is_file())

    def append(self, heading: str, lines: Iterable[str], *, day: date | None = None) -> Path | None:
        body = [line for line in lines if line.strip()]
        if not body:
            return None
        path = self.path_for(day)
        self.dir.mkdir(parents=True, exist_ok=True)
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        prefix = existing if existing else f"# {path.stem}\n"
        text = prefix.rstrip("\n") + f"\n\n{heading}\n" + "\n".join(body) + "\n"
        path.write_text(text, encoding="utf-8")
        return path

    def append_summary(self, summary: str, *, session_key: str | None = None) -> Path | None:
        """Record a compaction checkpoint as process notes for today."""
        lines: list[str] = []
        for raw in summary.splitlines():
            line = re.sub(r"^(\s*[-*]\s*)\[[a-z]+\]\s*", r"\1", raw.rstrip())
            line = strip_claim_markers(line)
            if line.strip() and line.strip() != "(nothing)":
                lines.append(line)
        source = f" · {session_key}" if session_key else ""
        return self.append(f"## {datetime.now():%H:%M} 会话摘要{source}", lines)


class ClaimStore:
    """SQLite claim ledger whose content is mirrored into Markdown files."""

    def __init__(
        self,
        workspace: Path,
        *,
        daily: DailyNotes | None = None,
        use_fts: bool | None = None,
    ) -> None:
        self.workspace = workspace
        self.memory_dir = workspace / "memory"
        self.db_path = self.memory_dir / "claims.sqlite3"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.user_file = workspace / "USER.md"
        self.daily = daily
        self._want_fts = use_fts
        self._fts: bool | None = None
        self._lock = threading.RLock()

    # -- storage -----------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            self._ensure_schema(conn)
            with conn:
                yield conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS claims (
                claim_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                confidence REAL NOT NULL,
                quote TEXT,
                session_key TEXT,
                message_id TEXT,
                source_file TEXT,
                source_line INTEGER,
                supersedes_claim_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                expires_at TEXT
            );
            CREATE INDEX IF NOT EXISTS claims_status ON claims(status, kind);
            CREATE TABLE IF NOT EXISTS index_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        if self._fts is None:
            self._fts = self._create_index_table(conn)

    def _create_index_table(self, conn: sqlite3.Connection) -> bool:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        if self._want_fts is not False:
            if "memory_fts" in tables:
                return True
            try:
                conn.execute(
                    "CREATE VIRTUAL TABLE memory_fts USING fts5("
                    "text, path UNINDEXED, line UNINDEXED, claim_id UNINDEXED, "
                    "tokenize = 'trigram')"
                )
                conn.execute("DELETE FROM index_meta WHERE key = 'signature'")
                conn.commit()
                return True
            except sqlite3.OperationalError:
                logger.info("SQLite FTS5 trigram search is unavailable; memory search uses LIKE")
        if "memory_index" not in tables:
            conn.execute(
                "CREATE TABLE memory_index (text TEXT NOT NULL, path TEXT NOT NULL, "
                "line INTEGER NOT NULL, claim_id TEXT)"
            )
            conn.execute("DELETE FROM index_meta WHERE key = 'signature'")
            conn.commit()
        return False

    @property
    def fts_enabled(self) -> bool:
        with self._connect():
            pass
        return bool(self._fts)

    @property
    def _index_table(self) -> str:
        return "memory_fts" if self._fts else "memory_index"

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        return Claim(
            claim_id=row["claim_id"],
            text=row["text"],
            kind=cast(ClaimKind, row["kind"]),
            status=cast(ClaimStatus, row["status"]),
            confidence=float(row["confidence"]),
            quote=row["quote"],
            session_key=row["session_key"],
            message_id=row["message_id"],
            source_file=row["source_file"],
            source_line=row["source_line"],
            supersedes_claim_id=row["supersedes_claim_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            expires_at=row["expires_at"],
        )

    def get(self, claim_id: str) -> Claim | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,)).fetchone()
        return self._row_to_claim(row) if row else None

    def list_claims(
        self,
        *,
        status: ClaimStatus | None = None,
        kind: ClaimKind | None = None,
        limit: int = 100,
    ) -> list[Claim]:
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM claims {where} ORDER BY updated_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    def active_commitments(self, limit: int) -> list[Claim]:
        """Unexpired active commitments, soonest deadline first."""
        if limit <= 0:
            return []
        now = _iso(_utc_now())
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM claims WHERE status = 'active' AND kind = 'commitment' "
                "AND (expires_at IS NULL OR expires_at > ?) "
                "ORDER BY expires_at IS NULL, expires_at, created_at LIMIT ?",
                (now, limit),
            ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    # -- markdown mirror ---------------------------------------------------

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.workspace).as_posix()

    def _target(self, kind: ClaimKind) -> tuple[Path, str]:
        if kind == "preference":
            return self.user_file, _SECTION_BY_KIND[kind]
        if kind == "progress" and self.daily is not None:
            return self.daily.path_for(), _DAILY_PROGRESS_SECTION
        return self.memory_file, _SECTION_BY_KIND[kind]

    @staticmethod
    def _format_line(text: str, expires_at: str | None, claim_id: str) -> str:
        due = f"（有效期至 {local_day(expires_at)}）" if expires_at else ""
        return f"- {text}{due}{claim_marker(claim_id)}"

    def _insert_line(self, path: Path, heading: str, line: str) -> int:
        """Insert *line* at the end of *heading*'s section and return its 1-based line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_text(encoding="utf-8") if path.exists() else ""
        if not content and path.parent.name == "daily":
            content = f"# {path.stem}\n"
        lines = content.splitlines()
        try:
            start = next(i for i, value in enumerate(lines) if value.strip() == heading)
        except StopIteration:
            while lines and not lines[-1].strip():
                lines.pop()
            if lines:
                lines.append("")
            lines.extend([heading, line])
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return len(lines)
        end = start + 1
        while end < len(lines) and not lines[end].lstrip().startswith("#"):
            end += 1
        insert_at = end
        while insert_at > start + 1 and not lines[insert_at - 1].strip():
            insert_at -= 1
        lines.insert(insert_at, line)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return insert_at + 1

    def _markdown_files(self) -> list[Path]:
        files = [self.user_file, self.memory_file]
        if self.daily is not None:
            files.extend(self.daily.files())
        else:
            # Notes written while daily notes were enabled stay searchable.
            files.extend(DailyNotes(self.memory_dir).files())
        return [path for path in files if path.is_file()]

    def _remove_marked_line(self, claim_id: str) -> LineMatch | None:
        for path in self._markdown_files():
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, value in enumerate(lines):
                if claim_marker(claim_id).strip() in value:
                    del lines[index]
                    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                    return LineMatch(
                        path=self._relative(path),
                        line=index + 1,
                        text=strip_claim_markers(value).strip(),
                    )
        return None

    def _remove_lines(self, matches: Sequence[LineMatch]) -> list[LineMatch]:
        removed: list[LineMatch] = []
        by_path: dict[str, list[LineMatch]] = {}
        for match in matches:
            by_path.setdefault(match.path, []).append(match)
        for relative, items in by_path.items():
            path = self.workspace / relative
            lines = path.read_text(encoding="utf-8").splitlines()
            for item in sorted(items, key=lambda value: value.line, reverse=True):
                index = item.line - 1
                if 0 <= index < len(lines) and strip_claim_markers(lines[index]).strip() == item.text:
                    del lines[index]
                    removed.append(item)
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return removed

    def _find_unmarked_lines(self, needle: str) -> list[LineMatch]:
        folded = needle.casefold()
        matches: list[LineMatch] = []
        for path in self._markdown_files():
            for index, value in enumerate(path.read_text(encoding="utf-8").splitlines()):
                stripped = value.strip()
                if not stripped or stripped.startswith("#") or _MARKER_RE.search(value):
                    continue
                if folded in stripped.casefold():
                    matches.append(LineMatch(self._relative(path), index + 1, stripped))
        return matches

    # -- claim lifecycle ---------------------------------------------------

    def remember(
        self,
        text: str,
        kind: ClaimKind,
        *,
        quote: str | None = None,
        confidence: float = 1.0,
        expires_at: str | None = None,
        supersedes: Sequence[str] = (),
        session_key: str | None = None,
        message_id: str | None = None,
    ) -> RememberResult:
        if kind not in CLAIM_KINDS:
            raise ValueError(f"kind must be one of {', '.join(CLAIM_KINDS)}")
        clean = _clean_line(text, _MAX_CLAIM_CHARS)
        if not clean:
            raise ValueError("claim text must not be empty")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        expiry = parse_expiry(expires_at)
        clean_quote = _clean_line(quote, _MAX_QUOTE_CHARS) if quote else None
        with self._lock:
            self.expire_due()
            now = _iso(_utc_now())
            with self._connect() as conn:
                for old_id in supersedes:
                    row = conn.execute(
                        "SELECT status FROM claims WHERE claim_id = ?", (old_id,),
                    ).fetchone()
                    if row is None:
                        raise ValueError(f"unknown claim_id {old_id}")
                    if row["status"] != "active":
                        raise ValueError(f"claim {old_id} is {row['status']}, not active")
                duplicate = conn.execute(
                    "SELECT * FROM claims WHERE status = 'active' AND kind = ? AND text = ?",
                    (kind, clean),
                ).fetchone()
            if duplicate is not None and not supersedes:
                return RememberResult(claim=self._row_to_claim(duplicate), created=False)

            superseded: list[Claim] = []
            for old_id in supersedes:
                self._remove_marked_line(old_id)
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE claims SET status = 'superseded', updated_at = ? WHERE claim_id = ?",
                        (now, old_id),
                    )
                old = self.get(old_id)
                if old is not None:
                    superseded.append(old)

            claim_id = f"c_{uuid4().hex[:12]}"
            path, heading = self._target(kind)
            line_no = self._insert_line(path, heading, self._format_line(clean, expiry, claim_id))
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO claims VALUES (?, ?, ?, 'active', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        claim_id, clean, kind, confidence, clean_quote, session_key, message_id,
                        self._relative(path), line_no,
                        supersedes[0] if supersedes else None,
                        now, now, expiry,
                    ),
                )
            self.rebuild_index()
            claim = self.get(claim_id)
            assert claim is not None
            return RememberResult(claim=claim, created=True, superseded=superseded)

    def forget(
        self,
        *,
        claim_ids: Sequence[str] = (),
        query: str | None = None,
        all_matches: bool = False,
    ) -> ForgetResult:
        """Retract claims by id, or every entry matching *query* when unambiguous."""
        with self._lock:
            self.expire_due()
            now = _iso(_utc_now())
            targets: list[Claim] = []
            for claim_id in claim_ids:
                claim = self.get(claim_id)
                if claim is None:
                    raise ValueError(f"unknown claim_id {claim_id}")
                if claim.status in {"active", "superseded", "expired"}:
                    targets.append(claim)
            lines: list[LineMatch] = []
            needle = (query or "").strip()
            if needle:
                folded = needle.casefold()
                for claim in self.list_claims(status="active", limit=10_000):
                    haystack = f"{claim.text}\n{claim.quote or ''}".casefold()
                    if folded in haystack and claim.claim_id not in {c.claim_id for c in targets}:
                        targets.append(claim)
                lines = self._find_unmarked_lines(needle)
                if len(targets) + len(lines) > 1 and not all_matches and not claim_ids:
                    candidates = [
                        LineMatch(c.source_file or "", c.source_line or 0, f"[{c.claim_id}] {c.text}")
                        for c in targets
                    ] + lines
                    return ForgetResult(candidates=candidates)

            retracted: list[Claim] = []
            for claim in targets:
                if claim.status == "active":
                    self._remove_marked_line(claim.claim_id)
                with self._connect() as conn:
                    conn.execute(
                        "UPDATE claims SET status = 'retracted', updated_at = ? WHERE claim_id = ?",
                        (now, claim.claim_id),
                    )
                updated = self.get(claim.claim_id)
                if updated is not None:
                    retracted.append(updated)
            removed = self._remove_lines(lines)
            terms = [needle] if needle else []
            for claim in retracted:
                terms.append(claim.text)
                if claim.quote:
                    terms.append(claim.quote)
            if retracted or removed:
                self.rebuild_index()
            return ForgetResult(retracted=retracted, removed_lines=removed, redact_terms=terms)

    def expire_due(self, *, now: datetime | None = None) -> list[Claim]:
        """Retire active claims past their expiry and archive them in today's daily note."""
        moment = _iso(now or _utc_now())
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM claims WHERE status = 'active' AND expires_at IS NOT NULL "
                    "AND expires_at <= ?",
                    (moment,),
                ).fetchall()
            expired = [self._row_to_claim(row) for row in rows]
            if not expired:
                return []
            for claim in expired:
                self._remove_marked_line(claim.claim_id)
            with self._connect() as conn:
                conn.executemany(
                    "UPDATE claims SET status = 'expired', updated_at = ? WHERE claim_id = ?",
                    [(moment, claim.claim_id) for claim in expired],
                )
            if self.daily is not None:
                self.daily.append(
                    _DAILY_EXPIRED_SECTION,
                    [
                        f"- {claim.text}（{claim.kind}，原有效期至 {local_day(claim.expires_at or moment)}）"
                        for claim in expired
                    ],
                )
            self.rebuild_index()
            logger.info("Expired {} memory claim(s)", len(expired))
            return expired

    # -- index -------------------------------------------------------------

    def _signature(self) -> str:
        parts: dict[str, list[int]] = {}
        for path in self._markdown_files():
            stat = path.stat()
            parts[self._relative(path)] = [stat.st_mtime_ns, stat.st_size]
        return json.dumps(parts, sort_keys=True)

    @staticmethod
    def _kind_for_location(relative: str, heading: str) -> ClaimKind:
        if relative == "USER.md":
            return "preference"
        if relative.startswith("memory/daily/"):
            return "progress"
        for kind, section in _SECTION_BY_KIND.items():
            if heading == section:
                return kind
        return "fact"

    def rebuild_index(self) -> int:
        """Rebuild the search index from the Markdown files and resync the claim ledger.

        The files win every disagreement: a marked entry deleted by hand retracts
        its claim, a restored entry reactivates it, and a marker unknown to the
        ledger (for example after deleting ``claims.sqlite3``) recreates its claim.
        """
        with self._lock:
            rows: list[tuple[str, str, int, str | None]] = []
            found: dict[str, tuple[str, int, str, ClaimKind]] = {}
            for path in self._markdown_files():
                relative = self._relative(path)
                heading = ""
                for index, value in enumerate(path.read_text(encoding="utf-8").splitlines()):
                    marker = _MARKER_RE.search(value)
                    text = strip_claim_markers(value).strip()
                    if text.startswith("#"):
                        heading = text
                        continue
                    if not text:
                        continue
                    claim_id = marker.group(1) if marker else None
                    text = _BULLET_RE.sub("", text)
                    if claim_id and claim_id not in found:
                        found[claim_id] = (
                            relative, index + 1, text, self._kind_for_location(relative, heading),
                        )
                    rows.append((text, relative, index + 1, claim_id))
            now = _iso(_utc_now())
            with self._connect() as conn:
                table = self._index_table
                conn.execute(f"DELETE FROM {table}")
                conn.executemany(
                    f"INSERT INTO {table} (text, path, line, claim_id) VALUES (?, ?, ?, ?)", rows,
                )
                known = {
                    row["claim_id"]: row
                    for row in conn.execute(
                        "SELECT claim_id, status, source_file, source_line FROM claims",
                    )
                }
                for claim_id, row in known.items():
                    location = found.get(claim_id)
                    if location is None:
                        if row["status"] == "active":
                            conn.execute(
                                "UPDATE claims SET status = 'retracted', updated_at = ? "
                                "WHERE claim_id = ?",
                                (now, claim_id),
                            )
                    elif row["status"] != "active" or (
                        row["source_file"], row["source_line"]
                    ) != location[:2]:
                        conn.execute(
                            "UPDATE claims SET status = 'active', source_file = ?, source_line = ?, "
                            "updated_at = ? WHERE claim_id = ?",
                            (location[0], location[1], now, claim_id),
                        )
                for claim_id, (relative, line, text, kind) in found.items():
                    if claim_id in known:
                        continue
                    conn.execute(
                        "INSERT INTO claims VALUES (?, ?, ?, 'active', 1.0, NULL, NULL, NULL, "
                        "?, ?, NULL, ?, ?, NULL)",
                        (claim_id, text, kind, relative, line, now, now),
                    )
                conn.execute(
                    "INSERT OR REPLACE INTO index_meta (key, value) VALUES ('signature', ?)",
                    (self._signature(),),
                )
            return len(rows)

    def ensure_index(self) -> None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT value FROM index_meta WHERE key = 'signature'",
                ).fetchone()
            if row is None or row["value"] != self._signature():
                self.rebuild_index()

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.0,
        exclude: Iterable[tuple[str, int]] = (),
    ) -> list[SearchHit]:
        terms = query_terms(query)
        if not terms or limit <= 0:
            return []
        self.ensure_index()
        excluded = set(exclude)
        candidates: dict[tuple[str, int], sqlite3.Row] = {}
        with self._connect() as conn:
            table = self._index_table
            long_terms = [term for term in terms if len(term) >= 3] if self._fts else []
            like_terms = [term for term in terms if term not in long_terms]
            if long_terms:
                expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in long_terms)
                for row in conn.execute(
                    f"SELECT text, path, line, claim_id FROM {table} WHERE {table} MATCH ? "
                    "ORDER BY rank LIMIT ?",
                    (expression, _CANDIDATE_LIMIT),
                ):
                    candidates[(row["path"], int(row["line"]))] = row
            if like_terms:
                where = " OR ".join("text LIKE ? ESCAPE '\\'" for _ in like_terms)
                params = [
                    "%" + term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
                    for term in like_terms
                ]
                for row in conn.execute(
                    f"SELECT text, path, line, claim_id FROM {table} WHERE {where} LIMIT ?",
                    (*params, _CANDIDATE_LIMIT),
                ):
                    candidates.setdefault((row["path"], int(row["line"])), row)
            claim_kinds = {
                row["claim_id"]: (row["kind"], row["expires_at"])
                for row in conn.execute(
                    "SELECT claim_id, kind, expires_at FROM claims WHERE status = 'active'",
                )
            }
        now = _iso(_utc_now())
        hits: list[SearchHit] = []
        for key, row in candidates.items():
            if key in excluded:
                continue
            claim_id = row["claim_id"]
            kind: str | None = None
            if claim_id:
                meta = claim_kinds.get(claim_id)
                if meta is None or (meta[1] is not None and meta[1] <= now):
                    continue
                kind = meta[0]
            score = _score(row["text"], terms)
            if score <= min_score:
                continue
            text = row["text"]
            if len(text) > _SNIPPET_CHARS:
                text = text[: _SNIPPET_CHARS - 1] + "…"
            hits.append(SearchHit(row["path"], int(row["line"]), text, claim_id, kind, score))
        hits.sort(key=lambda hit: (-hit.score, hit.claim_id is None, hit.path.startswith("memory/daily"),
                                   _reverse_path_key(hit.path), hit.line))
        return hits[:limit]


def _reverse_path_key(path: str) -> str:
    # Newer daily notes sort first among equally scored hits.
    return "".join(chr(0x10FFFF - ord(ch)) for ch in path)


def redact_terms(text: str, terms: Iterable[str]) -> str:
    """Replace forgotten phrases in free text with a neutral placeholder."""
    result = text
    for term in terms:
        stripped = term.strip()
        if len(stripped) < 2:
            continue
        result = re.sub(re.escape(stripped), _FORGOTTEN, result, flags=re.IGNORECASE)
    return result
