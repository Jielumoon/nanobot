"""Rebuildable, file-incremental SQLite projection of Markdown memory."""

from __future__ import annotations

import heapq
import json
import re
import sqlite3
from array import array
from bisect import insort
from collections.abc import Generator, Iterable, Sequence
from contextlib import closing, contextmanager, nullcontext
from dataclasses import dataclass, replace

from loguru import logger

from nanobot.agent.memory_documents import (
    HEADING_RE,
    HISTORY_REF_RE,
    Claim,
    Clock,
    Evidence,
    MemoryDocuments,
    decode_claim,
    decode_entity,
    digest,
    iso,
    legacy_reference,
    line_text,
    marked_claim_ids,
    utc_now,
)
from nanobot.utils.sqlite_cleanup import vacuum_database

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]+")
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MAX_QUERY_TERMS = 32
_SNIPPET_CHARS = 300
_FORGOTTEN = "[已遗忘]"
_VECTOR_MIN_SIMILARITY = 0.3
QueryEmbedding = tuple[str, Sequence[float]]
# Frequent function words carry no topic signal for recall.
_CJK_STOP_WORDS = frozenset({
    "什么", "怎么", "这个", "那个", "我们", "你们", "他们", "一下", "可以", "不是",
    "就是", "还是", "没有", "现在", "如果", "因为", "所以", "然后", "一个", "请问",
    "帮我", "告诉", "知道", "我的", "你的", "是不", "的是", "了吗", "吗？", "一些",
    "还有", "以及", "是否", "之前", "之后", "最近", "继续", "记得", "记住", "有关",
    "为什么", "是什么", "怎么样", "能不能", "有没有", "是不是", "可不可以", "我想问",
})
_CJK_STOP_RE = re.compile("|".join(sorted(_CJK_STOP_WORDS, key=len, reverse=True)))
_ASCII_STOP_WORDS = frozenset({
    "the", "and", "for", "you", "are", "was", "what", "with", "this", "that", "have",
    "from", "can", "how", "please", "about", "your", "my", "me", "is", "it", "to", "of",
    "in", "on", "do", "does", "did", "a", "an", "or", "be", "we", "i",
})


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

    # Remove whole stop phrases first, so overlapping trigrams cannot revive them.
    for word in _WORD_RE.findall(_CJK_STOP_RE.sub(" ", query.casefold())):
        position = 0
        for match in _CJK_RE.finditer(word):
            latin = word[position:match.start()]
            if len(latin) >= 2 and latin not in _ASCII_STOP_WORDS:
                add(latin)
            run = match.group()
            if len(run) == 1:
                pass
            elif len(run) == 2:
                add(run)
            else:
                for size in (3, 2):
                    for start in range(len(run) - size + 1):
                        gram = run[start:start + size]
                        add(gram)
            position = match.end()
        latin = word[position:]
        if len(latin) >= 2 and latin not in _ASCII_STOP_WORDS:
            add(latin)
    return terms[:_MAX_QUERY_TERMS]


def _score(text: str, terms: Sequence[str]) -> float:
    folded = text.casefold()
    return float(sum(len(term) for term in terms if term in folded))


@dataclass(frozen=True)
class SearchHit:
    path: str
    line: int
    text: str
    claim_id: str | None
    kind: str | None
    score: float
    status: str = "active"
    person_id: str | None = None
    legacy_ref: str | None = None

    def to_public(self) -> dict[str, object]:
        result: dict[str, object] = {
            "source": f"{self.path}:{self.line}", "claim_id": self.claim_id,
            "kind": self.kind, "text": self.text, "status": self.status,
        }
        if self.legacy_ref is not None:
            result["legacy_ref"] = self.legacy_ref
        return result


class MemoryIndex:
    def __init__(
        self, documents: MemoryDocuments, *, use_fts: bool | None = None, clock: Clock = utc_now,
    ) -> None:
        self.documents = documents
        self.db_path = documents.workspace / "memory/claims.sqlite3"
        self.want_fts = use_fts
        self.fts = False
        self.clock = clock
        self._warnings: set[tuple[str, str, str]] = set()
        self.core_text: dict[str, str] = {}

    @contextmanager
    def connect(self) -> Generator[sqlite3.Connection]:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.db_path, timeout=10)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA secure_delete = ON")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version in {0, 1, 2}:
                self._initialize(conn)
            elif version != 3:
                raise ValueError(f"不支持的记忆索引版本：{version}")
            self.fts = self.want_fts is not False and conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name = 'memory_fts'",
            ).fetchone() is not None
            with conn:
                yield conn

    def purge_deleted(self) -> None:
        """Rebuild FTS segments and reclaim deleted content outside transactions."""
        with self.connect() as conn:
            if conn.execute("SELECT 1 FROM sqlite_master WHERE name='memory_fts'").fetchone():
                conn.execute("INSERT INTO memory_fts(memory_fts) VALUES ('rebuild')")
            conn.commit()
            vacuum_database(conn)

    def _initialize(self, conn: sqlite3.Connection) -> None:
        # Every version is a disposable projection. Rebuild old schemas from
        # Markdown, without inventing metadata that a legacy line never carried.
        conn.executescript("""
            BEGIN IMMEDIATE;
            DROP TABLE IF EXISTS vectors;
            DROP TABLE IF EXISTS tombstones;
            DROP TABLE IF EXISTS claim_events;
            DROP TABLE IF EXISTS claim_links;
            DROP TABLE IF EXISTS claims;
            DROP TABLE IF EXISTS memory_fts;
            DROP TABLE IF EXISTS chunks;
            DROP TABLE IF EXISTS documents;
            DROP TABLE IF EXISTS memory_index;
            DROP TABLE IF EXISTS index_meta;
            CREATE TABLE documents (
                path TEXT PRIMARY KEY, content_hash TEXT NOT NULL,
                scope TEXT NOT NULL CHECK(scope IN (
                    'core','bank','people','projects','daily','ledger'
                )),
                entity_id TEXT UNIQUE, entity_json TEXT
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, path TEXT NOT NULL REFERENCES documents ON DELETE CASCADE,
                line_start INTEGER NOT NULL, line_end INTEGER NOT NULL,
                claim_id TEXT, source_ref TEXT,
                scope TEXT NOT NULL CHECK(scope IN ('core','bank','people','projects','daily')),
                text TEXT NOT NULL, content_hash TEXT NOT NULL, metadata TEXT,
                current INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX chunks_path ON chunks(path);
            CREATE INDEX chunks_claim ON chunks(claim_id);
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY, text TEXT NOT NULL,
                kind TEXT CHECK(kind IN (
                    'preference','fact','commitment','progress',
                    'experience','circumstance','decision'
                )),
                status TEXT NOT NULL CHECK(status IN ('active','superseded','expired','retracted')),
                path TEXT REFERENCES documents ON DELETE CASCADE, line INTEGER,
                is_core INTEGER NOT NULL, person_id TEXT, project_id TEXT,
                due_at TEXT, expires_at TEXT, metadata TEXT NOT NULL,
                content_hash TEXT NOT NULL
            );
            CREATE INDEX claims_status ON claims(status, expires_at);
            CREATE INDEX claims_kind ON claims(kind, status, expires_at);
            CREATE INDEX claims_person ON claims(person_id);
            CREATE INDEX claims_project ON claims(project_id);
            CREATE TABLE vectors (
                claim_id TEXT NOT NULL, content_hash TEXT NOT NULL, fingerprint TEXT NOT NULL,
                dim INTEGER NOT NULL, vec BLOB NOT NULL,
                PRIMARY KEY(claim_id, content_hash, fingerprint)
            );
            CREATE TABLE claim_links (
                new_id TEXT NOT NULL, old_id TEXT NOT NULL,
                relation TEXT NOT NULL CHECK(relation = 'supersedes'),
                PRIMARY KEY(new_id, old_id, relation)
            );
            CREATE INDEX claim_links_old ON claim_links(old_id);
            CREATE TABLE claim_events (
                event_id TEXT PRIMARY KEY, target_type TEXT NOT NULL, target_id TEXT NOT NULL,
                action TEXT NOT NULL, actor TEXT NOT NULL, at TEXT NOT NULL,
                run_id TEXT, source_ref TEXT
            );
            CREATE INDEX claim_events_run ON claim_events(run_id);
            CREATE TABLE tombstones (
                target_type TEXT NOT NULL, target_id TEXT NOT NULL,
                forgotten_at TEXT NOT NULL, event_id TEXT NOT NULL REFERENCES claim_events,
                PRIMARY KEY(target_type, target_id)
            );
        """)
        if self.want_fts is not False:
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE memory_fts USING fts5(
                        text, content='chunks', content_rowid='id', tokenize='trigram'
                    )
                """)
            except sqlite3.OperationalError as exc:
                if str(exc) not in {"no such module: fts5", "no such tokenizer: trigram"}:
                    raise
            else:
                conn.execute("""
                    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
                        INSERT INTO memory_fts(rowid, text) VALUES (new.id, new.text);
                    END;
                """)
                conn.execute("""
                    CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
                        INSERT INTO memory_fts(memory_fts, rowid, text)
                        VALUES ('delete', old.id, old.text);
                    END;
                """)
        conn.execute("PRAGMA user_version = 3")
        conn.commit()

    @staticmethod
    def row_claim(row: sqlite3.Row) -> Claim:
        # This JSON was produced by the validated Markdown codec, not a second
        # input boundary. Restore nested value types without revalidating dates/IDs.
        metadata = json.loads(row["metadata"])
        metadata["evidence"]["history_lines"] = tuple(metadata["evidence"].get("history_lines", ()))
        metadata["evidence"] = Evidence.model_construct(**metadata["evidence"])
        metadata["supersedes"] = tuple(metadata["supersedes"])
        metadata["status"] = row["status"]
        return Claim.model_construct(
            **metadata, claim_id=row["claim_id"], text=row["text"],
            source_file=row["path"], source_line=row["line"], is_core=bool(row["is_core"]),
        )

    def _warn(self, path: str, content_hash: str, issue: str) -> None:
        key = (path, content_hash, issue)
        if key not in self._warnings:
            logger.warning("{}：{}，已按普通文本处理", path, issue)
            self._warnings.add(key)

    def sync(self, paths: Iterable[str] | None = None, *, full: bool = False) -> int:
        """Update one transaction from changed files; never write Markdown."""
        with self.connect() as conn:
            known = {
                row["path"]: row["content_hash"]
                for row in conn.execute("SELECT path, content_hash FROM documents")
            }
            if paths is None:
                selected = {
                    path.relative_to(self.documents.workspace).as_posix()
                    for path in self.documents.files()
                } | set(known)
            else:
                selected = set(paths)
            selected.add("memory/ledger.md")
            changed: dict[str, str | None] = {}
            for path in sorted(selected):
                file = self.documents.workspace / path
                content = file.read_text(encoding="utf-8") if file.exists() else None
                if path in {"USER.md", "memory/MEMORY.md"}:
                    self.core_text[path] = content or ""
                content_hash = digest(content) if content is not None else None
                if full or content_hash != known.get(path):
                    changed[path] = content
            if not changed:
                return 0
            affected: set[str] = set()
            for path in changed:
                affected.update(row[0] for row in conn.execute(
                    "SELECT claim_id FROM chunks WHERE path=? AND claim_id IS NOT NULL", (path,),
                ))
            if "memory/ledger.md" in changed:
                conn.execute("DELETE FROM tombstones")
                conn.execute("DELETE FROM claim_events")
                for event in self.documents.events():
                    conn.execute("INSERT INTO claim_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (
                        event.event_id, event.target_type, event.target_id, event.action,
                        event.actor, event.at, event.run_id, event.source_ref,
                    ))
                    if event.action == "forget":
                        conn.execute("INSERT OR IGNORE INTO tombstones VALUES (?, ?, ?, ?)", (
                            event.target_type, event.target_id, event.at, event.event_id,
                        ))
            tombstones = {
                (row[0], row[1])
                for row in conn.execute("SELECT target_type, target_id FROM tombstones")
            }
            # Delete old locations together before inserting a renamed entity page.
            conn.executemany("DELETE FROM documents WHERE path=?", [(path,) for path in changed])
            for path, content in changed.items():
                if content is None:
                    continue
                if path in {"USER.md", "memory/MEMORY.md"}:
                    scope = "core"
                elif path == "memory/ledger.md":
                    scope = "ledger"
                else:
                    scope = path.split("/")[1]
                issues: set[str] = set()
                entity = (
                    decode_entity(content, issues=issues) if scope in {"people", "projects"}
                    else None
                )
                content_hash = digest(content)
                entity_id = entity.entity_id if entity else None
                if entity_id is not None and conn.execute(
                    "SELECT 1 FROM documents WHERE entity_id=?", (entity_id,),
                ).fetchone() is not None:
                    issues.add("重复实体 ID")
                    entity_id = None
                for issue in sorted(issues):
                    self._warn(path, content_hash, issue)
                conn.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", (
                    path, content_hash, scope,
                    entity_id, entity.model_dump_json() if entity_id and entity else None,
                ))
                if scope == "ledger" or (
                    entity is not None and ("entity", entity.entity_id) in tombstones
                ):
                    continue
                heading = ""
                issues = set()
                for number, raw in enumerate(content.splitlines(), 1):
                    text = line_text(raw)
                    if HEADING_RE.match(raw):
                        heading = text
                        continue
                    if not text:
                        continue
                    claim = decode_claim(raw, path, number, heading, issues=issues)
                    if claim is not None:
                        text = claim.text
                    claim_id = claim.claim_id if claim else None
                    if claim_id is not None:
                        affected.add(claim_id)
                    forgotten_id = any(
                        ("claim", marker_id) in tombstones for marker_id in marked_claim_ids(raw)
                    )
                    forgotten_text = claim is None and ("chunk", digest(text)) in tombstones
                    forgotten_entity = claim is not None and any(
                        ("entity", entity_id) in tombstones
                        for entity_id in (claim.person_id, claim.project_id)
                    )
                    history_sources = HISTORY_REF_RE.findall(raw)
                    forgotten_source = any(
                        ("history", source) in tombstones for source in history_sources
                    )
                    if forgotten_id or forgotten_text or forgotten_entity or forgotten_source:
                        continue
                    summary = claim is None and scope == "daily" and (
                        bool(history_sources) or "会话摘要" in heading
                    )
                    source_ref = (
                        f"history:{history_sources[0]}" if history_sources else f"{path}:{number}"
                    )
                    conn.execute("""
                        INSERT INTO chunks (
                            path, line_start, line_end, claim_id, source_ref,
                            scope, text, content_hash, metadata, current
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        path, number, number, claim_id, source_ref, scope, text,
                        digest(text), claim.metadata_json() if claim else None, not summary,
                    ))
                for issue in sorted(issues):
                    self._warn(path, content_hash, issue)
            entity_chunks = """
                SELECT ch.id FROM chunks ch JOIN documents d ON d.path=ch.path
                WHERE EXISTS (
                    SELECT 1 FROM tombstones t WHERE t.target_type='entity' AND (
                        t.target_id=d.entity_id
                        OR t.target_id=json_extract(ch.metadata, '$.person_id')
                        OR t.target_id=json_extract(ch.metadata, '$.project_id')
                    )
                )
            """
            affected.update(row[0] for row in conn.execute(f"""
                SELECT claim_id FROM chunks WHERE id IN ({entity_chunks})
                    AND claim_id IS NOT NULL
            """))
            conn.execute(f"DELETE FROM chunks WHERE id IN ({entity_chunks})")
            conn.execute("""
                DELETE FROM chunks
                WHERE claim_id IN (SELECT target_id FROM tombstones WHERE target_type='claim')
                   OR (claim_id IS NULL AND content_hash IN (
                       SELECT target_id FROM tombstones WHERE target_type='chunk'
                   ))
                   OR source_ref IN (
                       SELECT 'history:' || target_id FROM tombstones WHERE target_type='history'
                   )
            """)
            if "memory/ledger.md" in changed:
                affected.update(target for kind, target in tombstones if kind == "claim")
            conn.execute("""
                UPDATE documents SET entity_json=NULL WHERE entity_id IN (
                    SELECT target_id FROM tombstones WHERE target_type='entity'
                )
            """)
            self._project_claims(conn, affected)
            self._prune_vectors(conn)
            return len(changed)

    def _prune_vectors(self, conn: sqlite3.Connection) -> None:
        # No FK cascade: rebuilding one file temporarily replaces its claim rows,
        # while unchanged claim text must retain its cached vector.
        conn.execute("""
            DELETE FROM vectors WHERE NOT EXISTS (
                SELECT 1 FROM claims c WHERE c.claim_id=vectors.claim_id
                  AND c.content_hash=vectors.content_hash AND c.status='active'
                  AND (c.expires_at IS NULL OR c.expires_at>?)
            )
        """, (iso(self.clock()),))

    def pending_vectors(self, fingerprint: str, limit: int) -> list[tuple[str, str, str]]:
        with self.connect() as conn:
            return [tuple(row) for row in conn.execute("""
                SELECT c.claim_id, c.content_hash, c.text FROM claims c
                WHERE c.status='active' AND (c.expires_at IS NULL OR c.expires_at>?)
                  AND NOT EXISTS (
                    SELECT 1 FROM vectors v WHERE v.claim_id=c.claim_id
                      AND v.content_hash=c.content_hash AND v.fingerprint=?
                  )
                ORDER BY json_extract(c.metadata, '$.updated_at') DESC, c.claim_id LIMIT ?
            """, (iso(self.clock()), fingerprint, limit))]

    def save_vectors(
        self, fingerprint: str, pending: Sequence[tuple[str, str, str]],
        vectors: Sequence[array[float]],
    ) -> int:
        saved = 0
        with self.connect() as conn:
            for (claim_id, content_hash, _), vector in zip(pending, vectors):
                # The API wait can overlap a forget, correction, or manual edit.
                saved += conn.execute("""
                    INSERT OR REPLACE INTO vectors
                    SELECT claim_id, content_hash, ?, ?, ? FROM claims
                    WHERE claim_id=? AND content_hash=? AND status='active'
                      AND (expires_at IS NULL OR expires_at>?)
                """, (
                    fingerprint, len(vector), vector.tobytes(), claim_id, content_hash,
                    iso(self.clock()),
                )).rowcount
        return saved

    def _project_claims(self, conn: sqlite3.Connection, affected: set[str]) -> None:
        occurrences: dict[str, list[Claim]] = {}
        related: set[str] = set()
        for claim_id in affected:
            related.update(row[0] for row in conn.execute(
                "SELECT old_id FROM claim_links WHERE new_id=?", (claim_id,),
            ))
            conn.execute("DELETE FROM claims WHERE claim_id=?", (claim_id,))
            conn.execute("DELETE FROM claim_links WHERE new_id=?", (claim_id,))
            rows = conn.execute("""
                SELECT *, line_start AS line, scope='core' AS is_core,
                       json_extract(metadata, '$.status') AS status
                FROM chunks WHERE claim_id=? ORDER BY path, line_start
            """, (claim_id,))
            for row in rows:
                occurrences.setdefault(claim_id, []).append(self.row_claim(row))
        for claim_id, copies in occurrences.items():
            claim = copies[0]
            inactive = next((copy.status for copy in copies if copy.status != "active"), None)
            if inactive is not None:
                claim = claim.model_copy(update={"status": inactive})
            for copy in copies[1:]:
                path = copy.source_file
                content_hash = conn.execute(
                    "SELECT content_hash FROM documents WHERE path=?", (path,),
                ).fetchone()[0]
                self._warn(str(path), content_hash, "重复记忆标记")
            self._insert_claim(conn, claim)
            conn.executemany(
                "INSERT INTO claim_links VALUES (?, ?, 'supersedes')",
                [(claim_id, old) for old in claim.supersedes],
            )
            related.update(claim.supersedes)
        for claim_id in affected | related:
            tombstone = conn.execute("""
                SELECT forgotten_at FROM tombstones
                WHERE target_type='claim' AND target_id=?
            """, (claim_id,)).fetchone()
            if tombstone is not None:
                self._insert_claim(conn, Claim(
                    claim_id=claim_id, text="", status="retracted", updated_at=tombstone[0],
                ))
            else:
                conn.execute("""
                    UPDATE claims SET status=CASE
                        WHEN EXISTS(SELECT 1 FROM claim_links WHERE old_id=?)
                             AND json_extract(metadata, '$.status')='active'
                        THEN 'superseded'
                        ELSE json_extract(metadata, '$.status')
                    END WHERE claim_id=?
                """, (claim_id, claim_id))
            conn.execute("""
                UPDATE chunks SET current=COALESCE((
                    SELECT status='active' FROM claims
                    WHERE claims.claim_id=chunks.claim_id
                ), 0) WHERE claim_id=?
            """, (claim_id,))

    @staticmethod
    def _insert_claim(conn: sqlite3.Connection, claim: Claim) -> None:
        conn.execute("""
            INSERT OR REPLACE INTO claims VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            claim.claim_id, claim.text, claim.kind, claim.status, claim.source_file,
            claim.source_line, claim.is_core, claim.person_id, claim.project_id,
            claim.due_at, claim.expires_at, claim.metadata_json(), digest(claim.text),
        ))

    @staticmethod
    def _search_filter(
        *, now: str, include_history: bool, scopes: Sequence[str],
        active_projects_only: bool, person_ids: Sequence[str] | None,
    ) -> tuple[str, list[object]]:
        where = "ch.scope IN (" + ",".join("?" for _ in scopes) + ")"
        params: list[object] = list(scopes)
        if not include_history:
            where += """
                AND ch.current=1
                AND COALESCE(
                    (SELECT expires_at FROM claims lifecycle WHERE lifecycle.claim_id=ch.claim_id),
                    json_extract(ch.metadata, '$.expires_at'), ?
                ) > ?
            """
            params.extend(("9999", now))
        if active_projects_only:
            where += """
                AND (ch.scope!='projects' OR EXISTS (
                    SELECT 1 FROM documents p WHERE p.path=ch.path
                      AND json_extract(p.entity_json, '$.status')='active'
                ))
                AND NOT EXISTS (
                    SELECT 1 FROM documents p
                    WHERE (p.entity_id=json_extract(ch.metadata, '$.project_id')
                           OR p.path=ch.path AND p.scope='projects')
                      AND json_extract(p.entity_json, '$.status')!='active'
                )
            """
        if person_ids is not None:
            placeholders = ",".join("?" for _ in person_ids)
            where += f"""
                AND (json_extract(ch.metadata, '$.person_id') IS NULL
                     OR json_extract(ch.metadata, '$.person_id') IN ({placeholders}))
                AND (ch.scope!='people' OR ch.path IN (
                    SELECT path FROM documents WHERE entity_id IN ({placeholders})
                ))
            """
            params.extend(person_ids)
            params.extend(person_ids)
        return where, params

    def search(
        self, query: str, *, now: str, limit: int, min_score: float,
        exclude: Iterable[tuple[str, int]], include_history: bool,
        scopes: Sequence[str] = ("core", "bank", "projects", "people", "daily"),
        active_projects_only: bool = False, person_ids: Sequence[str] | None = None,
        exclude_texts: Iterable[str] = (), deduplicate: bool = False,
        semantic: QueryEmbedding | None = None,
        connection: sqlite3.Connection | None = None,
    ) -> list[SearchHit]:
        terms = query_terms(query)
        if not terms and semantic is None:
            return []
        where, filters = self._search_filter(
            now=now, include_history=include_history, scopes=scopes,
            active_projects_only=active_projects_only, person_ids=person_ids,
        )
        excluded = set(exclude)
        excluded_texts = {text.casefold() for text in exclude_texts}
        hits: list[SearchHit] = []
        candidate_limit = limit * 4 if semantic is not None else limit
        best: list[tuple[tuple[float, bool, bool, str, int], SearchHit]] = []
        with nullcontext(connection) if connection is not None else self.connect() as conn:
            if terms:
                long_terms = [term for term in terms if len(term) >= 3] if self.fts else []
                like_terms = [term for term in terms if term not in long_terms]
                matches: list[str] = []
                params: list[object] = []
                if long_terms:
                    matches.append(
                        "ch.id IN (SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?)"
                    )
                    expression = " OR ".join('"' + term.replace('"', '""') + '"' for term in long_terms)
                    params.append(expression)
                for term in like_terms:
                    matches.append("ch.text LIKE ? ESCAPE '\\'")
                    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                    params.append("%" + escaped + "%")
                matched = " OR ".join(matches)
                rows = conn.execute(f"""
                    SELECT ch.*, c.claim_id AS selected_id, c.kind,
                           COALESCE(c.expires_at, lifecycle.expires_at) AS expires_at,
                           COALESCE(c.status, lifecycle.status,
                               CASE WHEN ch.current=0 THEN 'historical' END,
                               json_extract(ch.metadata, '$.status')) AS status,
                           COALESCE(json_extract(ch.metadata, '$.person_id'),
                               CASE WHEN d.scope='people' THEN d.entity_id END) AS person_id
                    FROM chunks ch LEFT JOIN claims c
                      ON c.claim_id=ch.claim_id AND c.path=ch.path AND c.line=ch.line_start
                    LEFT JOIN claims lifecycle ON lifecycle.claim_id=ch.claim_id
                    JOIN documents d ON d.path=ch.path
                    WHERE ({matched}) AND {where}
                """, [*params, *filters])
                for row in rows:
                    if (row["path"], row["line_start"]) in excluded:
                        continue
                    if row["text"].casefold() in excluded_texts:
                        continue
                    score = _score(row["text"], terms)
                    if score <= min_score:
                        continue
                    status = row["status"] or "active"
                    if (
                        status == "active" and row["expires_at"] is not None
                        and row["expires_at"] <= now
                    ):
                        status = "expired"
                    hit = SearchHit(
                        row["path"], row["line_start"], row["text"][:_SNIPPET_CHARS],
                        row["selected_id"], row["kind"], score, status, row["person_id"],
                        legacy_ref=(
                            legacy_reference(row["path"], row["line_start"], row["text"])
                            if row["scope"] == "core" and row["claim_id"] is None
                            and "<!--" not in self.core_text[row["path"]].splitlines()[row["line_start"] - 1]
                            else None
                        ),
                    )
                    key = (
                        -hit.score, hit.claim_id is None,
                        hit.path.startswith("memory/daily"), hit.path, hit.line,
                    )
                    if deduplicate:
                        duplicate = next((item for item in best
                                          if item[1].text.casefold() == hit.text.casefold()), None)
                        if duplicate is not None:
                            if duplicate[0] <= key:
                                continue
                            best.remove(duplicate)
                    if len(best) < candidate_limit or key < best[-1][0]:
                        insort(best, (key, hit))
                        if len(best) > candidate_limit:
                            best.pop()
            hits = [hit for _, hit in best]
            if semantic is not None:
                vectors = self._vector_hits(
                    conn, semantic, now=now, where=where, params=filters,
                    excluded=excluded, excluded_texts=excluded_texts,
                    limit=candidate_limit, deduplicate=deduplicate,
                )
                hits = self._fuse(hits[:candidate_limit], vectors)
        if deduplicate:
            unique: dict[str, SearchHit] = {}
            for hit in hits:
                unique.setdefault(hit.text.casefold(), hit)
            hits = list(unique.values())
        return hits[:limit]

    @staticmethod
    def _vector_hits(
        conn: sqlite3.Connection, semantic: QueryEmbedding, *, now: str,
        where: str, params: Sequence[object], excluded: set[tuple[str, int]],
        excluded_texts: set[str], limit: int, deduplicate: bool,
    ) -> list[SearchHit]:
        fingerprint, query = semantic
        rows = conn.execute(f"""
            SELECT ch.path, ch.line_start, ch.text, c.claim_id, c.kind,
                   COALESCE(c.person_id,
                       CASE WHEN d.scope='people' THEN d.entity_id END) AS person_id,
                   v.vec
            FROM vectors v JOIN claims c
              ON c.claim_id=v.claim_id AND c.content_hash=v.content_hash
            JOIN chunks ch
              ON ch.claim_id=c.claim_id AND ch.path=c.path AND ch.line_start=c.line
            JOIN documents d ON d.path=ch.path
            WHERE v.fingerprint=? AND v.dim=? AND c.status='active'
              AND (c.expires_at IS NULL OR c.expires_at>?) AND {where}
        """, [fingerprint, len(query), now, *params])
        best: list[tuple[float, str, SearchHit]] = []
        for row in rows:
            if (row["path"], row["line_start"]) in excluded:
                continue
            text = row["text"]
            if text.casefold() in excluded_texts:
                continue
            vector = array("f")
            vector.frombytes(row["vec"])
            score = sum(left * right for left, right in zip(query, vector))
            if score < _VECTOR_MIN_SIMILARITY:
                continue
            hit = SearchHit(
                row["path"], row["line_start"], text[:_SNIPPET_CHARS], row["claim_id"],
                row["kind"], score, "active", row["person_id"],
            )
            candidate = (score, row["claim_id"], hit)
            if deduplicate:
                duplicate = next((item for item in best
                                  if item[2].text.casefold() == hit.text.casefold()), None)
                if duplicate is not None:
                    if candidate[:2] <= duplicate[:2]:
                        continue
                    best.remove(duplicate)
                    heapq.heapify(best)
            if len(best) < limit:
                heapq.heappush(best, candidate)
            elif candidate[:2] > best[0][:2]:
                heapq.heapreplace(best, candidate)
        return [item[2] for item in sorted(best, reverse=True)]

    @staticmethod
    def _fuse(lexical: list[SearchHit], semantic: list[SearchHit]) -> list[SearchHit]:
        scores: dict[str | tuple[str, int], float] = {}
        hits: dict[str | tuple[str, int], SearchHit] = {}
        for ranking in (lexical, semantic):
            for rank, hit in enumerate(ranking, 1):
                key = hit.claim_id or (hit.path, hit.line)
                scores[key] = scores.get(key, 0.0) + 1.0 / (60 + rank)
                hits.setdefault(key, hit)
        return [replace(hits[key], score=scores[key])
                for key in sorted(scores, key=lambda key: scores[key], reverse=True)]
