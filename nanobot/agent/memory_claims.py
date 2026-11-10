"""Memory lifecycle. Markdown and the permanent ledger own all durable state."""

from __future__ import annotations

import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from nanobot.agent.memory_documents import (
    SECTIONS,
    Claim,
    ClaimDraft,
    ClaimKind,
    ClaimStatus,
    Clock,
    Entity,
    EntityKind,
    EntityStatus,
    Evidence,
    LedgerEvent,
    MemoryDocuments,
    decode_claim,
    decode_entity,
    digest,
    encode_claim,
    insert_lines,
    iso,
    line_text,
    local_day,
    marked_claim_ids,
    strip_claim_markers,
    utc_now,
)
from nanobot.agent.memory_entities import MemoryEntities
from nanobot.agent.memory_index import MemoryIndex, QueryEmbedding, SearchHit


@dataclass(frozen=True)
class RememberResult:
    claim: Claim
    created: bool
    superseded: list[Claim] = field(default_factory=list)


@dataclass(frozen=True)
class LineMatch:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class ForgetResult:
    retracted: list[Claim] = field(default_factory=list)
    removed_lines: list[LineMatch] = field(default_factory=list)
    candidates: list[LineMatch] = field(default_factory=list)
    redact_terms: list[str] = field(default_factory=list)
    history_ids: list[int] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    needs_confirmation: bool = False

    @property
    def changed(self) -> bool:
        return bool(self.retracted or self.removed_lines or self.history_ids or self.entities)


def document_lines(snapshot: dict[str, str]) -> Iterable[tuple[LineMatch, Claim | None]]:
    for path, content in sorted(snapshot.items()):
        heading = ""
        issues: set[str] = set()
        for number, raw in enumerate(content.splitlines(), 1):
            text = line_text(raw)
            if text.startswith("#"):
                heading = text
                continue
            if text:
                claim = (
                    None if path.startswith("dreams/")
                    else decode_claim(raw, path, number, heading, issues=issues)
                )
                yield LineMatch(path, number, raw), claim


class ClaimStore:
    def __init__(
        self, workspace: Path, *,
        use_fts: bool | None = None, timezone: str = "UTC", clock: Clock = utc_now,
        lock: threading.RLock,
    ) -> None:
        self.workspace = workspace
        self.timezone = timezone
        self.clock = clock
        self.lock = lock
        self.documents = MemoryDocuments(workspace)
        self.index = MemoryIndex(self.documents, use_fts=use_fts, clock=clock)
        self.entities = MemoryEntities(self.index)
        self.db_path = self.index.db_path

    @property
    def fts_enabled(self) -> bool:
        with self.index.connect():
            return self.index.fts

    def sync_index(self, paths: Iterable[str] | None = None) -> int:
        with self.lock:
            return self.index.sync(paths)

    def rebuild_index(self) -> int:
        with self.lock:
            return self.index.sync(full=True)

    def get(self, claim_id: str) -> Claim | None:
        with self.lock:
            self.index.sync()
            with self.index.connect() as conn:
                row = conn.execute("SELECT * FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
                return self.index.row_claim(row) if row is not None else None

    def list_claims(
        self, *, status: ClaimStatus | None = None, kind: ClaimKind | None = None,
        person_id: str | None = None, project_id: str | None = None,
        current: bool = False, limit: int = 100,
    ) -> list[Claim]:
        with self.lock:
            self.index.sync()
            clauses: list[str] = []
            params: list[object] = []
            for column, value in (
                ("status", status), ("kind", kind),
                ("person_id", person_id), ("project_id", project_id),
            ):
                if value is not None:
                    clauses.append(f"{column}=?")
                    params.append(value)
            if current:
                clauses.append("status='active' AND (expires_at IS NULL OR expires_at>?)")
                params.append(iso(self.clock()))
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            with self.index.connect() as conn:
                return [self.index.row_claim(row) for row in conn.execute(
                    f"""
                        SELECT * FROM claims{where}
                        ORDER BY json_extract(metadata, '$.updated_at') DESC,
                                 path, line DESC, claim_id LIMIT ?
                    """, (*params, limit),
                )]

    def active_commitments(self, limit: int) -> list[Claim]:
        with self.lock:
            self.index.sync()
            with self.index.connect() as conn:
                rows = conn.execute("""
                    SELECT * FROM claims WHERE status='active' AND kind='commitment'
                      AND (expires_at IS NULL OR expires_at>?)
                    ORDER BY due_at IS NULL, due_at, claim_id LIMIT ?
                """, (iso(self.clock()), limit))
                return [self.index.row_claim(row) for row in rows]

    def _target(self, draft: ClaimDraft, pages: dict[str, tuple[Entity, str]]) -> str:
        if draft.project_id:
            return pages[draft.project_id][1]
        if draft.person_id:
            return pages[draft.person_id][1]
        if draft.core:
            return "USER.md" if draft.kind == "preference" else "memory/MEMORY.md"
        if draft.kind == "progress":
            return f"memory/daily/{local_day(iso(self.clock()), self.timezone)}.md"
        bank = {
            "preference": "preferences", "experience": "experiences",
            "circumstance": "circumstances",
        }.get(draft.kind, "facts")
        return f"memory/bank/{bank}.md"

    def remember(
        self, draft: ClaimDraft, *, session_key: str | None = None,
        message_id: str | None = None, actor: str = "user", run_id: str | None = None,
        source_ref: str | None = None,
        source_timestamp: str | None = None,
        source_kind: Literal["message", "summary"] | None = None,
    ) -> RememberResult:
        with self.lock:
            self.index.sync()
            if (draft.person or draft.person_id) and not (draft.quote or source_ref):
                raise ValueError("只记录有证据的人物信息，请提供原话 quote 或可核验来源")
            now = iso(self.clock())
            snapshot = self.documents.snapshot()
            changes: dict[str, str] = {}
            pages = self.entities.pages()
            links: dict[str, str] = {}
            requested_entities: tuple[tuple[EntityKind, str | None, tuple[str, ...]], ...] = (
                ("project", draft.project, ()), ("person", draft.person, draft.aliases),
            )
            for kind, name, aliases in requested_entities:
                if name:
                    entity, path = self.entities.ensure(
                        name, kind, aliases, at=now, snapshot=snapshot, changes=changes,
                    )
                    pages[entity.entity_id] = entity, path
                    links[f"{kind}_id"] = entity.entity_id
            draft = draft.model_copy(update=links)
            entity_events = [
                LedgerEvent(
                    target_type="entity", target_id=entity_id,
                    action="update" if pages[entity_id][1] in snapshot else "remember",
                    actor=actor, at=now, run_id=run_id,
                ) for entity_id in links.values()
            ]
            with self.index.connect() as conn:
                superseded: list[Claim] = []
                for old_id in draft.supersedes:
                    row = conn.execute(
                        "SELECT * FROM claims WHERE claim_id=?", (old_id,),
                    ).fetchone()
                    if row is None:
                        raise ValueError(f"找不到记忆 ID：{old_id}")
                    if row["status"] not in {"active", "superseded"}:
                        raise ValueError(f"记忆 {old_id} 状态为 {row['status']}，不能取代")
                    superseded.append(self.index.row_claim(row))
                duplicate = conn.execute("""
                    SELECT * FROM claims WHERE status='active' AND text=? AND kind=?
                      AND (expires_at IS NULL OR expires_at>?)
                      AND person_id IS ? AND project_id IS ?
                """, (
                    draft.text, draft.kind, iso(self.clock()), draft.person_id, draft.project_id,
                )).fetchone()
                if duplicate is not None and not draft.supersedes:
                    self.documents.apply(changes)
                    self.documents.append_events(entity_events)
                    self.index.sync(changes)
                    return RememberResult(self.index.row_claim(duplicate), False)
            path = self._target(draft, pages)
            claim = Claim.model_construct(
                claim_id=f"c_{uuid4().hex[:12]}", created_at=now, updated_at=now,
                evidence=Evidence.model_construct(
                    quote=draft.quote, source_ref=source_ref,
                    session_key=session_key, message_id=message_id,
                    source_timestamp=source_timestamp, source_kind=source_kind,
                ),
                **draft.model_dump(exclude={"quote", "core", "person", "project", "aliases"}),
            )
            heading = SECTIONS[draft.kind]
            if draft.project_id:
                heading = {
                    "decision": "## 决定", "commitment": "## 待办",
                }.get(draft.kind, "## 进度")
            changes[path] = insert_lines(
                changes.get(path, snapshot.get(path, "")), heading,
                [encode_claim(claim, timezone=self.timezone)],
            )
            content = changes.pop(path)
            changes[path] = content
            # Publish the replacement and its complete relationship list first.
            self.documents.apply(changes)
            self.documents.append_events([
                *entity_events,
                LedgerEvent(
                    target_type="claim", target_id=claim.claim_id, action="remember",
                    actor=actor, at=now, run_id=run_id, source_ref=source_ref,
                ),
                *(LedgerEvent(
                    target_type="claim", target_id=old.claim_id, action="supersede",
                    actor=actor, at=now, run_id=run_id, source_ref=source_ref,
                ) for old in superseded),
            ])
            self.index.sync(changes)
            if superseded:
                self.reconcile_files()
            with self.index.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM claims WHERE claim_id=?", (claim.claim_id,),
                ).fetchone()
                saved = self.index.row_claim(row)
            return RememberResult(
                saved, True,
                [old.model_copy(update={"status": "superseded"}) for old in superseded],
            )

    def forget(
        self, *, claim_ids: Sequence[str] = (), query: str | None = None,
        all_matches: bool = False, history_entries: Sequence[tuple[int, str]] = (),
        actor: str = "user", run_id: str | None = None,
        person: str | None = None, project: str | None = None,
    ) -> ForgetResult:
        with self.lock:
            self.index.sync()
            # Raw marker IDs keep malformed comments and every duplicate addressable.
            snapshot = self.documents.snapshot(include_dreams=True)
            lines = [
                (line, claim, marked_claim_ids(line.text))
                for line, claim in document_lines(snapshot)
            ]
            tombstones = self.documents.tombstones()
            ids = set(claim_ids)
            entity_page = None
            entity_paths: set[str] = set()
            selector = person if person is not None else project
            if selector is not None:
                entity_page = self.entities.find(selector, "person" if person else "project")
                if entity_page is None:
                    raise ValueError("找不到该人物或项目，请使用已保存的名称或实体 ID")
                entity, _ = entity_page
                for path, content in snapshot.items():
                    if path.startswith(("memory/people/", "memory/projects/")):
                        page_entity = decode_entity(content, issues=set())
                        if page_entity is not None and page_entity.entity_id == entity.entity_id:
                            entity_paths.add(path)
                for line, claim, markers in lines:
                    if line.path in entity_paths or (
                        claim is not None and entity.entity_id in (claim.person_id, claim.project_id)
                    ):
                        ids.update(markers)
            known = {claim_id for _, _, markers in lines for claim_id in markers}
            for claim_id in ids:
                if claim_id not in known and ("claim", claim_id) not in tombstones:
                    raise ValueError(f"找不到记忆 ID：{claim_id}")
            needle = query.strip().casefold() if query else ""
            targets: dict[str, Claim] = {}
            matches: list[LineMatch] = []
            terms: set[str] = {query.strip()} if query else set()
            for line, claim, markers in lines:
                if claim is not None:
                    haystack = (claim.text + "\n" + (claim.quote or "")).casefold()
                    if claim.claim_id in ids or (needle and needle in haystack):
                        targets.setdefault(claim.claim_id, claim)
                        terms.add(claim.text)
                        if claim.quote:
                            terms.add(claim.quote)
                elif markers & ids:
                    terms.add(line_text(line.text))
                elif needle and needle in line_text(line.text).casefold():
                    matches.append(line)
                if line.path in entity_paths and claim is None:
                    matches.append(line)
                    terms.add(line_text(line.text))
            history_matches = [
                LineMatch("memory/history.jsonl", cursor, line)
                for cursor, content in history_entries for line in content.splitlines()
                if needle and needle in line.casefold()
            ]
            candidates = [
                LineMatch(c.source_file or "", c.source_line or 0, f"[{c.claim_id}] {c.text}")
                for c in targets.values()
            ] + matches
            if not candidates:
                candidates = history_matches
            if entity_page is not None and not all_matches:
                return ForgetResult(
                    candidates=candidates, entities=[entity_page[0].entity_id],
                    needs_confirmation=True,
                )
            if needle and not all_matches and not ids and len(candidates) > 1:
                return ForgetResult(candidates=candidates)
            terms = {term for term in terms if len(term.strip()) >= 2}
            folded = [term.casefold() for term in terms]
            remove: dict[str, set[int]] = {}
            events: list[LedgerEvent] = []
            now = iso(self.clock())

            def tombstone(
                kind: Literal["claim", "chunk", "history", "entity"],
                target: str, source: str | None,
            ) -> None:
                events.append(LedgerEvent(
                    target_type=kind, target_id=target, action="forget", actor=actor,
                    at=now, run_id=run_id, source_ref=source,
                ))

            for claim_id in ids | targets.keys():
                tombstone("claim", claim_id, None)
            if entity_page is not None:
                tombstone("entity", entity_page[0].entity_id, None)
            removed: list[LineMatch] = []
            for line, claim, markers in lines:
                haystack = line_text(line.text) + ("\n" + (claim.quote or "") if claim else "")
                by_id = bool(markers & (ids | targets.keys()))
                by_text = any(term in haystack.casefold() for term in folded)
                if by_id or by_text:
                    remove.setdefault(line.path, set()).add(line.line)
                    text = line_text(line.text)
                    if len(text.strip()) >= 2:
                        tombstone("chunk", digest(text), f"{line.path}:{line.line}")
                    for claim_id in markers - (ids | targets.keys()):
                        tombstone("claim", claim_id, None)
                        ids.add(claim_id)
                    if claim is not None:
                        targets.setdefault(claim.claim_id, claim)
                    else:
                        removed.append(LineMatch(line.path, line.line, line.text.strip()))
            for claim in targets.values():
                terms.add(claim.text)
                if claim.quote:
                    terms.add(claim.quote)
            terms = {term for term in terms if len(term.strip()) >= 2}
            folded = [term.casefold() for term in terms]
            history_ids: list[int] = []
            for cursor, content in history_entries:
                matched = False
                for line in content.splitlines():
                    if any(term in line.casefold() for term in folded):
                        target = f"{cursor}:{digest(line)}"
                        tombstone("history", target, f"history:{target}")
                        matched = True
                if matched:
                    history_ids.append(cursor)
            # A failed cleanup still leaves a durable recall barrier.
            self.documents.append_events(events)
            changes: dict[str, str] = {}
            for path, numbers in remove.items():
                if path in entity_paths:
                    continue
                changes[path] = "".join(
                    raw + "\n" for number, raw in enumerate(snapshot[path].splitlines(), 1)
                    if number not in numbers
                )
            self.documents.apply(changes)
            for path in entity_paths:
                (self.workspace / path).unlink()
            self.index.sync()
            retracted = [
                Claim(claim_id=claim_id, text="", status="retracted", updated_at=now)
                for claim_id in sorted(ids | targets.keys())
            ]
            return ForgetResult(
                retracted, removed, redact_terms=sorted(terms), history_ids=history_ids,
                entities=[entity_page[0].entity_id] if entity_page is not None else [],
            )

    def reconcile_files(self) -> list[str]:
        """Explicitly clean restored tombstones and finish interrupted archive moves."""
        with self.lock:
            self.index.sync()
            with self.index.connect() as conn:
                claims = {
                    row["claim_id"]: self.index.row_claim(row)
                    for row in conn.execute("SELECT * FROM claims")
                }
            return self._reconcile(claims, at=iso(self.clock()))

    def _reconcile(self, claims: dict[str, Claim], *, at: str) -> list[str]:
        snapshot = self.documents.snapshot(include_dreams=True)
        tombstones = self.documents.tombstones()
        deleted_pages: list[str] = []
        for path, content in snapshot.items():
            if path.startswith(("memory/people/", "memory/projects/")):
                entity = decode_entity(content, issues=set())
                if entity is not None and ("entity", entity.entity_id) in tombstones:
                    deleted_pages.append(path)
        removed: dict[str, set[int]] = {}
        archives: dict[str, Claim] = {}
        lines = list(document_lines(snapshot))
        archived_ids = {
            claim.claim_id for line, claim in lines
            if claim is not None and claim.status in {"expired", "superseded"}
            and line.path.startswith("memory/daily/")
        }
        for line, claim in lines:
            if line.path in deleted_pages:
                continue
            forgotten_id = any(
                ("claim", claim_id) in tombstones for claim_id in marked_claim_ids(line.text)
            )
            forgotten_text = claim is None and ("chunk", digest(line_text(line.text))) in tombstones
            forgotten_entity = claim is not None and any(
                ("entity", entity_id) in tombstones
                for entity_id in (claim.person_id, claim.project_id)
            )
            if forgotten_id or forgotten_text or forgotten_entity:
                removed.setdefault(line.path, set()).add(line.line)
            elif claim is not None:
                current = claims[claim.claim_id]
                if (line.path, line.line) != (current.source_file, current.source_line):
                    continue
                if current.status in {"expired", "superseded"}:
                    if claim.status != current.status or not line.path.startswith("memory/daily/"):
                        removed.setdefault(line.path, set()).add(line.line)
                        if claim.claim_id not in archived_ids:
                            archives[claim.claim_id] = current.model_copy(update={"updated_at": at})
        changes: dict[str, str] = {}
        for path, numbers in removed.items():
            changes[path] = "".join(
                raw + "\n" for number, raw in enumerate(snapshot[path].splitlines(), 1)
                if number not in numbers
            )
        if archives:
            day = local_day(at, self.timezone)
            archive_path = f"memory/daily/{day}.md"
            text = changes.pop(archive_path, snapshot.get(archive_path, f"# {day}\n"))
            for status, heading in (("expired", "## 已过期归档"), ("superseded", "## 已取代归档")):
                additions = [
                    encode_claim(claim, timezone=self.timezone)
                    for claim in archives.values() if claim.status == status
                ]
                if additions:
                    text = insert_lines(text, heading, additions)
            # Preserve archived evidence before deleting the live source.
            changes = {archive_path: text, **changes}
        self.documents.apply(changes)
        for path in deleted_pages:
            (self.workspace / path).unlink()
        self.index.sync(path for path in [*changes, *deleted_pages] if not path.startswith("dreams/"))
        return [*changes, *deleted_pages]

    def expire_due(
        self, *, now: datetime | None = None, actor: str = "system", run_id: str | None = None,
    ) -> list[Claim]:
        moment = iso(now if now is not None else self.clock())
        with self.lock:
            self.index.sync()
            with self.index.connect() as conn:
                claims = {
                    row["claim_id"]: self.index.row_claim(row)
                    for row in conn.execute("SELECT * FROM claims")
                }
            expired = [
                claim.model_copy(update={"status": "expired", "updated_at": moment})
                for claim in claims.values() if claim.status == "active"
                and claim.expires_at is not None and claim.expires_at <= moment
            ]
            if expired:
                claims.update({claim.claim_id: claim for claim in expired})
                self._reconcile(claims, at=moment)
                self.documents.append_events(
                    LedgerEvent(
                        target_type="claim", target_id=claim.claim_id,
                        action="expire", actor=actor, at=moment, run_id=run_id,
                    ) for claim in expired
                )
                self.index.sync([])
            return expired

    def update_project(
        self, project: str, status: EntityStatus, *, rename: str | None = None,
        actor: str = "user", run_id: str | None = None,
    ) -> Entity:
        with self.lock:
            self.index.sync()
            found = self.entities.find(project, "project")
            if found is None:
                raise ValueError("找不到该项目，请先用 memory_remember 保存项目记忆")
            entity, path = found
            snapshot = self.documents.snapshot()
            now = iso(self.clock())
            updated = entity.model_copy(update={"status": status, "updated_at": now})
            new_path = path
            if rename is not None and rename != entity.display_name:
                existing = self.entities.find(rename, "project")
                if existing is not None and existing[0].entity_id != entity.entity_id:
                    raise ValueError("项目名称已被使用，请换一个名称")
                updated = updated.model_copy(update={
                    "display_name": rename,
                    "aliases": tuple(dict.fromkeys((*entity.aliases, entity.display_name))),
                })
                new_path = self.entities.path_for(rename, "project", snapshot)
            self.documents.apply({new_path: self.entities.render(updated, snapshot[path])})
            if new_path != path:
                (self.workspace / path).unlink()
            self.documents.append_events([LedgerEvent(
                target_type="entity", target_id=entity.entity_id, action="update",
                actor=actor, at=now, run_id=run_id,
            )])
            self.index.sync([path, new_path])
            return updated

    def explain(self, claim_id: str, *, limit: int = 40) -> dict[str, object]:
        """Bounded bidirectional evidence graph; tombstones never expose payloads."""
        with self.lock:
            self.index.sync()
            pages = self.entities.pages()
            with self.index.connect() as conn:
                row = conn.execute(
                    "SELECT * FROM claims WHERE claim_id=?", (claim_id,),
                ).fetchone()
                if row is None:
                    raise ValueError(f"找不到记忆 ID：{claim_id}")
                if row["status"] == "retracted":
                    claim = self.index.row_claim(row)
                    return {"status": "已遗忘", "forgotten_at": claim.updated_at}
                pending = [claim_id]
                visited: set[str] = set()
                nodes: list[dict[str, object]] = []
                edges: set[tuple[str, str]] = set()
                while pending and len(visited) < limit:
                    current = pending.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    row = conn.execute(
                        "SELECT * FROM claims WHERE claim_id=?", (current,),
                    ).fetchone()
                    if row is None:
                        nodes.append({"claim_id": current, "status": "来源文件已移除"})
                    else:
                        claim = self.index.row_claim(row)
                        if claim.status == "retracted":
                            nodes.append({
                                "claim_id": current, "status": "已遗忘",
                                "forgotten_at": claim.updated_at,
                            })
                        else:
                            node = claim.to_public()
                            if (
                                claim.status == "active" and claim.expires_at is not None
                                and claim.expires_at <= iso(self.clock())
                            ):
                                node["status"] = "expired"
                            node.update({
                                "updated_at": claim.updated_at,
                                "evidence": claim.evidence.model_dump(),
                                "events": [dict(event) for event in conn.execute("""
                                    SELECT * FROM claim_events
                                    WHERE target_type='claim' AND target_id=? ORDER BY at, event_id
                                """, (current,))],
                                "entities": [
                                    pages[entity_id][0].model_dump()
                                    for entity_id in (claim.person_id, claim.project_id)
                                    if entity_id in pages
                                ],
                            })
                            nodes.append(node)
                    for link in conn.execute("""
                        SELECT new_id, old_id FROM claim_links
                        WHERE new_id=? OR old_id=? ORDER BY new_id, old_id LIMIT ?
                    """, (current, current, limit + 1)):
                        edges.add((link[0], link[1]))
                        neighbor = link[1] if link[0] == current else link[0]
                        if neighbor not in visited:
                            pending.append(neighbor)
                return {
                    "claim_id": claim_id, "claims": nodes,
                    "supersedes": [
                        {"new_id": new, "old_id": old} for new, old in sorted(edges)
                    ],
                    "truncated": bool(set(pending) - visited),
                }

    def project_lines(
        self, text: str, *, path: str, stable_core: bool = False,
    ) -> list[tuple[int, str]]:
        """Preserve source positions through validity and stable-core filtering."""
        with self.lock:
            self.index.sync()
            with self.index.connect() as conn:
                inactive_lines = {row[0] for row in conn.execute("""
                    SELECT ch.line_start FROM chunks ch LEFT JOIN claims c
                      ON c.claim_id=ch.claim_id AND c.path=ch.path AND c.line=ch.line_start
                    WHERE ch.path=? AND (
                        ch.current=0
                        OR COALESCE(c.expires_at, json_extract(ch.metadata, '$.expires_at'))<=?
                        OR (? AND COALESCE(c.kind, json_extract(ch.metadata, '$.kind'))
                            IN ('commitment', 'progress'))
                    )
                """, (path, iso(self.clock()), stable_core))}
            tombstones = self.documents.tombstones()
            kept: list[tuple[int, str]] = []
            for number, raw in enumerate(text.splitlines(), 1):
                if number in inactive_lines:
                    continue
                if stable_core and raw.strip() in (SECTIONS["commitment"], SECTIONS["progress"]):
                    continue
                markers = marked_claim_ids(raw)
                if any(("claim", claim_id) in tombstones for claim_id in markers):
                    continue
                if ("chunk", digest(line_text(raw))) in tombstones and not markers:
                    continue
                claim = decode_claim(raw, path, number, issues=set())
                if claim is not None and any(
                    ("entity", entity_id) in tombstones
                    for entity_id in (claim.person_id, claim.project_id)
                ):
                    continue
                kept.append((number, strip_claim_markers(raw)))
            return kept

    def project_text(self, text: str, *, path: str, stable_core: bool = False) -> str:
        lines = self.project_lines(text, path=path, stable_core=stable_core)
        return "\n".join(line for _, line in lines) + ("\n" if text.endswith("\n") else "")

    def search(
        self, query: str, *, limit: int = 5, min_score: float = 0.0,
        exclude: Iterable[tuple[str, int]] = (), include_history: bool = False,
        scopes: Sequence[str] = ("core", "bank", "projects", "people", "daily"),
        active_projects_only: bool = False, person_ids: Sequence[str] | None = None,
        exclude_texts: Iterable[str] = (), deduplicate: bool = False,
        semantic: QueryEmbedding | None = None,
    ) -> list[SearchHit]:
        with self.lock:
            self.index.sync()
            return self.index.search(
                query, now=iso(self.clock()), limit=limit, min_score=min_score,
                exclude=exclude, include_history=include_history,
                scopes=scopes, active_projects_only=active_projects_only, person_ids=person_ids,
                exclude_texts=exclude_texts, deduplicate=deduplicate, semantic=semantic,
            )
