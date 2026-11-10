"""One synchronized read view shared by stable core and current-turn memory."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from nanobot.agent.memory_documents import (
    SECTIONS,
    Claim,
    Entity,
    decode_claim,
    digest,
    line_text,
    marked_claim_ids,
    strip_claim_markers,
)
from nanobot.agent.memory_index import MemoryIndex
from nanobot.runtime_context import RuntimeContextBlock

_PERSON_ID = """COALESCE(c.person_id, (
    SELECT entity_id FROM documents page WHERE page.path=c.path AND page.scope='people'
))"""


@dataclass(frozen=True)
class PromptMemory:
    user: str
    memory: str
    runtime: RuntimeContextBlock | None = None
    generation: str = ""


class MemoryReadView:
    """Used only while the store lock and this SQLite read transaction are held."""

    def __init__(self, index: MemoryIndex, conn: sqlite3.Connection, now: str) -> None:
        self.index = index
        self.conn = conn
        self.now = now
        self.tombstones = {
            (row[0], row[1]) for row in conn.execute("SELECT target_type, target_id FROM tombstones")
        }

    def project_lines(
        self, text: str, *, path: str, stable_core: bool = False,
    ) -> list[tuple[int, str]]:
        inactive = {row[0] for row in self.conn.execute("""
            SELECT ch.line_start FROM chunks ch LEFT JOIN claims c ON c.claim_id=ch.claim_id
            WHERE ch.path=? AND (
                ch.current=0
                OR COALESCE(c.expires_at, json_extract(ch.metadata, '$.expires_at'))<=?
                OR (? AND (
                    COALESCE(c.kind, json_extract(ch.metadata, '$.kind'))
                        IN ('commitment', 'progress')
                    OR json_extract(ch.metadata, '$.person_id') IS NOT NULL
                    OR json_extract(ch.metadata, '$.project_id') IS NOT NULL
                ))
            )
        """, (path, self.now, stable_core))}
        kept: list[tuple[int, str]] = []
        for number, raw in enumerate(text.splitlines(), 1):
            if number in inactive:
                continue
            if stable_core and raw.strip() in (SECTIONS["commitment"], SECTIONS["progress"]):
                continue
            markers = marked_claim_ids(raw)
            if any(("claim", claim_id) in self.tombstones for claim_id in markers):
                continue
            if ("chunk", digest(line_text(raw))) in self.tombstones and not markers:
                continue
            claim = decode_claim(raw, path, number, issues=set())
            if claim is not None and any(
                ("entity", entity_id) in self.tombstones
                for entity_id in (claim.person_id, claim.project_id)
            ):
                continue
            kept.append((number, strip_claim_markers(raw)))
        return kept

    def people(self) -> list[Entity]:
        return [Entity.model_construct(**json.loads(row[0])) for row in self.conn.execute("""
            SELECT entity_json FROM documents
            WHERE scope='people' AND entity_json IS NOT NULL
        """)]

    def current_filter(self, mentioned: set[str]) -> tuple[str, list[object]]:
        placeholders = ",".join("?" for _ in mentioned)
        return f"""
            c.status='active' AND (c.expires_at IS NULL OR c.expires_at>?)
            AND (
                {_PERSON_ID} IS NULL AND NOT EXISTS (
                    SELECT 1 FROM documents p WHERE p.path=c.path AND p.scope='people'
                ) OR {_PERSON_ID} IN ({placeholders})
            )
            AND NOT EXISTS (
                SELECT 1 FROM documents d
                WHERE (d.entity_id=c.project_id OR d.path=c.path AND d.scope='projects')
                  AND json_extract(d.entity_json, '$.status')!='active'
            )
        """, [self.now, *sorted(mentioned)]

    def claims(
        self, mentioned: set[str], *, kind: str | None = None,
        person_id: str | None = None, project_id: str | None = None, limit: int,
    ) -> list[Claim]:
        where, params = self.current_filter(mentioned)
        for column, value in (("kind", kind), ("project_id", project_id)):
            if value is not None:
                where += f" AND c.{column}=?"
                params.append(value)
        if person_id is not None:
            where += """
                AND (c.person_id=? OR c.person_id IS NULL AND c.path IN (
                    SELECT path FROM documents WHERE scope='people' AND entity_id=?
                ))
            """
            params.extend((person_id, person_id))
        if project_id is not None:
            where += " AND c.kind IN ('progress', 'commitment')"
        order = (
            "c.due_at IS NULL, c.due_at, c.claim_id" if kind == "commitment"
            else "json_extract(c.metadata, '$.updated_at') DESC, c.path, c.line DESC, c.claim_id"
        )
        return [self.index.row_claim(row).model_copy(update={"person_id": row["context_person_id"]})
                for row in self.conn.execute(f"""
            SELECT c.*, {_PERSON_ID} AS context_person_id
            FROM claims c WHERE {where} ORDER BY {order} LIMIT ?
        """, [*params, limit])]

    def projects(self, mentioned: set[str]) -> list[Entity]:
        where, params = self.current_filter(mentioned)
        rows = self.conn.execute(f"""
            SELECT d.entity_json FROM documents d
            LEFT JOIN claims c ON c.project_id=d.entity_id AND {where}
            WHERE d.scope='projects' AND json_extract(d.entity_json, '$.status')='active'
            GROUP BY d.entity_id
            ORDER BY MAX(
                COALESCE(MAX(json_extract(c.metadata, '$.updated_at')), ''),
                COALESCE(json_extract(d.entity_json, '$.updated_at'), '')
            ) DESC, d.entity_id LIMIT 3
        """, params)
        return [Entity.model_construct(**json.loads(row[0])) for row in rows]
