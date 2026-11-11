"""Stable person/project identities and their Markdown pages."""

from __future__ import annotations

import json
import re

from nanobot.agent.memory_documents import (
    ENTITY_RE,
    Entity,
    EntityKind,
    encode_entity,
)
from nanobot.agent.memory_index import MemoryIndex


class MemoryEntities:
    """Called under the claim store lock, after syncing its file projection."""

    def __init__(self, index: MemoryIndex) -> None:
        self.index = index

    def pages(self) -> dict[str, tuple[Entity, str]]:
        with self.index.connect() as conn:
            rows = conn.execute("""
                SELECT entity_id, entity_json, path FROM documents
                WHERE entity_id IS NOT NULL AND entity_id NOT IN (
                    SELECT target_id FROM tombstones WHERE target_type='entity'
                )
            """)
            pages: dict[str, tuple[Entity, str]] = {}
            for row in rows:
                data = json.loads(row["entity_json"])
                data["aliases"] = tuple(data["aliases"])
                pages[row["entity_id"]] = Entity.model_construct(**data), row["path"]
            return pages

    def find(self, name: str, kind: EntityKind) -> tuple[Entity, str] | None:
        needle = name.casefold()
        matches = [
            page for page in self.pages().values() if page[0].kind == kind
            and needle in {
                page[0].entity_id, page[0].display_name.casefold(),
                *(alias.casefold() for alias in page[0].aliases),
            }
        ]
        if len(matches) > 1:
            raise ValueError(f"名称“{name}”有歧义，请使用实体 ID")
        return matches[0] if matches else None

    @staticmethod
    def path_for(name: str, kind: EntityKind, snapshot: dict[str, str]) -> str:
        slug = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "", name).strip(" .")
        slug = re.sub(r"\s+", "-", slug).lower() or kind
        # Keep multibyte names within filename limits and avoid Windows device names.
        slug = slug[:60]
        if re.fullmatch(r"(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?", slug):
            slug = "_" + slug
        folder = "people" if kind == "person" else "projects"
        base = f"memory/{folder}/{slug}"
        path = f"{base}.md"
        suffix = 2
        while path in snapshot:
            path = f"{base}-{suffix}.md"
            suffix += 1
        return path

    @staticmethod
    def render(entity: Entity, content: str = "") -> str:
        if not content:
            sections = "\n## 进度\n\n## 决定\n\n## 待办\n" if entity.kind == "project" else ""
            return f"# {entity.display_name}\n\n{encode_entity(entity)}\n{sections}"
        content = re.sub(r"(?m)^# [^\n]*", lambda _: f"# {entity.display_name}", content, count=1)
        return ENTITY_RE.sub(lambda _: encode_entity(entity), content, count=1)

    def ensure(
        self, name: str, kind: EntityKind, aliases: tuple[str, ...], *,
        at: str, snapshot: dict[str, str], changes: dict[str, str],
    ) -> tuple[Entity, str]:
        found = self.find(name, kind)
        if found is None:
            entity = Entity.model_construct(
                kind=kind, display_name=name, aliases=aliases, created_at=at, updated_at=at,
            )
            path = self.path_for(name, kind, {**snapshot, **changes})
            changes[path] = self.render(entity)
        else:
            entity, path = found
            entity = entity.model_copy(update={
                "aliases": tuple(dict.fromkeys((*entity.aliases, *aliases))), "updated_at": at,
            })
            changes[path] = self.render(entity, snapshot[path])
        return entity, path
