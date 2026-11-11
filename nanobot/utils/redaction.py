"""Literal redaction for persisted text and JSON copies."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from nanobot.utils.file_io import write_text_atomic

_IDENTIFIERS = frozenset({
    "role", "type", "kind", "event", "id", "name", "key", "session_key",
    "parent_session_key", "tool_call_id", "call_id", "turn_id", "run_id",
    "request_id", "message_id", "chat_id", "sender_id", "phase", "status", "state",
    "_recovery_followup_id", "_webui_recovery_id", "generation", "session_generation",
})


class Redactor:
    def __init__(self, terms: Sequence[str]) -> None:
        self.terms = tuple(sorted({term for term in terms if term}, key=len, reverse=True))
        patterns = []
        for term in self.terms:
            escaped = re.escape(term)
            if re.fullmatch(r"[A-Za-z0-9_]+", term):
                escaped = rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])"
            patterns.append(escaped)
        self._pattern = re.compile("|".join(patterns), re.IGNORECASE) if patterns else None

    def matches(self, text: str) -> bool:
        return self._pattern is not None and self._pattern.search(text) is not None

    def text(self, text: str) -> str:
        return self._pattern.sub("[已遗忘]", text) if self._pattern is not None else text

    def json(self, value: Any, *, identifiers: bool = True) -> Any:
        if isinstance(value, dict):
            return {
                key: item if identifiers and key in _IDENTIFIERS and isinstance(item, str) else
                self.arguments(item) if key == "arguments" else
                self.json(item, identifiers=identifiers)
                for key, item in cast(dict[str, Any], value).items()
            }
        if isinstance(value, list):
            return [self.json(item, identifiers=identifiers) for item in cast(list[Any], value)]
        return self.text(value) if isinstance(value, str) else value

    def arguments(self, value: Any) -> Any:
        if not isinstance(value, str):
            return self.json(value, identifiers=False)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            # Interrupted provider output can contain incomplete arguments.
            return self.text(value)
        redacted = self.json(parsed, identifiers=False)
        return json.dumps(redacted, ensure_ascii=False) if redacted != parsed else value

    def scrub_lines(self, path: Path) -> None:
        content = path.read_text(encoding="utf-8")
        cleaned = "".join(self.text(line) for line in content.splitlines(keepends=True))
        if cleaned != content:
            write_text_atomic(path, cleaned)

