"""Cleanup of explicitly supported local copies outside the memory projection."""

from __future__ import annotations

import json
from pathlib import Path

from nanobot.config.paths import get_cli_history_path, get_data_dir
from nanobot.utils.file_io import write_text_atomic
from nanobot.utils.redaction import Redactor
from nanobot.utils.rotating_output import scrub_log


def _scrub_jsonl(path: Path, redact: Redactor) -> None:
    content = path.read_text(encoding="utf-8")
    kept: list[str] = []
    for line in content.split("\n"):
        if not line.strip():
            kept.append(line)
            continue
        value = json.loads(line)
        if redact.json(value, identifiers=False) == value:
            kept.append(line)
    cleaned = "\n".join(kept)
    if cleaned != content:
        write_text_atomic(path, cleaned)


def scrub_local_copies(workspace: Path, sessions_dir: Path | None, terms: list[str]) -> list[str]:
    """Return completed scopes; malformed user files and I/O failures propagate."""
    redact = Redactor(terms)
    completed: list[str] = []
    memory = workspace / "memory"
    for path in (memory / "HISTORY.md", *memory.glob("HISTORY.md.bak*")):
        if path.exists():
            redact.scrub_lines(path)
    completed.append("旧 HISTORY 及迁移备份")
    cli = get_cli_history_path()
    if cli.exists():
        redact.scrub_lines(cli)
    completed.append("CLI 输入历史")

    directories = [workspace / "sessions"]
    if sessions_dir is not None:
        directories.extend([sessions_dir, sessions_dir / "jsonl-backup"])
    for directory in directories:
        for path in directory.glob("*"):
            if path.suffix == ".jsonl":
                _scrub_jsonl(path, redact)
            elif path.name.endswith(".checkpoint.json"):
                value = json.loads(path.read_text(encoding="utf-8"))
                cleaned = redact.json(value)
                if cleaned != value:
                    write_text_atomic(path, json.dumps(cleaned, ensure_ascii=False) + "\n")
    completed.append("旧会话 JSONL 及检查点")

    data = get_data_dir()
    webui = data / "webui"
    for path in (*webui.glob("*.jsonl"), *webui.glob("*.segments/*.jsonl")):
        _scrub_jsonl(path, redact)
    completed.append("WebUI 旧会话及分段副本")

    logs = data / "logs"
    bases = {path for prefix in ("gateway", "api") for path in logs.glob(f"{prefix}*.log*")}
    bases = {path if path.suffix == ".log" else path.with_suffix("") for path in bases}
    for path in sorted(bases):
        scrub_log(path, redact)
    completed.append("gateway/api 日志及轮转副本")
    return completed
