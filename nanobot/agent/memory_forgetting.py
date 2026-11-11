"""Cleanup of explicitly supported local copies outside the memory projection."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from nanobot.config.paths import get_cli_history_path, get_data_dir
from nanobot.utils.file_io import write_text_atomic
from nanobot.utils.redaction import Redactor
from nanobot.utils.rotating_output import scrub_log

_SCRUB_LOCK = threading.Lock()


def _scrub_jsonl(path: Path, redact: Redactor) -> None:
    if path.is_symlink():
        raise ValueError("本地副本清理拒绝符号链接")
    content = path.read_text(encoding="utf-8")
    kept: list[str] = []
    for line in content.split("\n"):
        if not line.strip():
            kept.append(line)
            continue
        value = json.loads(line)
        cleaned = redact.json(value, identifiers=False)
        kept.append(line if cleaned == value else json.dumps(cleaned, ensure_ascii=False))
    cleaned = "\n".join(kept)
    if cleaned != content:
        write_text_atomic(path, cleaned)


def scrub_local_copies(workspace: Path, sessions_dir: Path | None, terms: list[str]) -> list[str]:
    """Return completed scopes; malformed user files and I/O failures propagate."""
    with _SCRUB_LOCK:
        return _scrub_local_copies(workspace, sessions_dir, terms)


def _scrub_local_copies(workspace: Path, sessions_dir: Path | None, terms: list[str]) -> list[str]:
    redact = Redactor(terms)
    completed: list[str] = []
    memory = workspace / "memory"
    if memory.is_symlink():
        raise ValueError("本地副本清理拒绝符号链接")
    for path in (memory / "HISTORY.md", *memory.glob("HISTORY.md.bak*")):
        if path.exists():
            if path.is_symlink():
                raise ValueError("本地副本清理拒绝符号链接")
            redact.scrub_lines(path)
    completed.append("旧 HISTORY 及迁移备份")
    cli = get_cli_history_path()
    if cli.exists():
        if cli.is_symlink():
            raise ValueError("本地副本清理拒绝符号链接")
        redact.scrub_lines(cli)
    completed.append("CLI 输入历史")

    directories = [workspace / "sessions"]
    if sessions_dir is not None:
        directories.extend([sessions_dir, sessions_dir / "jsonl-backup"])
    for directory in directories:
        if directory.is_symlink():
            raise ValueError("本地副本清理拒绝符号链接")
        for path in directory.glob("*"):
            if path.is_symlink():
                raise ValueError("本地副本清理拒绝符号链接")
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
    if webui.is_symlink():
        raise ValueError("本地副本清理拒绝符号链接")
    segment_dirs = list(webui.glob("*.segments"))
    if any(directory.is_symlink() for directory in segment_dirs):
        raise ValueError("本地副本清理拒绝符号链接")
    segment_files = [path for directory in segment_dirs for path in directory.glob("*.jsonl")]
    for path in (*webui.glob("*.jsonl"), *segment_files):
        _scrub_jsonl(path, redact)
    completed.append("WebUI 旧会话及分段副本")

    logs = data / "logs"
    if logs.is_symlink():
        raise ValueError("本地副本清理拒绝符号链接")
    bases = {path for prefix in ("gateway", "api") for path in logs.glob(f"{prefix}*.log*")}
    bases = {path if path.suffix == ".log" else path.with_suffix("") for path in bases}
    for path in sorted(bases):
        scrub_log(path, redact)
    completed.append("gateway/api 日志及轮转副本")
    return completed
