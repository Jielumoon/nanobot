from __future__ import annotations

import json
import os
import tempfile
import time
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from nanobot.bus.events import INBOUND_META_TURN_ID
from nanobot.utils.atomic_file import replace_atomic

RESTART_NOTIFY_CHANNEL_ENV = "NANOBOT_RESTART_NOTIFY_CHANNEL"
RESTART_NOTIFY_CHAT_ID_ENV = "NANOBOT_RESTART_NOTIFY_CHAT_ID"
RESTART_NOTIFY_METADATA_ENV = "NANOBOT_RESTART_NOTIFY_METADATA"
RESTART_STARTED_AT_ENV = "NANOBOT_RESTART_STARTED_AT"


@dataclass(frozen=True)
class RestartNotice:
    channel: str
    chat_id: str
    started_at_raw: str
    metadata: dict[str, Any] = field(default_factory=dict)
    update_target: str | None = None
    previous_version: str | None = None
    previous_commit: str | None = None


def save_update_notice(path: Path, notice: RestartNotice) -> None:
    """原子保存私有回执，交给重启后的进程发送。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".update-notice-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(asdict(notice), handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        replace_atomic(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def read_update_notice(path: Path) -> RestartNotice | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    fields = ("channel", "chat_id", "started_at_raw", "update_target", "previous_version")
    if not isinstance(raw, dict) or any(
        not isinstance(raw.get(key), str) or not raw[key].strip() for key in fields
    ):
        raise ValueError("更新回执缺少有效的来源或版本")
    if not isinstance(raw.get("metadata", {}), dict) or (
        raw.get("previous_commit") is not None and not isinstance(raw["previous_commit"], str)
    ):
        raise ValueError("更新回执格式无效")
    try:
        return RestartNotice(**raw)
    except TypeError as exc:
        raise ValueError("更新回执字段无效") from exc


def format_update_completed_message(notice: RestartNotice) -> str:
    from nanobot import __version__
    from nanobot.build_info import COMMIT

    current = f"{__version__}（{COMMIT[:9]}）" if COMMIT else __version__
    previous = notice.previous_version or "未知"
    if notice.previous_commit:
        previous += f"（{notice.previous_commit[:9]}）"
    content = f"服务已恢复运行。\n当前版本：{current}\n更新前：{previous}\n更新目标：{notice.update_target}"
    if notice.previous_version == __version__ and notice.previous_commit == COMMIT:
        content += "\n版本标识未变化，未确认更新成功；请核对更新记录。"
    return content


def format_restart_completed_message(started_at_raw: str) -> str:
    """Build restart completion text and include elapsed time when available."""
    elapsed_suffix = ""
    if started_at_raw:
        with suppress(ValueError):
            elapsed_s = max(0.0, time.time() - float(started_at_raw))
            elapsed_suffix = f"，用时 {elapsed_s:.1f}s"
    return f"重启完成{elapsed_suffix}。"


def set_restart_notice_to_env(
    *, channel: str, chat_id: str, metadata: dict[str, Any] | None = None,
) -> None:
    """Write restart notice env values for the next process."""
    os.environ[RESTART_NOTIFY_CHANNEL_ENV] = channel
    os.environ[RESTART_NOTIFY_CHAT_ID_ENV] = chat_id
    os.environ[RESTART_STARTED_AT_ENV] = str(time.time())
    persisted_metadata = dict(metadata or {})
    persisted_metadata.pop(INBOUND_META_TURN_ID, None)
    if persisted_metadata:
        try:
            os.environ[RESTART_NOTIFY_METADATA_ENV] = json.dumps(
                persisted_metadata,
                default=str,
            )
        except (TypeError, ValueError):
            os.environ.pop(RESTART_NOTIFY_METADATA_ENV, None)
    else:
        os.environ.pop(RESTART_NOTIFY_METADATA_ENV, None)


def consume_restart_notice_from_env() -> RestartNotice | None:
    """Read and clear restart notice env values once for this process."""
    channel = os.environ.pop(RESTART_NOTIFY_CHANNEL_ENV, "").strip()
    chat_id = os.environ.pop(RESTART_NOTIFY_CHAT_ID_ENV, "").strip()
    started_at_raw = os.environ.pop(RESTART_STARTED_AT_ENV, "").strip()
    metadata_raw = os.environ.pop(RESTART_NOTIFY_METADATA_ENV, "").strip()
    if not (channel and chat_id):
        return None
    metadata: dict[str, Any] = {}
    if metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
        except (TypeError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            metadata = cast(dict[str, Any], parsed)
    return RestartNotice(
        channel=channel,
        chat_id=chat_id,
        started_at_raw=started_at_raw,
        metadata=metadata,
    )


def should_show_cli_restart_notice(notice: RestartNotice, session_id: str) -> bool:
    """Return True when a restart notice should be shown in this CLI session."""
    if notice.channel != "cli":
        return False
    if ":" in session_id:
        _, cli_chat_id = session_id.split(":", 1)
    else:
        cli_chat_id = session_id
    return not notice.chat_id or notice.chat_id == cli_chat_id
