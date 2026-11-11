"""Workspace admission for Dream and snapshot restore; no lock spans model waits."""

import threading
from pathlib import Path

_RUNNING: set[Path] = set()
_LOCK = threading.Lock()


class MemoryOperationBusyError(ValueError):
    """Dream or restore already owns this workspace's admission slot."""


def memory_operation_running(workspace: Path) -> bool:
    with _LOCK:
        return workspace.resolve() in _RUNNING


def acquire_memory_operation(workspace: Path) -> bool:
    with _LOCK:
        workspace = workspace.resolve()
        if workspace in _RUNNING:
            return False
        _RUNNING.add(workspace)
        return True


def release_memory_operation(workspace: Path) -> None:
    with _LOCK:
        _RUNNING.remove(workspace.resolve())
