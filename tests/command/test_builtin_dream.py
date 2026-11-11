from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from nanobot.agent.memory import MemoryStore
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.command.application import build_help_text, builtin_command_palette
from nanobot.command.builtin import (
    cmd_dream,
    cmd_dream_log,
    cmd_dream_prompt,
    cmd_dream_restore,
)
from nanobot.command.router import CommandContext
from nanobot.session.manager import SessionManager
from nanobot.utils.gitstore import CommitInfo


class _FakeStore:
    def __init__(self, git, last_dream_cursor: int = 1):
        self.git = git
        self._last_dream_cursor = last_dream_cursor

    def get_last_dream_cursor(self) -> int:
        return self._last_dream_cursor

    def restore_memory(self, sha: str, *, message_prefix: str) -> str | None:
        return self.git.revert(sha, message_prefix=message_prefix)


class _FakeGit:
    def __init__(
        self,
        *,
        initialized: bool = True,
        commits: list[CommitInfo] | None = None,
        diff_map: dict[str, tuple[CommitInfo, str] | None] | None = None,
        revert_result: str | None = None,
    ):
        self._initialized = initialized
        self._commits = commits or []
        self._diff_map = diff_map or {}
        self._revert_result = revert_result
        self.revert_calls: list[tuple[str, str | None]] = []

    def is_initialized(self) -> bool:
        return self._initialized

    def log(
        self,
        max_entries: int = 20,
        message_prefix: str | None = None,
    ) -> list[CommitInfo]:
        commits = self._commits
        if message_prefix is not None:
            commits = [c for c in commits if c.message.startswith(message_prefix)]
        return commits[:max_entries]

    def show_commit_diff(
        self,
        sha: str,
        max_entries: int = 20,
        message_prefix: str | None = None,
    ):
        result = self._diff_map.get(sha)
        if result and message_prefix is not None and not result[0].message.startswith(message_prefix):
            return None
        return result

    def revert(self, sha: str, *, message_prefix: str | None = None) -> str | None:
        self.revert_calls.append((sha, message_prefix))
        return self._revert_result

    def auto_commit(self, message: str) -> str | None:
        return None


class _FakeBus:
    def __init__(self):
        self.outbound = []

    async def publish_outbound(self, message):
        self.outbound.append(message)


def _make_sessions(tmp_path) -> SessionManager:
    return SessionManager(
        tmp_path / "workspace",
        sessions_root=tmp_path / "runtime",
    )


def _make_ctx(raw: str, git: _FakeGit, *, args: str = "", last_dream_cursor: int = 1) -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content=raw)
    store = _FakeStore(git, last_dream_cursor=last_dream_cursor)
    loop = SimpleNamespace(consolidator=SimpleNamespace(store=store))
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


def _make_dream_ctx(tmp_path) -> tuple[CommandContext, _FakeBus]:
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content="/dream")
    store = MemoryStore(tmp_path / "workspace")
    bus = _FakeBus()
    loop = SimpleNamespace(
        bus=bus,
        context=SimpleNamespace(memory=store, timezone="UTC"),
        sessions=_make_sessions(tmp_path),
    )
    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/dream", args="", loop=loop)
    return ctx, bus


def _make_dream_prompt_ctx(tmp_path, raw: str = "/dream-prompt", args: str = "") -> CommandContext:
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content=raw)
    loop = SimpleNamespace(context=SimpleNamespace(memory=MemoryStore(tmp_path)))
    return CommandContext(msg=msg, session=None, key=msg.session_key, raw=raw, args=args, loop=loop)


async def _run_dream_command(ctx: CommandContext) -> OutboundMessage:
    before = asyncio.all_tasks()
    immediate = await cmd_dream(ctx)
    try:
        await asyncio.gather(*(asyncio.all_tasks() - before))
    finally:
        await ctx.loop.sessions.state.aclose()
    return immediate


@pytest.mark.asyncio
async def test_dream_no_history_explains_how_to_create_input(tmp_path) -> None:
    ctx, bus = _make_dream_ctx(tmp_path)

    immediate = await _run_dream_command(ctx)

    assert immediate.content == "正在整理 Dream 记忆……"
    assert len(bus.outbound) == 1
    content = bus.outbound[0].content
    assert "Dream 暂无可处理的对话历史。" in content
    assert "`memory/history.jsonl`" in content
    assert "空闲自动压缩" in content
    assert "Dream 处理游标" in content
    assert "agents.defaults.idleCompactAfterMinutes" in content
    assert "/dream-prompt" in content


@pytest.mark.asyncio
async def test_dream_internal_run_silences_progress(tmp_path) -> None:
    msg = InboundMessage(channel="feishu", sender_id="u1", chat_id="chat1", content="/dream")
    store = MemoryStore(tmp_path / "workspace")
    store.append_history("需要整理的对话")
    bus = _FakeBus()
    calls = []
    progress_sentinel = "private Dream progress sentinel"

    async def process_direct(*args, **kwargs):
        calls.append((args, kwargs))
        await kwargs["on_progress"](progress_sentinel)
        await kwargs["on_progress"](
            "",
            tool_events=[{"phase": "start", "name": "edit_file", "hint": progress_sentinel}],
        )
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="done",
            metadata={"_stop_reason": "completed"},
        )

    dream_runtime = object()
    loop = SimpleNamespace(
        bus=bus,
        context=SimpleNamespace(memory=store, timezone="UTC"),
        sessions=_make_sessions(tmp_path),
        process_direct=process_direct,
        dream_runtime=lambda: dream_runtime,
    )
    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/dream", args="", loop=loop)

    immediate = await _run_dream_command(ctx)

    assert immediate.content == "正在整理 Dream 记忆……"
    assert (immediate.channel, immediate.chat_id) == ("feishu", "chat1")
    assert len(calls) == 1
    assert calls[0][1]["runtime"] is dream_runtime
    assert len(bus.outbound) == 1
    completion = bus.outbound[0]
    assert (completion.channel, completion.chat_id) == ("feishu", "chat1")
    assert completion.content.startswith("Dream 记忆整理完成")
    assert progress_sentinel not in completion.content


def _build_runnable_dream(
    tmp_path,
    *,
    initialized: bool,
    content_diff: str,
    stop_reason: str = "completed",
    tool_error: bool = False,
) -> tuple[CommandContext, MemoryStore]:
    """Exercise the command and shared lifecycle with a controlled model result."""
    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content="/dream")
    store = MemoryStore(tmp_path / "workspace")
    for index in range(6):
        store.append_history(f"记录 {index}")
    store.set_last_dream_cursor(5)
    if initialized:
        store.git.init()

    async def process_direct(*args, **kwargs):
        if content_diff:
            store.write_soul("# Soul\n- 新的行为规则")
        if tool_error:
            await kwargs["on_progress"](
                "",
                tool_events=[{
                    "phase": "error",
                    "name": "edit_file",
                    "error": "edit failed",
                }],
            )
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="done",
            metadata={"_stop_reason": stop_reason},
        )

    bus = _FakeBus()
    loop = SimpleNamespace(
        bus=bus,
        context=SimpleNamespace(memory=store, timezone="UTC"),
        sessions=_make_sessions(tmp_path),
        process_direct=process_direct,
        dream_runtime=lambda: None,
    )
    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/dream", args="", loop=loop)
    return ctx, store


@pytest.mark.asyncio
async def test_dream_advances_cursor_when_diff_nonempty(tmp_path) -> None:
    ctx, store = _build_runnable_dream(tmp_path, initialized=True, content_diff="SOUL.md: +1 -0")
    await _run_dream_command(ctx)
    assert store.get_last_dream_cursor() == 6


@pytest.mark.asyncio
async def test_dream_advances_cursor_on_completed_noop(tmp_path) -> None:
    ctx, store = _build_runnable_dream(tmp_path, initialized=True, content_diff="")
    await _run_dream_command(ctx)
    assert store.get_last_dream_cursor() == 6
    assert "记忆无变更" in ctx.loop.bus.outbound[0].content


@pytest.mark.asyncio
async def test_dream_keeps_cursor_when_incomplete_with_diff(tmp_path) -> None:
    ctx, store = _build_runnable_dream(
        tmp_path,
        initialized=True,
        content_diff="SOUL.md: +1 -0",
        stop_reason="length",
    )
    await _run_dream_command(ctx)
    assert store.get_last_dream_cursor() == 5
    assert "记忆整理未完成" in ctx.loop.bus.outbound[0].content


@pytest.mark.asyncio
async def test_dream_advances_cursor_when_completed_after_tool_error(tmp_path) -> None:
    ctx, store = _build_runnable_dream(
        tmp_path,
        initialized=True,
        content_diff="",
        tool_error=True,
    )
    await _run_dream_command(ctx)
    assert store.get_last_dream_cursor() == 6
    assert "记忆无变更" in ctx.loop.bus.outbound[0].content


@pytest.mark.asyncio
async def test_dream_noop_batch_unlocks_following_history(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    store = MemoryStore(workspace)
    store.write_soul("# Soul")
    store.write_memory("# Memory")
    for index in range(1, 22):
        store.append_history(f"entry-{index:02d}")
    store.git.init()

    processed_prompts: list[str] = []

    async def process_direct(prompt, *args, **kwargs):
        processed_prompts.append(prompt)
        return OutboundMessage(
            channel="cli",
            chat_id="direct",
            content="done",
            metadata={"_stop_reason": "completed"},
        )

    msg = InboundMessage(channel="cli", sender_id="u1", chat_id="direct", content="/dream")
    bus = _FakeBus()
    loop = SimpleNamespace(
        bus=bus,
        context=SimpleNamespace(memory=store, timezone="UTC"),
        sessions=_make_sessions(tmp_path),
        process_direct=process_direct,
        dream_runtime=lambda: None,
    )
    ctx = CommandContext(msg=msg, session=None, key=msg.session_key, raw="/dream", args="", loop=loop)

    await _run_dream_command(ctx)

    assert len(processed_prompts) == 1
    assert "entry-20" in processed_prompts[0]
    assert "entry-21" not in processed_prompts[0]
    assert store.get_last_dream_cursor() == 20
    next_result = store.build_dream_prompt()
    assert next_result is not None
    next_prompt, next_cursor = next_result.prompt, next_result.cursor
    assert next_cursor == 21
    assert "entry-21" in next_prompt
    assert "entry-01" not in next_prompt


@pytest.mark.asyncio
async def test_dream_initializes_new_workspace_and_completes(tmp_path) -> None:
    ctx, store = _build_runnable_dream(
        tmp_path, initialized=False, content_diff="", stop_reason="completed",
    )
    await _run_dream_command(ctx)
    assert store.get_last_dream_cursor() == 6


@pytest.mark.asyncio
async def test_dream_log_latest_is_more_user_friendly() -> None:
    commit = CommitInfo(sha="abcd1234", message="dream: 2026-04-04, 2 change(s)", timestamp="2026-04-04 12:00")
    diff = (
        "diff --git a/SOUL.md b/SOUL.md\n"
        "--- a/SOUL.md\n"
        "+++ b/SOUL.md\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    git = _FakeGit(commits=[commit], diff_map={commit.sha: (commit, diff)})

    out = await cmd_dream_log(_make_ctx("/dream-log", git))

    assert "## Dream 记忆变更" in out.content
    assert "以下是最新的 Dream 记忆变更。" in out.content
    assert "- 提交：`abcd1234`" in out.content
    assert "- 变更文件：`SOUL.md`" in out.content
    assert "使用 `/dream-restore abcd1234` 撤销这次变更。" in out.content
    assert "```diff" in out.content


@pytest.mark.asyncio
async def test_dream_log_latest_skips_non_dream_commit() -> None:
    backup = CommitInfo(
        sha="bbbb2222", message="backup: workspace snapshot", timestamp="2026-04-04 13:00",
    )
    dream = CommitInfo(
        sha="abcd1234", message="dream: latest", timestamp="2026-04-04 12:00",
    )
    diff = "diff --git a/SOUL.md b/SOUL.md\n"
    git = _FakeGit(
        commits=[backup, dream],
        diff_map={dream.sha: (dream, diff), backup.sha: (backup, "unrelated diff")},
    )

    out = await cmd_dream_log(_make_ctx("/dream-log", git))

    assert "`abcd1234`" in out.content
    assert "`bbbb2222`" not in out.content


@pytest.mark.asyncio
async def test_dream_log_missing_commit_guides_user() -> None:
    git = _FakeGit(diff_map={})

    out = await cmd_dream_log(_make_ctx("/dream-log deadbeef", git, args="deadbeef"))

    assert "找不到 Dream 变更 `deadbeef`。" in out.content
    assert "使用 `/dream-restore` 列出近期版本" in out.content


@pytest.mark.asyncio
async def test_dream_log_before_first_run_is_clear() -> None:
    git = _FakeGit(initialized=False)

    out = await cmd_dream_log(_make_ctx("/dream-log", git, last_dream_cursor=0))

    assert "Dream 尚未运行。" in out.content
    assert "运行 `/dream`" in out.content
    assert "/dream-prompt" in out.content


@pytest.mark.asyncio
async def test_dream_log_without_saved_versions_mentions_prompt_command() -> None:
    git = _FakeGit(initialized=True, commits=[])

    out = await cmd_dream_log(_make_ctx("/dream-log", git))

    assert "Dream 记忆尚无已保存的版本。" in out.content
    assert "/dream-prompt" in out.content


@pytest.mark.asyncio
async def test_dream_prompt_reports_default_prompt(tmp_path) -> None:
    out = await cmd_dream_prompt(_make_dream_prompt_ctx(tmp_path))

    assert "Dream 记忆整理规则：nanobot 默认" in out.content
    assert "prompts/dream.md" in out.content
    assert str(tmp_path) not in out.content
    assert "/dream-prompt init" in out.content


@pytest.mark.asyncio
async def test_dream_prompt_init_copies_default_prompt(tmp_path) -> None:
    ctx = _make_dream_prompt_ctx(tmp_path, "/dream-prompt init", "init")

    out = await cmd_dream_prompt(ctx)

    prompt_file = tmp_path / "prompts" / "dream.md"
    assert "已创建 Dream 记忆整理规则文件" in out.content
    assert "prompts/dream.md" in out.content
    assert str(tmp_path) not in out.content
    assert "完全替代当前工作区的 nanobot 默认 Dream 规则" in out.content
    assert prompt_file.read_text(encoding="utf-8") == MemoryStore.default_dream_prompt() + "\n"


@pytest.mark.asyncio
async def test_dream_prompt_init_does_not_overwrite_existing_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "prompts" / "dream.md"
    prompt_file.parent.mkdir()
    prompt_file.write_text("custom", encoding="utf-8")
    ctx = _make_dream_prompt_ctx(tmp_path, "/dream-prompt init", "init")

    out = await cmd_dream_prompt(ctx)

    assert "已存在" in out.content
    assert "prompts/dream.md" in out.content
    assert str(tmp_path) not in out.content
    assert prompt_file.read_text(encoding="utf-8") == "custom"


@pytest.mark.asyncio
async def test_dream_prompt_init_recreates_empty_prompt(tmp_path) -> None:
    prompt_file = tmp_path / "prompts" / "dream.md"
    prompt_file.parent.mkdir()
    prompt_file.write_text("  \n", encoding="utf-8")
    ctx = _make_dream_prompt_ctx(tmp_path, "/dream-prompt init", "init")

    out = await cmd_dream_prompt(ctx)

    assert "已创建 Dream 记忆整理规则文件" in out.content
    assert prompt_file.read_text(encoding="utf-8") == MemoryStore.default_dream_prompt() + "\n"


def test_dream_prompt_command_in_help_and_palette() -> None:
    palette = builtin_command_palette()
    dream_prompt = next(item for item in palette if item["command"] == "/dream-prompt")

    assert dream_prompt["arg_hint"] == "[init]"
    assert dream_prompt["lifecycle"] == "side_channel"
    assert dream_prompt["accepts_args"] is True
    assert "/dream-prompt [init]" in build_help_text()


@pytest.mark.asyncio
async def test_dream_restore_lists_versions_with_next_steps() -> None:
    commits = [
        CommitInfo(sha="abcd1234", message="dream: latest", timestamp="2026-04-04 12:00"),
        CommitInfo(sha="cccc3333", message="backup: workspace", timestamp="2026-04-04 10:00"),
        CommitInfo(sha="bbbb2222", message="dream: older", timestamp="2026-04-04 08:00"),
    ]
    git = _FakeGit(commits=commits)

    out = await cmd_dream_restore(_make_ctx("/dream-restore", git))

    assert "## Dream 记忆恢复" in out.content
    assert "选择要恢复的 Dream 记忆版本" in out.content
    assert "`abcd1234` 2026-04-04 12:00 - dream: latest" in out.content
    assert "`bbbb2222` 2026-04-04 08:00 - dream: older" in out.content
    assert "backup: workspace" not in out.content
    assert "使用 `/dream-log <sha>` 预览版本" in out.content
    assert "使用 `/dream-restore <sha>` 恢复版本。" in out.content


@pytest.mark.asyncio
async def test_dream_restore_success_mentions_files_and_followup() -> None:
    commit = CommitInfo(sha="abcd1234", message="dream: latest", timestamp="2026-04-04 12:00")
    diff = (
        "diff --git a/SOUL.md b/SOUL.md\n"
        "--- a/SOUL.md\n"
        "+++ b/SOUL.md\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "diff --git a/memory/MEMORY.md b/memory/MEMORY.md\n"
        "--- a/memory/MEMORY.md\n"
        "+++ b/memory/MEMORY.md\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    git = _FakeGit(
        diff_map={commit.sha: (commit, diff)},
        revert_result="eeee9999",
    )

    out = await cmd_dream_restore(_make_ctx("/dream-restore abcd1234", git, args="abcd1234"))

    assert "已将 Dream 记忆恢复至 `abcd1234` 变更之前的状态。" in out.content
    assert "- 新建的安全提交：`eeee9999`" in out.content
    assert "- 已恢复文件：`SOUL.md`, `memory/MEMORY.md`" in out.content
    assert "使用 `/dream-log eeee9999` 查看恢复产生的差异。" in out.content
    assert git.revert_calls == [("abcd1234", "dream:")]


@pytest.mark.asyncio
async def test_dream_restore_rejects_non_dream_commit_clearly() -> None:
    commit = CommitInfo(
        sha="cccc3333", message="backup: workspace", timestamp="2026-04-04 10:00",
    )
    git = _FakeGit(
        diff_map={commit.sha: (commit, "unrelated diff")},
        revert_result="eeee9999",
    )

    out = await cmd_dream_restore(_make_ctx("/dream-restore cccc3333", git, args="cccc3333"))

    assert "只能恢复 Dream 记忆版本。" in out.content
    assert "使用 `/dream-restore` 列出近期版本。" in out.content
    assert git.revert_calls == []
