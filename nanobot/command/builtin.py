from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from nanobot import __version__
from nanobot.bus.events import OutboundMessage
from nanobot.command.router import CommandContext, CommandRouter
from nanobot.command.spec import BuiltinCommandSpec
from nanobot.command.spec import CommandLifecycle as CommandLifecycle
from nanobot.command.user_shell import cmd_user_shell
from nanobot.providers.base import LLMUsage
from nanobot.utils.cancellation import shield_and_drain
from nanobot.utils.helpers import build_status_content
from nanobot.utils.restart import set_restart_notice_to_env
from nanobot.utils.workspace_prompts import initialize_workspace_prompt

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop
    from nanobot.session.manager import Session
    from nanobot.utils.gitstore import CommitInfo

USER_SHELL_COMMAND = "/__shell"


CORE_COMMAND_SPECS: tuple[BuiltinCommandSpec, ...] = (
    BuiltinCommandSpec(
        "/new",
        "New chat",
        "重置当前聊天并开始新对话。",
        "square-pen",
        lifecycle="finalize_active_turn",
    ),
    BuiltinCommandSpec(
        "/compact",
        "Compact context",
        "压缩当前聊天的上下文并继续对话。",
        "archive",
    ),
    BuiltinCommandSpec(
        "/stop",
        "Stop current task",
        "停止当前聊天的本轮执行。",
        "square",
        lifecycle="stop_active_turn",
    ),
    BuiltinCommandSpec(
        "/restart",
        "Restart nanobot",
        "重启机器人进程。",
        "rotate-cw",
    ),
    BuiltinCommandSpec(
        "/status",
        "Show status",
        "查看运行、模型服务及渠道状态。",
        "activity",
    ),
    BuiltinCommandSpec(
        "/model",
        "Switch model preset",
        "查看或切换当前模型预设。",
        "brain",
        "[preset]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/history",
        "Show conversation history",
        "显示最近 N 条已保存的对话消息。",
        "history",
        "[n]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/trigger",
        "Create named local trigger",
        "创建绑定当前会话的命名 CLI 触发器。",
        "zap",
        "<name>",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/dream",
        "Run Dream",
        "手动触发记忆整理。",
        "sparkles",
    ),
    BuiltinCommandSpec(
        "/dream-log",
        "Show Dream log",
        "查看上次 Dream 记忆整理的变更。",
        "book-open",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/dream-restore",
        "Restore memory",
        "将记忆恢复至之前的 Dream 快照。",
        "undo-2",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/dream-prompt",
        "Dream memory",
        "自定义当前工作区的 Dream 记忆整理规则。",
        "file-text",
        "[init]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/evaluator-prompt",
        "Heartbeat evaluator",
        "自定义当前工作区心跳通知的审核提示词。",
        "file-text",
        "[init]",
        accepts_args=True,
    ),
    BuiltinCommandSpec(
        "/skill",
        "List skills",
        "列出已启用且可用的技能。",
        "wrench",
    ),
    BuiltinCommandSpec(
        "/help",
        "Show help",
        "列出可用的斜杠命令。",
        "circle-help",
    ),
    BuiltinCommandSpec(
        "/pairing",
        "Manage pairing",
        "查看、批准、拒绝配对请求，或撤销授权。",
        "shield",
        "[list|approve <code>|deny <code>|revoke <user_id>]",
        accepts_args=True,
    ),
)


async def cmd_stop(ctx: CommandContext) -> OutboundMessage:
    """Cancel all active tasks and subagents for the session."""
    loop = ctx.loop
    msg = ctx.msg
    total = await loop._cancel_active_tasks(ctx.key)  # pyright: ignore[reportPrivateUsage]
    # Also drain pending queue to prevent mid-turn injection deadlock
    pending = loop._pending_queues.pop(ctx.key, None)  # pyright: ignore[reportPrivateUsage]
    if pending is not None:
        while not pending.empty():
            try:
                pending.get_nowait()
                total += 1
            except Exception:
                break
    content = f"已停止 {total} 个任务。" if total else "没有正在运行的任务。"
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content=content,
        metadata=dict(msg.metadata or {})
    )


async def cmd_restart(ctx: CommandContext) -> OutboundMessage:
    """Restart the process."""
    msg = ctx.msg
    set_restart_notice_to_env(
        channel=msg.channel,
        chat_id=msg.chat_id,
        metadata=dict(msg.metadata or {}),
    )

    async def _do_restart():
        await asyncio.sleep(1)
        argv = [sys.executable, "-m", "nanobot"] + sys.argv[1:]
        mode = ctx.loop.restart_mode or "auto"
        if mode == "auto":
            mode = "spawn" if sys.platform == "win32" else "exec"
        if mode == "exec":
            os.execv(sys.executable, argv)
            return
        if mode == "spawn":
            kwargs: dict[str, Any] = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            subprocess.Popen(argv, **kwargs)
        os._exit(0)

    asyncio.create_task(_do_restart())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="正在重启……",
        metadata=dict(msg.metadata or {})
    )


async def cmd_status(ctx: CommandContext) -> OutboundMessage:
    """Build an outbound status message for a session."""
    loop = ctx.loop
    session = ctx.session or await loop.sessions.state.get(ctx.key)
    runtime = ctx.runtime or await loop.runtime_for_session(session)
    ctx_est = 0
    with suppress(Exception):
        ctx_est, _ = loop.consolidator.estimate_session_prompt_tokens(
            session,
            runtime=runtime,
        )
    last_usage = LLMUsage.from_dict(session.metadata.get("_last_usage"))
    if ctx_est <= 0:
        ctx_est = last_usage.input_tokens if last_usage is not None else 0

    # Await web search provider usage; network requests have a bounded timeout.
    search_usage_text: str | None = None
    # Never let usage fetch break /status
    with suppress(Exception):
        from nanobot.utils.searchusage import fetch_search_usage
        search_cfg = loop.web_config.search
        usage = await fetch_search_usage(
            provider=search_cfg.provider,
            api_key=search_cfg.api_key or None,
        )
        search_usage_text = usage.format()
    active_tasks = loop._active_tasks.get(ctx.key, [])  # pyright: ignore[reportPrivateUsage]
    task_count = sum(1 for t in active_tasks if not t.done())
    with suppress(Exception):
        task_count += loop.subagents.get_running_count_by_session(ctx.key)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_status_content(
            version=__version__, model=runtime.model,
            start_time=loop._start_time, last_usage=last_usage,  # pyright: ignore[reportPrivateUsage]
            context_window_tokens=runtime.context_window_tokens,
            session_msg_count=len(session.get_history(max_messages=0)),
            context_tokens_estimate=ctx_est,
            search_usage_text=search_usage_text,
            active_task_count=task_count,
            max_completion_tokens=runtime.generation.max_tokens,
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_new(ctx: CommandContext) -> OutboundMessage:
    """Stop active task and start a fresh session."""
    loop = ctx.loop
    await loop._cancel_active_tasks(ctx.key)  # pyright: ignore[reportPrivateUsage]
    loop.discard_session_file_state(ctx.key)
    session = ctx.session or await loop.sessions.state.get(ctx.key)
    snapshot = list(session.messages)
    archive_snapshot = None
    runtime = None
    if session.last_archived < len(snapshot):
        runtime = ctx.runtime or await loop.runtime_for_session(session)
        archive_snapshot = replace(
            session,
            messages=snapshot,
            metadata=dict(session.metadata),
            provider_state=None,
        )
    async def reset_and_schedule_archive() -> None:
        await loop.sessions.state.reset(session.key)
        if archive_snapshot is not None and runtime is not None:
            loop.schedule_background(
                loop.consolidator.archive_session(  # pyright: ignore[reportUnknownMemberType]
                    archive_snapshot,
                    archive_end=len(snapshot),
                    runtime=runtime,
                )
            )

    await shield_and_drain(reset_and_schedule_archive())
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content="已开始新对话。",
        metadata=dict(ctx.msg.metadata or {})
    )


async def cmd_compact(ctx: CommandContext) -> None:
    """Compact the current session without resetting the conversation."""
    loop = ctx.loop
    session = ctx.session or await loop.sessions.state.get(ctx.key)
    runtime = ctx.runtime or await loop.runtime_for_session(session)
    delivery = loop.turn_delivery_factory.create(ctx.msg, ctx.key)

    try:
        await loop.consolidator.compact_idle_session(
            ctx.key,
            runtime=runtime,
            events=delivery.events,
            notify=True,
            reason="manual",
        )
    except Exception:
        logger.exception("Manual context compaction failed for {}", ctx.key)
        return



def _format_preset_names(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "（尚未配置）"


def _model_preset_names(loop: AgentLoop) -> list[str]:
    names = set(loop.model_presets)
    names.add("default")
    return ["default", *sorted(name for name in names if name != "default")]


def _command_error_message(exc: Exception) -> str:
    return str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc)


async def _model_command_status(loop: AgentLoop, session: Session) -> str:
    names = _model_preset_names(loop)
    try:
        runtime = await loop.runtime_for_session(session, recover_removed=False)
    except (KeyError, ValueError) as exc:
        return "\n".join([
            "## 模型",
            f"- 当前模型配置错误：{_command_error_message(exc)}",
            f"- 可用预设：{_format_preset_names(names)}",
            "- 使用 `/model <preset>` 切换预设。",
        ])
    active = runtime.model_preset or "default"
    return "\n".join([
        "## 模型",
        f"- 当前模型：`{runtime.model}`",
        f"- 当前预设：`{active}`",
        f"- 可用预设：{_format_preset_names(names)}",
    ])


async def cmd_model(ctx: CommandContext) -> OutboundMessage:
    """Show or switch model presets."""
    loop = ctx.loop
    args = ctx.args.strip()
    metadata = {**dict(ctx.msg.metadata or {}), "render_as": "text"}

    if not args:
        session = ctx.session or await loop.sessions.state.get(ctx.key)
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=await _model_command_status(loop, session),
            metadata=metadata,
        )

    name = args
    try:
        runtime = await loop.set_session_model_preset(ctx.key, name)
    except (KeyError, ValueError) as exc:
        names = _model_preset_names(loop)
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=(
                f"无法切换模型预设：{_command_error_message(exc)}\n\n"
                f"可用预设：{_format_preset_names(names)}"
            ),
            metadata=metadata,
        )

    max_tokens = runtime.generation.max_tokens
    lines = [
        f"已切换至模型预设 `{runtime.model_preset}`。",
        "- 生效范围：当前会话",
        f"- 模型：`{runtime.model}`",
        f"- 上下文窗口：{runtime.context_window_tokens}",
    ]
    lines.append(f"- 最大输出 token 数：{max_tokens}")
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content="\n".join(lines),
        metadata=metadata,
    )


async def cmd_dream(ctx: CommandContext) -> OutboundMessage:
    """Manually trigger a Dream consolidation run."""
    import time

    loop = ctx.loop
    msg = ctx.msg

    async def _run_dream():
        from nanobot.agent.memory import MemoryStore

        async def _silent(*_args: Any, **_kwargs: Any) -> None:
            pass

        dream_session_key = MemoryStore.dream_session_key
        build_dream_commit_message = MemoryStore.build_dream_commit_message

        store = loop.context.memory
        content = ""
        resp = None
        diff_body = ""
        t0 = time.monotonic()
        try:
            result = store.build_dream_prompt()
            if result is None:
                await loop.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content=_format_dream_no_input_message(),
                    metadata={"render_as": "text"},
                ))
                return
            prompt, last_cursor = result
            key = dream_session_key()
            dream_runtime = loop.dream_runtime()
            resp = await loop.process_direct(
                prompt,
                session_key=key,
                ephemeral=True,
                tools=store.build_dream_tools(),
                on_progress=_silent,
                runtime=dream_runtime,
            )
            elapsed = time.monotonic() - t0
            # The real file delta grounds the audit record; normal completion
            # decides whether this history batch has finished processing.
            diff_body = store.dream_content_diff()
            completed = MemoryStore.dream_run_completed(resp)
            if completed:
                store.set_last_dream_cursor(last_cursor)
                if diff_body:
                    content = f"Dream 记忆整理完成，用时 {elapsed:.1f}s。"
                else:
                    content = f"Dream 记忆整理完成，用时 {elapsed:.1f}s；记忆无变更。"
            else:
                reason = MemoryStore.dream_incompletion_reason(resp)
                content = (
                    f"Dream 记忆整理未完成，用时 {elapsed:.1f}s（{reason}）；"
                    "记忆处理游标未推进。"
                )
        except Exception as e:
            elapsed = time.monotonic() - t0
            content = f"Dream 记忆整理失败，用时 {elapsed:.1f}s：{e}"
        finally:
            if store.git.is_initialized():
                commit_msg = build_dream_commit_message("dream: manual run", diff_body)
                sha = store.git.auto_commit(commit_msg)
                if sha:
                    content += f"（提交 {sha}）"
            store.compact_history()
            await loop.sessions.state.prune_dream_sessions()
        await loop.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    asyncio.create_task(_run_dream())
    return OutboundMessage(
        channel=msg.channel, chat_id=msg.chat_id, content="正在整理 Dream 记忆……",
    )


async def cmd_dream_prompt(ctx: CommandContext) -> OutboundMessage:
    """Show or set up the workspace Dream memory instructions."""
    store = ctx.loop.context.memory
    path = store.dream_prompt_file
    display_path = path.relative_to(store.workspace).as_posix()
    args = ctx.args.strip().lower()

    if args == "init":
        if not initialize_workspace_prompt(path, store.default_dream_prompt()):
            content = (
                f"Dream 记忆整理规则文件已存在：`{display_path}`。\n\n"
                "可直接编辑该文件；删除或清空文件即可恢复 nanobot 默认规则。"
            )
        else:
            content = (
                f"已创建 Dream 记忆整理规则文件：`{display_path}`。\n\n"
                "编辑该文件可指定 Dream 如何整理记忆。"
                "该文件会完全替代当前工作区的 nanobot 默认 Dream 规则。"
                "删除或清空文件即可恢复默认规则。"
            )
    elif args:
        content = "用法：/dream-prompt [init]"
    elif store.has_dream_prompt_override():
        content = (
            "Dream 记忆整理规则：当前工作区自定义\n\n"
            f"- 路径：`{display_path}`\n"
            "- 删除或清空此文件即可恢复 nanobot 默认规则。"
        )
    else:
        content = (
            "Dream 记忆整理规则：nanobot 默认\n\n"
            f"- 可编辑文件：`{display_path}`\n"
            "- 运行 `/dream-prompt init` 创建可编辑副本。"
        )

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_evaluator_prompt(ctx: CommandContext) -> OutboundMessage:
    """Show or set up the workspace heartbeat evaluator prompt."""
    from nanobot.utils.evaluator import (
        default_evaluator_prompt,
        evaluator_prompt_file,
        has_evaluator_prompt_override,
    )

    workspace = ctx.loop.context.memory.workspace
    path = evaluator_prompt_file(workspace)
    display_path = path.relative_to(workspace).as_posix()
    args = ctx.args.strip().lower()

    if args == "init":
        if not initialize_workspace_prompt(path, default_evaluator_prompt()):
            content = (
                f"心跳通知审核提示词文件已存在：`{display_path}`。\n\n"
                "可直接编辑该文件；删除或清空文件即可恢复 nanobot 默认提示词。"
            )
        else:
            content = (
                f"已创建心跳通知审核提示词文件：`{display_path}`。\n\n"
                "编辑该文件可控制何时发送心跳通知。"
                "提示词仍须要求模型调用 `evaluate_notification` 工具，"
                "否则审核会拒绝放行，不发送通知。"
                "删除或清空文件即可恢复 nanobot 默认提示词。"
            )
    elif args:
        content = "用法：/evaluator-prompt [init]"
    elif has_evaluator_prompt_override(workspace):
        content = (
            "心跳通知审核提示词：当前工作区自定义\n\n"
            f"- 路径：`{display_path}`\n"
            "- 删除或清空此文件即可恢复 nanobot 默认提示词。"
        )
    else:
        content = (
            "心跳通知审核提示词：nanobot 默认\n\n"
            f"- 可编辑文件：`{display_path}`\n"
            "- 运行 `/evaluator-prompt init` 创建可编辑副本。"
        )

    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


def _format_dream_no_input_message() -> str:
    return "\n".join([
        "Dream 暂无可处理的对话历史。",
        "",
        "Dream 从 `memory/history.jsonl` 读取当前处理游标之后的新记录。",
        (
            "短对话只有在触发 token 压缩或空闲自动压缩后才会写入该文件，"
            "因此新建或较短的 WebUI 对话可能尚无 Dream 可处理的内容。"
        ),
        "",
        "接下来可以：",
        "- 启用 `agents.defaults.idleCompactAfterMinutes`，让结束的对话自动成为 Dream 的输入。",
        "- 手动压缩操作可用时，将当前对话压缩到记忆中。",
        "- 如果预期已有历史，请检查 `memory/history.jsonl` 在 Dream 处理游标之后是否有新记录。",
        "- 使用 `/dream-prompt` 查看或修改 Dream 记忆整理规则。",
    ])


def _extract_changed_files(diff: str) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for line in diff.splitlines():
        if not line.startswith("diff --git "):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        path = parts[3]
        if path.startswith("b/"):
            path = path[2:]
        if path in seen:
            continue
        seen.add(path)
        files.append(path)
    return files


def _format_changed_files(diff: str) -> str:
    files = _extract_changed_files(diff)
    if not files:
        return "受版本管理的记忆文件没有变更。"
    return ", ".join(f"`{path}`" for path in files)


_DREAM_COMMIT_PREFIX = "dream:"


def _format_dream_log_content(
    commit: CommitInfo,
    diff: str,
    *,
    requested_sha: str | None = None,
) -> str:
    files_line = _format_changed_files(diff)
    lines = [
        "## Dream 记忆变更",
        "",
        "以下是指定的 Dream 记忆变更。" if requested_sha else "以下是最新的 Dream 记忆变更。",
        "",
        f"- 提交：`{commit.sha}`",
        f"- 时间：{commit.timestamp}",
        f"- 变更文件：{files_line}",
    ]
    if diff:
        lines.extend([
            "",
            f"使用 `/dream-restore {commit.sha}` 撤销这次变更。",
            "",
            "```diff",
            diff.rstrip(),
            "```",
        ])
    else:
        lines.extend([
            "",
            "Dream 已记录此版本，但没有可显示的文件差异。",
        ])
    return "\n".join(lines)


def _format_dream_restore_list(commits: list[CommitInfo]) -> str:
    lines = [
        "## Dream 记忆恢复",
        "",
        "选择要恢复的 Dream 记忆版本，最新版本在前：",
        "",
    ]
    for c in commits:
        lines.append(f"- `{c.sha}` {c.timestamp} - {c.subject()}")
    lines.extend([
        "",
        "恢复前可使用 `/dream-log <sha>` 预览版本。",
        "使用 `/dream-restore <sha>` 恢复版本。",
    ])
    return "\n".join(lines)


async def cmd_dream_log(ctx: CommandContext) -> OutboundMessage:
    """Show what the last Dream changed.

    Default: diff of the latest Dream commit versus its parent.
    With /dream-log <sha>: diff of that specific commit.
    """
    store = ctx.loop.consolidator.store
    git = store.git

    if not git.is_initialized():
        if store.get_last_dream_cursor() == 0:
            msg = (
                "Dream 尚未运行。可运行 `/dream`，或等待下一次定时 Dream 任务。\n\n"
                "使用 `/dream-prompt` 查看或修改 Dream 记忆整理规则。"
            )
        else:
            msg = "记忆版本管理尚未初始化，无法查看 Dream 历史。"
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content=msg, metadata={"render_as": "text"},
        )

    args = ctx.args.strip()

    if args:
        sha = args.split()[0]
        result = git.show_commit_diff(sha)
        if not result:
            content = (
                f"找不到 Dream 变更 `{sha}`。\n\n"
                "使用 `/dream-restore` 列出近期版本，"
                "或使用 `/dream-log` 查看最新版本。"
            )
        else:
            commit, diff = result
            content = _format_dream_log_content(commit, diff, requested_sha=sha)
    else:
        commits = git.log(max_entries=1, message_prefix=_DREAM_COMMIT_PREFIX)
        result = (
            git.show_commit_diff(
                commits[0].sha,
                max_entries=1,
                message_prefix=_DREAM_COMMIT_PREFIX,
            )
            if commits else None
        )
        if result:
            commit, diff = result
            content = _format_dream_log_content(commit, diff)
        else:
            content = (
                "Dream 记忆尚无已保存的版本。\n\n"
                "使用 `/dream-prompt` 查看或修改 Dream 记忆整理规则。"
            )

    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


async def cmd_dream_restore(ctx: CommandContext) -> OutboundMessage:
    """Restore memory files from a previous dream commit.

    Usage:
        /dream-restore          — list recent commits
        /dream-restore <sha>    — revert a specific commit
    """
    store = ctx.loop.consolidator.store
    git = store.git
    if not git.is_initialized():
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="记忆版本管理尚未初始化，无法查看 Dream 历史。",
        )

    args = ctx.args.strip()
    if not args:
        commits = git.log(max_entries=10, message_prefix=_DREAM_COMMIT_PREFIX)
        if not commits:
            content = "Dream 记忆尚无可恢复的已保存版本。"
        else:
            content = _format_dream_restore_list(commits)
    else:
        sha = args.split()[0]
        result = git.show_commit_diff(sha, message_prefix=_DREAM_COMMIT_PREFIX)
        if not result:
            content = (
                f"无法恢复 Dream 变更 `{sha}`。\n\n"
                "只能恢复 Dream 记忆版本。"
                "使用 `/dream-restore` 列出近期版本。"
            )
        else:
            changed_files = _format_changed_files(result[1])
            new_sha = git.revert(sha, message_prefix=_DREAM_COMMIT_PREFIX)
            if new_sha:
                content = (
                    f"已将 Dream 记忆恢复至 `{sha}` 变更之前的状态。\n\n"
                    f"- 新建的安全提交：`{new_sha}`\n"
                    f"- 已恢复文件：{changed_files}\n\n"
                    f"使用 `/dream-log {new_sha}` 查看恢复产生的差异。"
                )
            else:
                content = (
                    f"无法恢复 Dream 变更 `{sha}`。\n\n"
                    "这可能是首个已保存版本，没有更早的状态可供恢复。"
                )
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=content, metadata={"render_as": "text"},
    )


_HISTORY_DEFAULT_COUNT = 10
_HISTORY_MAX_COUNT = 50
_HISTORY_MAX_CONTENT_CHARS = 200


def _format_history_message(msg: dict[str, Any]) -> str | None:
    role = msg.get("role")
    if role not in ("user", "assistant"):
        return None
    content = msg.get("content") or ""
    if isinstance(content, list):
        parts = [
            text
            for block in cast(list[object], content)
            if (item := cast(dict[str, Any], block) if isinstance(block, dict) else None)
            and item.get("type") == "text"
            and isinstance(text := item.get("text"), str)
        ]
        content = " ".join(parts)
    content = str(content).strip()
    if not content:
        return None
    if len(content) > _HISTORY_MAX_CONTENT_CHARS:
        content = content[:_HISTORY_MAX_CONTENT_CHARS] + "…"
    label = "👤 你" if role == "user" else "🤖 助手"
    return f"{label}: {content}"


async def cmd_history(ctx: CommandContext) -> OutboundMessage:
    """Show the last N messages of the current session (default 10, max 50).

    Usage: /history [count]
    """
    count = _HISTORY_DEFAULT_COUNT
    if ctx.args.strip():
        try:
            count = max(1, min(int(ctx.args.strip()), _HISTORY_MAX_COUNT))
        except ValueError:
            return OutboundMessage(
                channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
                content="用法：/history [count] — 例如 /history 5（默认 10 条，最多 50 条）",
                metadata=dict(ctx.msg.metadata or {}),
            )

    session = ctx.session or await ctx.loop.sessions.state.get(ctx.key)
    history = session.get_history(max_messages=0, include_runtime_context=False)
    visible = [_format_history_message(m) for m in history]
    visible = [m for m in visible if m is not None]
    recent = visible[-count:]

    if not recent:
        return OutboundMessage(
            channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
            content="暂无对话历史。",
            metadata=dict(ctx.msg.metadata or {}),
        )

    header = f"最近 {len(recent)} 条消息：\n"
    return OutboundMessage(
        channel=ctx.msg.channel, chat_id=ctx.msg.chat_id,
        content=header + "\n".join(recent),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


async def cmd_pairing(ctx: CommandContext) -> OutboundMessage:
    """List, approve, deny or revoke pairing requests."""
    from nanobot.pairing import PAIRING_COMMAND_META_KEY, handle_pairing_command
    from nanobot.pairing.events import PairingChanged

    def changed() -> None:
        ctx.loop.bus.publish_nowait(PairingChanged())

    reply = handle_pairing_command(ctx.msg.channel, ctx.args, on_change=changed)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=reply,
        metadata={PAIRING_COMMAND_META_KEY: True},
    )


async def cmd_skill(ctx: CommandContext) -> OutboundMessage:
    """List all enabled skills (name and description only)."""
    loop = ctx.loop
    skills = loop.context.skills.list_skills(filter_unavailable=False)
    if not skills:
        content = "暂无可用技能。"
    else:
        lines = [f"可用技能（{len(skills)}）：", ""]
        for entry in skills:
            desc = loop.context.skills.get_skill_description(entry["name"])
            lines.append(f"- **{entry['name']}** — {desc}")
        content = "\n".join(lines)
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=content,
        metadata=dict(ctx.msg.metadata or {}),
    )


async def cmd_trigger(ctx: CommandContext) -> OutboundMessage:
    """Create a local trigger bound to the current session."""
    name = ctx.args.strip()
    if not name:
        return OutboundMessage(
            channel=ctx.msg.channel,
            chat_id=ctx.msg.chat_id,
            content=(
                "用法：/trigger <name>\n\n"
                "创建绑定当前会话的命名本地触发器。"
            ),
            metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
        )

    from nanobot.triggers.local_store import LocalTriggerStore

    loop = ctx.loop
    store = loop.local_trigger_store
    if store is None:
        store = LocalTriggerStore(loop.workspace)

    from nanobot.session.keys import UNIFIED_SESSION_KEY

    session_key = (
        ctx.msg.session_key
        if ctx.key == UNIFIED_SESSION_KEY
        else ctx.key
    )
    trigger = store.create(
        name=name,
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        session_key=session_key,
        sender_id="trigger",
        origin_metadata=dict(ctx.msg.metadata or {}),
    )
    command = f'nanobot trigger {trigger.id} "message"'
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=(
            f"已创建触发器：{trigger.name}\n"
            f"ID: {trigger.id}\n\n"
            f"命令：\n{command}"
        ),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )

async def cmd_help(ctx: CommandContext) -> OutboundMessage:
    """Return available slash commands."""
    return OutboundMessage(
        channel=ctx.msg.channel,
        chat_id=ctx.msg.chat_id,
        content=build_help_text(ctx.loop.commands.specs),
        metadata={**dict(ctx.msg.metadata or {}), "render_as": "text"},
    )


def build_help_text(specs: tuple[BuiltinCommandSpec, ...] = CORE_COMMAND_SPECS) -> str:
    """Build help text for the commands registered in one application."""
    lines = ["🐈 nanobot 命令："]
    for spec in specs:
        command = spec.command
        if spec.arg_hint:
            command = f"{command} {spec.arg_hint}"
        lines.append(f"{command} — {spec.description}")
    return "\n".join(lines)


def register_core_commands(router: CommandRouter) -> None:
    """Register slash commands owned by the core runtime."""
    for spec in CORE_COMMAND_SPECS:
        router.add_spec(spec)
    router.priority("/stop", cmd_stop)
    router.priority("/restart", cmd_restart)
    router.priority("/status", cmd_status)
    router.exact("/new", cmd_new)
    router.exact("/compact", cmd_compact, session_queue=True)
    router.exact("/status", cmd_status)
    router.exact("/model", cmd_model)
    router.prefix("/model ", cmd_model)
    router.exact("/history", cmd_history)
    router.prefix("/history ", cmd_history)
    router.exact("/trigger", cmd_trigger)
    router.prefix("/trigger ", cmd_trigger)
    router.exact("/dream", cmd_dream)
    router.exact("/dream-log", cmd_dream_log)
    router.prefix("/dream-log ", cmd_dream_log)
    router.exact("/dream-restore", cmd_dream_restore)
    router.prefix("/dream-restore ", cmd_dream_restore)
    router.exact("/dream-prompt", cmd_dream_prompt)
    router.prefix("/dream-prompt ", cmd_dream_prompt)
    router.exact("/evaluator-prompt", cmd_evaluator_prompt)
    router.prefix("/evaluator-prompt ", cmd_evaluator_prompt)
    router.exact("/skill", cmd_skill)
    router.exact("/help", cmd_help)
    router.exact("/pairing", cmd_pairing)
    router.prefix("/pairing ", cmd_pairing)
    router.exact(USER_SHELL_COMMAND, cmd_user_shell)
    router.prefix(f"{USER_SHELL_COMMAND} ", cmd_user_shell)
