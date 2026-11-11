from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from loguru import logger

from nanobot.utils.atomic_file import atomic_write_text

if TYPE_CHECKING:
    from dulwich.objects import Blob, Commit, ObjectID, Tree
    from dulwich.refs import Ref
    from dulwich.repo import Repo

# Cap on the unified-diff block embedded in Dream commit messages. Memory files
# are tiny in practice, but a pathological rewrite must not blow up the audit
# record. The structured per-file summary is always emitted in full regardless.
_WORKING_TREE_DIFF_MAX_CHARS = 6000
_MEMORY_REPO_MARKER = ".git/nanobot-memory-store"


class GitStoreError(RuntimeError):
    """Raised when the memory Git repository cannot complete an operation."""


@dataclass
class CommitInfo:
    sha: str  # Short SHA (8 chars)
    message: str
    timestamp: str

    def subject(self) -> str:
        """First line of the commit message, or a placeholder if empty."""
        lines = self.message.splitlines()
        return lines[0] if lines else "(no message)"

    def format(self, diff: str = "") -> str:
        """Format this commit for display, optionally with a diff."""
        header = f"## {self.subject()}\n`{self.sha}` — {self.timestamp}\n"
        if diff:
            return f"{header}\n```diff\n{diff}\n```"
        return f"{header}\n(no file changes)"


class GitStore:
    """Git-backed version control for memory files."""

    def __init__(
        self,
        workspace: Path,
        tracked_files: list[str],
        tracked_dirs: list[str] | None = None,
        restore_excluded: list[str] | None = None,
    ):
        self._workspace = workspace
        self._tracked_files = tracked_files
        # Directories whose ``*.md`` files are tracked as they appear, such as
        # dated memory notes.
        self._tracked_dirs = list(tracked_dirs or [])
        self._restore_excluded = set(restore_excluded or [])
        self._ownership_cache: bool | None = None

    def is_initialized(self) -> bool:
        """Check whether Nanobot owns the Git repo at this workspace root."""
        if self._ownership_cache is None:
            marker = self._workspace / _MEMORY_REPO_MARKER
            self._ownership_cache = marker.is_file() or self._is_legacy_memory_repo()
        return self._ownership_cache


    def init(self) -> bool:
        """Initialize a git repo if not already initialized.

        Creates .gitignore and makes an initial commit.
        Returns True if a new repo was created, False if already exists.
        """
        if self.is_initialized():
            self._update_gitignore()
            return False

        if (self._workspace / ".git").is_dir():
            return False

        if self._is_inside_git_repo():
            logger.warning(
                "Workspace {} is already inside a git repo; "
                "skipping nested repo initialization",
                self._workspace,
            )
            return False

        try:
            from dulwich import porcelain

            porcelain.init(str(self._workspace))

            self._update_gitignore()

            for rel in self._tracked_files:
                p = self._workspace / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    atomic_write_text(p, "")

            porcelain.add(
                str(self._workspace),
                paths=self._staging_paths(".gitignore", *self.tracked_paths()),
            )
            porcelain.commit(
                str(self._workspace),
                message=b"init: nanobot memory store",
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            atomic_write_text(self._workspace / _MEMORY_REPO_MARKER, "nanobot memory store\n")
            self._ownership_cache = True
            logger.debug("Git store initialized at {}", self._workspace)
            return True
        except Exception as exc:
            raise GitStoreError(f"Git store init failed for {self._workspace}") from exc


    def auto_commit(self, message: str) -> str | None:
        """Stage tracked memory files and commit if there are changes.

        Returns the short commit SHA, or None if nothing to commit.
        """
        if not self.is_initialized():
            return None

        try:
            from dulwich import porcelain

            # Stage first so Dulwich refreshes the content hashes.  A status
            # check can miss rapid same-size rewrites when the filesystem also
            # preserves the file's mtime.
            # Paths present only in HEAD stage the deletion of a tracked-dir file.
            porcelain.add(
                str(self._workspace),
                paths=self._staging_paths(".gitignore", *self.tracked_paths()),
            )
            st = porcelain.status(str(self._workspace))
            staged = cast(dict[object, list[object]], st.staged)
            if not any(staged.values()):
                return None

            message_value = cast(object, message)
            msg_bytes = (
                message_value.encode("utf-8")
                if isinstance(message_value, str)
                else cast(bytes, message_value)
            )
            sha_bytes = porcelain.commit(
                str(self._workspace),
                message=msg_bytes,
                author=b"nanobot <nanobot@dream>",
                committer=b"nanobot <nanobot@dream>",
            )
            if cast(object, sha_bytes) is None:
                return None
            # porcelain.commit returns the id as a 40-char hex string that is
            # already encoded to bytes; .hex() would encode those ASCII bytes
            # again and produce an id no git command can resolve.
            sha = sha_bytes.decode()[:8]
            logger.debug("Git auto-commit: {} ({})", sha, message)
            return sha
        except Exception as exc:
            raise GitStoreError(f"Git auto-commit failed: {message}") from exc


    def _working_dir_files(self, directory: str) -> list[str]:
        root = self._workspace / directory
        if not root.is_dir():
            return []
        return sorted(
            path.relative_to(self._workspace).as_posix()
            for path in root.glob("*.md")
            if path.is_file()
        )

    def _tree_dir_files(self, repo: "Repo", tree: "Tree | None", directory: str) -> list[str]:
        current = tree
        for part in Path(directory).parts:
            if current is None:
                return []
            try:
                entry = current[part.encode()]
            except KeyError:
                return []
            obj = repo[entry[1]]
            current = cast("Tree", obj) if obj.type_name == b"tree" else None
        if current is None:
            return []
        names: list[str] = []
        for item in current.items():
            name = item.path.decode("utf-8", errors="replace")
            if name.endswith(".md") and repo[item.sha].type_name == b"blob":
                names.append(f"{Path(directory).as_posix()}/{name}")
        return sorted(names)

    def tracked_paths(self) -> list[str]:
        """Tracked files plus every ``*.md`` file in tracked directories (working tree or HEAD)."""
        paths = list(self._tracked_files)
        if not self._tracked_dirs:
            return paths
        head_files: list[str] = []
        if self.is_initialized():
            from dulwich.repo import Repo

            with Repo(str(self._workspace)) as repo:
                head_tree = self._head_tree(repo)
                for directory in self._tracked_dirs:
                    head_files.extend(self._tree_dir_files(repo, head_tree, directory))
        for directory in self._tracked_dirs:
            for path in sorted({*self._working_dir_files(directory), *(
                f for f in head_files if f.startswith(f"{Path(directory).as_posix()}/")
            )}):
                if path not in paths:
                    paths.append(path)
        return paths

    def _staging_paths(self, *paths: str) -> list[str]:
        return [str((self._workspace / path).absolute()) for path in paths]

    def _resolve_sha(self, short_sha: str) -> bytes | None:
        try:
            from dulwich.repo import Repo

            with Repo(str(self._workspace)) as repo:
                try:
                    sha: ObjectID | None = repo.refs[cast("Ref", b"HEAD")]
                except KeyError:
                    return None

                while sha:
                    if sha.decode().startswith(short_sha):
                        return sha
                    commit_obj = repo[sha]
                    if commit_obj.type_name != b"commit":
                        break
                    commit = cast("Commit", commit_obj)
                    sha = commit.parents[0] if commit.parents else None
            return None
        except Exception as exc:
            raise GitStoreError(f"Git SHA resolution failed: {short_sha}") from exc

    def _is_inside_git_repo(self) -> bool:
        """Check if self._workspace is already inside a git repository.

        Walks up from self._workspace to the filesystem root, returning True
        if any parent directory contains a .git entry.

        Git worktrees and submodules can use a ``.git`` file instead of a
        directory, so we must treat either form as "already inside a repo".
        """
        current = self._workspace.resolve()
        while current != current.parent:
            if (current / ".git").exists():
                return True
            current = current.parent
        return False

    def _is_legacy_memory_repo(self) -> bool:
        if not (self._workspace / ".git").is_dir():
            return False
        try:
            from dulwich.repo import Repo

            with Repo(str(self._workspace)) as repo:
                sha = repo.refs[cast("Ref", b"HEAD")]
                while sha:
                    commit = repo[sha]
                    if commit.type_name != b"commit":
                        return False
                    value = cast("Commit", commit)
                    if value.parents:
                        sha = value.parents[0]
                    else:
                        if (
                            value.message.strip() == b"init: nanobot memory store"
                            and value.author == b"nanobot <nanobot@dream>"
                        ):
                            root = cast("Commit", repo[sha])
                            tree = cast("Tree", repo[root.tree])
                            gitignore = self._read_blob_from_tree(repo, tree, ".gitignore")
                            return gitignore is not None and gitignore.startswith("/*\n")
                        return False
            return False
        except Exception:
            return False

    def _update_gitignore(self) -> None:
        path = self._workspace / ".gitignore"
        existing = path.read_text(encoding="utf-8") if path.exists() else ""
        existing_lines = set(existing.splitlines())
        additions = [
            line for line in self._build_gitignore().splitlines() if line not in existing_lines
        ]
        if additions:
            prefix = existing.rstrip("\n") + "\n" if existing else ""
            atomic_write_text(path, prefix + "\n".join(additions) + "\n")

    def _build_gitignore(self) -> str:
        dirs: set[str] = set()
        for f in self._tracked_files:
            parent = str(Path(f).parent)
            if parent != ".":
                dirs.add(parent)
        for d in self._tracked_dirs:
            parts = Path(d).parts
            for depth in range(1, len(parts) + 1):
                dirs.add(Path(*parts[:depth]).as_posix())
        lines = ["/*"]
        for d in sorted(dirs):
            lines.append(f"!{d}/")
        for f in self._tracked_files:
            lines.append(f"!{f}")
        lines.extend(["!.gitignore", "*.sqlite", "*.sqlite3", "*-wal", "*-shm"])
        return "\n".join(lines) + "\n"


    def log(
        self,
        max_entries: int = 20,
        message_prefix: str | None = None,
    ) -> list[CommitInfo]:
        """Return simplified commit log, optionally filtered by message prefix.

        When filtering, *max_entries* counts matching commits rather than every
        commit traversed in the repository.
        """
        if not self.is_initialized():
            return []

        try:
            from dulwich.repo import Repo

            entries: list[CommitInfo] = []
            with Repo(str(self._workspace)) as repo:
                try:
                    head = repo.refs[cast("Ref", b"HEAD")]
                except KeyError:
                    return []

                sha: ObjectID | None = head
                while sha and len(entries) < max_entries:
                    commit_obj = repo[sha]
                    if commit_obj.type_name != b"commit":
                        break
                    commit = cast("Commit", commit_obj)
                    ts = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(commit.commit_time),
                    )
                    msg = commit.message.decode("utf-8", errors="replace").strip()
                    if message_prefix is None or msg.startswith(message_prefix):
                        entries.append(CommitInfo(
                            sha=sha.decode()[:8],
                            message=msg,
                            timestamp=ts,
                        ))
                    sha = commit.parents[0] if commit.parents else None

            return entries
        except Exception as exc:
            raise GitStoreError("Git log failed") from exc

    def diff_commits(self, sha1: str, sha2: str) -> str:
        """Show diff between two commits."""
        if not self.is_initialized():
            return ""

        try:
            from dulwich import porcelain

            full1 = self._resolve_sha(sha1)
            full2 = self._resolve_sha(sha2)
            if not full1 or not full2:
                return ""

            out = io.BytesIO()
            porcelain.diff(
                str(self._workspace),
                commit=full1,
                commit2=full2,
                outstream=out,
            )
            return out.getvalue().decode("utf-8", errors="replace")
        except Exception as exc:
            raise GitStoreError(f"Git diff failed for {sha1}..{sha2}") from exc

    def summarize_working_tree(self, paths: list[str]) -> str:
        """Structured summary of working-tree changes vs HEAD for *paths*.

        Pure filesystem/git ground truth — never LLM narrative — suitable as a
        truthful audit record. Returns "" when the repo is not initialized or
        none of *paths* differ from HEAD.

        Format::

            SOUL.md: +3 -1
            memory/MEMORY.md: +12 -8

            2 files changed, 15 insertions(+), 9 deletions(-)

            ```diff
            --- SOUL.md
            +++ SOUL.md
            @@ ...
            - old
            + new
            ```
        """
        if not self.is_initialized():
            return ""

        try:
            import difflib

            from dulwich.repo import Repo
        except ImportError as exc:
            raise GitStoreError("Git working-tree summary dependencies are unavailable") from exc

        summary_lines: list[str] = []
        diff_lines: list[str] = []
        total_added = 0
        total_removed = 0
        changed = 0

        try:
            with Repo(str(self._workspace)) as repo:
                head_tree = self._head_tree(repo)
                for path in paths:
                    head_text = (
                        self._read_blob_from_tree(repo, head_tree, path)
                        if head_tree is not None
                        else None
                    )
                    if head_text is None:
                        head_text = ""
                    wt_path = self._workspace / path
                    try:
                        wt_text = (
                            wt_path.read_bytes().decode("utf-8")
                            if wt_path.exists()
                            else ""
                        )
                    except UnicodeDecodeError:
                        # Non-UTF-8 (binary/corrupt) working-tree file: record
                        # the change without a unified diff, which would
                        # otherwise be polluted with replacement characters and
                        # misrepresent the audit record.
                        changed += 1
                        summary_lines.append(f"{path}: binary or non-UTF-8 file changed")
                        continue
                    # Treat CRLF and LF as equivalent without hiding other
                    # newline changes, such as a missing final newline.
                    if head_text.replace("\r\n", "\n") == wt_text.replace("\r\n", "\n"):
                        continue
                    head_lines = head_text.splitlines()
                    wt_lines = wt_text.splitlines()
                    changed += 1
                    hunks = list(difflib.unified_diff(
                        head_lines,
                        wt_lines,
                        fromfile=path,
                        tofile=path,
                        lineterm="",
                    ))
                    added = sum(1 for line in hunks if line.startswith("+") and not line.startswith("+++"))
                    removed = sum(1 for line in hunks if line.startswith("-") and not line.startswith("---"))
                    total_added += added
                    total_removed += removed
                    summary_lines.append(f"{path}: +{added} -{removed}")
                    diff_lines.extend(hunks)
        except Exception as exc:
            raise GitStoreError("Git working-tree summary failed") from exc

        if changed == 0:
            return ""

        diff_text = "\n".join(diff_lines)
        if len(diff_text) > _WORKING_TREE_DIFF_MAX_CHARS:
            diff_text = diff_text[:_WORKING_TREE_DIFF_MAX_CHARS] + "\n...[diff truncated]"

        body = "\n".join(summary_lines)
        body += (
            f"\n{changed} file{'s' if changed != 1 else ''} changed, "
            f"{total_added} insertion{'s' if total_added != 1 else ''}(+), "
            f"{total_removed} deletion{'s' if total_removed != 1 else ''}(-)"
        )
        if diff_lines:
            body += f"\n\n```diff\n{diff_text}\n```"
        return body

    @staticmethod
    def _head_tree(repo: "Repo") -> "Tree | None":
        """Return the tree object at HEAD, or None if there are no commits."""
        try:
            head = repo.refs[cast("Ref", b"HEAD")]
        except KeyError:
            return None
        commit_obj = repo[head]
        if commit_obj.type_name != b"commit":
            return None
        commit = cast("Commit", commit_obj)
        return cast("Tree", repo[commit.tree])

    def show_commit_diff(
        self,
        short_sha: str,
        max_entries: int = 20,
        message_prefix: str | None = None,
    ) -> tuple[CommitInfo, str] | None:
        """Find a commit and return it with its diff vs its actual parent."""
        try:
            from dulwich.repo import Repo

            commits = self.log(max_entries=max_entries, message_prefix=message_prefix)
            for c in commits:
                if c.sha.startswith(short_sha):
                    full_sha = self._resolve_sha(c.sha)
                    if not full_sha:
                        return None
                    with Repo(str(self._workspace)) as repo:
                        commit = cast("Commit", repo[full_sha])
                        parent = commit.parents[0] if commit.parents else None
                    diff = self.diff_commits(parent.decode()[:8], c.sha) if parent else ""
                    return c, diff
            return None
        except Exception as exc:
            raise GitStoreError(f"Git commit display failed for {short_sha}") from exc


    def revert(self, commit: str, *, message_prefix: str | None = None) -> str | None:
        """Restore the selected parent snapshot and commit it."""
        if not self.restore_snapshot(commit, message_prefix=message_prefix):
            return None
        return self.auto_commit(f"revert: undo {commit}")

    def restore_snapshot(self, commit: str, *, message_prefix: str | None = None) -> bool:
        """Restore the complete managed set to the selected commit's parent, without committing.

        The caller reconciles permanent tombstones before publishing the restored state.
        Files absent from that snapshot are removed, including files added by later commits.
        """
        if not self.is_initialized():
            return False

        try:
            from dulwich.repo import Repo

            full_sha = self._resolve_sha(commit)
            if not full_sha:
                logger.warning("Git revert: SHA not found: {}", commit)
                return False

            with Repo(str(self._workspace)) as repo:
                commit_obj = repo[full_sha]
                if commit_obj.type_name != b"commit":
                    return False
                typed_commit = cast("Commit", commit_obj)

                commit_message = typed_commit.message.decode(
                    "utf-8",
                    errors="replace",
                ).strip()
                if message_prefix is not None and not commit_message.startswith(message_prefix):
                    logger.warning(
                        "Git revert: commit {} does not match message prefix {!r}",
                        commit,
                        message_prefix,
                    )
                    return False

                if not typed_commit.parents:
                    logger.warning("Git revert: cannot revert root commit {}", commit)
                    return False

                parent_obj = cast("Commit", repo[typed_commit.parents[0]])
                tree = cast("Tree", repo[parent_obj.tree])
                current_paths = set(self.tracked_paths())
                restore_paths = set(self._tracked_files)
                for directory in self._tracked_dirs:
                    restore_paths.update(self._tree_dir_files(repo, tree, directory))
                selected = (current_paths | restore_paths) - self._restore_excluded
                for filepath in sorted(selected):
                    content = self._read_blob_from_tree(repo, tree, filepath)
                    dest = self._workspace / filepath
                    if content is not None:
                        atomic_write_text(dest, content)
                    elif dest.exists():
                        dest.unlink()
            return True
        except Exception as exc:
            raise GitStoreError(f"Git revert failed for {commit}") from exc

    @staticmethod
    def _read_blob_from_tree(
        repo: "Repo",
        tree: "Tree",
        filepath: str,
    ) -> str | None:
        """Read a blob's content from a tree object by walking path parts."""
        parts = Path(filepath).parts
        current = tree
        for part in parts:
            try:
                entry = current[part.encode()]
            except KeyError:
                return None
            obj = repo[entry[1]]
            if obj.type_name == b"blob":
                blob = cast("Blob", obj)
                return blob.data.decode("utf-8", errors="replace")
            if obj.type_name == b"tree":
                current = cast("Tree", obj)
            else:
                return None
        return None
