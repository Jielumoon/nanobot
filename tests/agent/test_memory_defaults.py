from datetime import datetime, timezone

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_documents import ClaimDraft
from nanobot.config.schema import MemoryConfig, MemoryEmbeddingConfig, MemoryPeopleConfig

NOW = datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc)


def test_default_memory_reads_old_workspace_without_rewriting_and_appends_on_write(tmp_path):
    user = tmp_path / "USER.md"
    memory = tmp_path / "memory/MEMORY.md"
    memory.parent.mkdir()
    user.write_text("# 我的偏好\n\n- 保留手写格式  和空格\n")
    memory.write_text("# 旧记忆\n\n## 手写章节\n项目在上海\n")
    original = {path: path.read_bytes() for path in (user, memory)}
    builder = ContextBuilder(tmp_path, timezone="Asia/Shanghai", clock=lambda: NOW)
    store = builder.memory
    assert store.claims is not None and store.daily is not None
    assert store.config.people.enabled and store.embeddings is None
    assert "项目在上海" in builder.build_system_prompt()
    assert store.claims.search("上海")
    assert {path: path.read_bytes() for path in original} == original
    store.claims.remember(ClaimDraft(text="另一个已确认的事实", kind="fact", core=True))
    assert memory.read_bytes().startswith(original[memory])
    assert "## 已记住的事实" in memory.read_text()
    assert user.read_bytes() == original[user]
    store.append_daily_summary("- 今天完成预演")
    assert "今天完成预演" in store.daily.path_for().read_text()


def test_explicit_disable_retains_legacy_memory_and_makes_no_projection(tmp_path):
    path = tmp_path / "memory/MEMORY.md"
    path.parent.mkdir()
    text = "# 旧记忆\n" + "已有笔记\n" * 1500 + "末尾原文"
    path.write_text(text)
    config = MemoryConfig(
        claims_enabled=False, daily_notes=False, people=MemoryPeopleConfig(enabled=False),
        embedding=MemoryEmbeddingConfig(enabled=True),
    )
    builder = ContextBuilder(tmp_path, memory_config=config)
    assert builder.memory.claims is None and builder.memory.daily is None
    assert builder.memory.embeddings is None
    assert "末尾原文" in builder.build_system_prompt()
    builder.memory.append_daily_summary("不应镜像的摘要")
    assert not (tmp_path / "memory/daily").exists()
    assert not (tmp_path / "memory/claims.sqlite3").exists()
    assert path.read_text() == text


def test_default_core_budget_truncates_prompt_without_changing_file(tmp_path):
    store = MemoryStore(tmp_path)
    content = "# Memory\n" + "- 已有记忆\n" * 1500 + "- 最后一个已确认的事实\n"
    store.write_memory(content)
    prompt = store.prompt_memory(content)
    assert "MEMORY.md truncated" in prompt and "最后一个已确认的事实" not in prompt
    assert store.read_memory() == content
    assert store.claims.search("最后一个已确认的事实")
