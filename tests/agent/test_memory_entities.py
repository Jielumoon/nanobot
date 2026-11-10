"""User paths through entity pages, durable forgetting, and dynamic context."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_claims import ClaimStore
from nanobot.agent.memory_documents import ClaimDraft, ClaimKind, decode_entity, encode_entity
from nanobot.agent.memory_recall import MEMORY_CONTEXT_CHARS, MemoryRuntimeContext
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.memory_claims import (
    MemoryExplainTool,
    MemoryForgetTool,
    MemoryProjectTool,
    MemoryRememberTool,
)
from nanobot.config.schema import MemoryConfig, MemoryPeopleConfig

NOW = datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        tmp_path, clock=lambda: NOW, timezone="Asia/Shanghai",
        config=MemoryConfig(claims_enabled=True, people=MemoryPeopleConfig(enabled=True)),
    )


@pytest.fixture
def claims(store: MemoryStore) -> ClaimStore:
    assert store.claims is not None
    return store.claims


@pytest.mark.parametrize(("kind", "path"), [
    ("preference", "bank/preferences.md"), ("experience", "bank/experiences.md"),
    ("circumstance", "bank/circumstances.md"), ("fact", "bank/facts.md"),
    ("decision", "bank/facts.md"), ("commitment", "bank/facts.md"),
    ("progress", "daily/2030-05-04.md"),
])
def test_memory_kind_routes_without_daily_mirroring(
    claims: ClaimStore, kind: ClaimKind, path: str,
) -> None:
    saved = claims.remember(ClaimDraft(text="记住这条信息", kind=kind)).claim
    assert saved.source_file == f"memory/{path}"
    claims.db_path.unlink()
    assert claims.get(saved.claim_id) == saved


async def test_memory_project_create_rename_complete_and_manual_reactivate(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    remembered = claims.remember(ClaimDraft(
        text="数据库迁移已完成预演", kind="progress", project="数据库/迁移",
    )).claim
    path = store.workspace / str(remembered.source_file)
    text = path.read_text()
    assert path.name == "数据库迁移.md"
    assert text.startswith("# 数据库/迁移\n")
    assert all(heading in text for heading in ("## 进度", "## 决定", "## 待办"))
    entity = decode_entity(text, issues=set())
    assert entity is not None and entity.entity_id == remembered.project_id
    provider = MemoryRuntimeContext(store, claims)
    block = provider.build("")
    assert block is not None and remembered.text in block.content

    result = json.loads(await MemoryProjectTool(store).execute(
        project="数据库/迁移", status="completed", rename="迁移二期",
    ))
    assert result["project"]["entity_id"] == entity.entity_id
    renamed = store.workspace / "memory/projects/迁移二期.md"
    assert not path.exists() and renamed.exists()
    assert provider.build("数据库迁移预演") is None
    assert claims.search("预演")[0].claim_id == remembered.claim_id
    assert any(event.target_id == entity.entity_id and event.action == "update"
               for event in claims.documents.events())

    renamed.write_text(renamed.read_text().replace('"status":"completed"', '"status":"active"'))
    claims.db_path.unlink()
    block = provider.build("")
    assert block is not None and remembered.text in block.content
    followup = claims.remember(ClaimDraft(
        text="迁移将在周五交付", kind="commitment", project="数据库/迁移", due_at="2030-05-02",
    )).claim
    assert followup.project_id == entity.entity_id
    assert followup.source_file == "memory/projects/迁移二期.md"
    await MemoryProjectTool(store).execute(project=entity.entity_id, status="paused")
    assert provider.build("迁移周五交付") is None


def test_memory_project_slug_collisions_keep_distinct_ids(claims: ClaimStore) -> None:
    first = claims.remember(ClaimDraft(text="第一项工作", kind="progress", project="研发/A")).claim
    second = claims.remember(ClaimDraft(text="第二项工作", kind="progress", project="研发:A")).claim
    assert first.project_id != second.project_id
    assert first.source_file == "memory/projects/研发a.md"
    assert second.source_file == "memory/projects/研发a-2.md"


async def test_memory_people_require_enabled_config_and_evidence(
    tmp_path: Path, store: MemoryStore, claims: ClaimStore,
) -> None:
    disabled = MemoryStore(tmp_path / "disabled", config=MemoryConfig(
        claims_enabled=True, people=MemoryPeopleConfig(enabled=False),
    ))
    result = await MemoryRememberTool(disabled).execute(
        text="小林负责设计", kind="fact", person="小林", quote="小林负责设计",
    )
    assert "人物记忆未开启" in result
    tool = MemoryRememberTool(store)
    result = await tool.execute(text="小林负责设计", kind="fact", person="小林")
    assert "只记录有证据" in result
    assert not (store.memory_dir / "people").exists()
    with request_context(RequestContext(
        channel="cli", chat_id="direct", session_key="cli:direct", message_id="m-123",
    )):
        saved = json.loads(await tool.execute(
            text="小林负责设计", kind="fact", person="小林", aliases=["Lin"],
        ))["claim"]
    claim = claims.get(saved["claim_id"])
    assert claim is not None
    assert claim.evidence.source_ref == "session:cli:direct#m-123"
    assert claim.source_file == "memory/people/小林.md"
    linked = claims.remember(ClaimDraft(
        text="小林负责上线方案", kind="decision", person="Lin", project="上线",
        quote="上线方案由小林负责",
    )).claim
    assert linked.person_id == claim.person_id and linked.project_id is not None
    assert linked.source_file == "memory/projects/上线.md"
    assert "小林负责上线方案" not in (store.workspace / str(claim.source_file)).read_text()
    provider = MemoryRuntimeContext(store, claims)
    assert provider.build("设计方案") is None
    block = provider.build("和 Lin 聊聊")
    assert block is not None and linked.text in block.content and claim.text in block.content
    store.config.people.enabled = False
    assert provider.build("小林 Lin 上线方案") is None


def test_memory_people_mentions_are_unambiguous_and_use_name_boundaries(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    for name, aliases, text in (
        ("Ann", ("安妮",), "偏好陶瓷茶具"), ("Anne", ("安妮",), "偏好木制茶具"),
        ("李", (), "喜欢雪山旅行"), ("张明", ("小明", "张"), "喜欢沙漠旅行"),
    ):
        claims.remember(ClaimDraft(
            text=text, kind="fact", person=name, aliases=aliases, quote=text,
        ))
    provider = MemoryRuntimeContext(store, claims)
    assert provider.build("annual 安妮 李 张") is None
    block = provider.build("Ask Ann, 然后找小明")
    assert block is not None and "陶瓷茶具" in block.content and "沙漠旅行" in block.content
    assert "木制茶具" not in block.content
    # A hand-edited duplicate display name is ambiguous even though IDs differ.
    path = store.workspace / "memory/people/anne.md"
    content = path.read_text()
    entity = decode_entity(content, issues=set())
    assert entity is not None
    path.write_text(content.replace("# Anne\n", "# Ann\n", 1).replace(
        encode_entity(entity), encode_entity(entity.model_copy(update={"display_name": "Ann"})),
    ))
    assert provider.build("Ann") is None
    with pytest.raises(ValueError, match="歧义"):
        claims.remember(ClaimDraft(text="新信息", kind="fact", person="Ann", quote="新信息"))


@pytest.mark.parametrize("selector", ["person", "project"])
@pytest.mark.parametrize("restore", ["manual", "git"])
async def test_memory_entity_forgetting_cleans_links_history_and_restores(
    store: MemoryStore, claims: ClaimStore, selector: str, restore: str,
) -> None:
    if restore == "git":
        store.git.init()
    old = claims.remember(ClaimDraft(
        text="小林负责邮件系统", kind="decision", project="邮件改造", person="小林",
        quote="邮件系统让小林做",
    )).claim
    new = claims.remember(ClaimDraft(
        text="小林负责通知系统", kind="decision", project="邮件改造", person="小林",
        quote="通知系统让小林做", supersedes=(old.claim_id,),
    )).claim
    store.append_history("- 小林负责邮件系统\n- 其他无关记录")
    archived = claims.get(old.claim_id)
    assert archived is not None and archived.source_file != new.source_file
    snapshot = claims.documents.snapshot()
    if restore == "git":
        store.git.auto_commit("dream: 保存实体记忆")
    kwargs = {selector: "小林" if selector == "person" else "邮件改造"}
    tool = MemoryForgetTool(store)
    candidates = json.loads(await tool.execute(**kwargs))
    assert candidates["status"] == "ambiguous" and len(candidates["candidates"]) == 2
    assert claims.get(new.claim_id) == new
    result = json.loads(await tool.execute(**kwargs, all_matches=True))
    assert set(result["retracted"]) == {old.claim_id, new.claim_id}
    deleted = "memory/people/小林.md" if selector == "person" else "memory/projects/邮件改造.md"
    assert not (store.workspace / deleted).exists()
    forgotten_entity = new.person_id if selector == "person" else new.project_id
    assert ("entity", forgotten_entity) in claims.documents.tombstones()
    assert store.read_unprocessed_history(0)[0]["content"] == "[已遗忘]\n- 其他无关记录"
    for claim_id in (old.claim_id, new.claim_id):
        explanation = claims.explain(claim_id)
        assert explanation == {"status": "已遗忘", "forgotten_at": "2030-05-03T16:30:00Z"}

    if restore == "git":
        forgotten_commit = store.git.auto_commit("dream: 遗忘实体记忆")
        assert forgotten_commit is not None
        ledger = claims.documents.ledger.read_text()
        assert store.git.revert(forgotten_commit) is not None
        assert claims.documents.ledger.read_text() == ledger
    else:
        claims.documents.apply(snapshot)
    assert (store.workspace / deleted).exists()
    claims.db_path.unlink()
    assert claims.search("小林") == []
    assert MemoryRuntimeContext(store, claims).build("小林 邮件改造") is None
    assert claims.documents.snapshot() == snapshot  # sync never rewrites restored source files.
    claims.reconcile_files()
    assert not (store.workspace / deleted).exists()
    with sqlite3.connect(claims.db_path) as conn:
        assert all(row == ("", None) for row in conn.execute(
            "SELECT text, json_extract(metadata, '$.evidence.quote') FROM claims",
        ))


async def test_memory_explain_traverses_both_directions_with_evidence(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    a = claims.remember(ClaimDraft(text="旧方案甲", kind="decision", quote="先用甲方案")).claim
    b = claims.remember(ClaimDraft(text="旧方案乙", kind="decision")).claim
    c = claims.remember(ClaimDraft(
        text="合并方案", kind="decision", supersedes=(a.claim_id, b.claim_id), project="计划",
    ), session_key="cli:direct", message_id="m-1", source_ref="session:cli:direct#m-1").claim
    d = claims.remember(ClaimDraft(
        text="并行方案", kind="decision", supersedes=(a.claim_id, b.claim_id),
    )).claim
    e = claims.remember(ClaimDraft(
        text="最终方案", kind="decision", supersedes=(c.claim_id,),
    )).claim
    result = json.loads(await MemoryExplainTool(store).execute(claim_id=c.claim_id))
    nodes = {node["claim_id"]: node for node in result["claims"]}
    assert set(nodes) == {claim.claim_id for claim in (a, b, c, d, e)}
    assert nodes[c.claim_id]["evidence"]["message_id"] == "m-1"
    assert nodes[c.claim_id]["entities"][0]["display_name"] == "计划"
    assert nodes[a.claim_id]["quote"] == "先用甲方案"
    assert {event["action"] for event in nodes[c.claim_id]["events"]} == {"remember", "supersede"}
    bounded = claims.explain(c.claim_id, limit=2)
    assert bounded["truncated"] is True
    assert len(bounded["claims"]) == 2
    by_query = json.loads(await MemoryExplainTool(store).execute(query="最终方案"))
    assert by_query["claim_id"] == e.claim_id


@pytest.mark.parametrize("use_fts", [True, False])
def test_memory_recall_filters_before_limit_and_accepts_meaningful_chinese_bigrams(
    store: MemoryStore, claims: ClaimStore, use_fts: bool,
) -> None:
    claims.index.want_fts = use_fts
    store.config.recall_limit = 1
    for i in range(6):
        claims.remember(ClaimDraft(
            text=f"咖啡知识归档 {i}", kind="decision", project=f"旧项目{i}",
        ))
        claims.update_project(f"旧项目{i}", "completed")
    saved = claims.remember(ClaimDraft(text="喜欢手冲咖啡", kind="preference")).claim
    store.write_memory("# Memory\n- 稳定事实\n")
    provider = MemoryRuntimeContext(store, claims)
    block = provider.build("咖啡")
    assert block is not None and saved.claim_id in block.content
    assert "归档" not in block.content
    claims.remember(ClaimDraft(text="我们喜欢登山", kind="fact"))
    assert provider.build("我们可以帮我一下吗") is None


def test_memory_context_budget_dedup_and_three_recent_projects(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    for i in range(5):
        claims.clock = lambda i=i: NOW + timedelta(days=i)
        claims.remember(ClaimDraft(text=f"第{i}项工程已预演", kind="progress", project=f"工程{i}"))
    provider = MemoryRuntimeContext(store, claims)
    block = provider.build("")
    assert block is not None
    assert all(f"第{i}项" in block.content for i in (2, 3, 4))
    assert all(f"第{i}项" not in block.content for i in (0, 1))
    for i in range(12):
        claims.remember(ClaimDraft(
            text=f"承诺{i} " + "仍需处理的事项" * 40, kind="commitment", project="工程4",
        ))
        claims.remember(ClaimDraft(
            text=f"人物{i} " + "偏好长距离徒步" * 40, kind="fact",
            person=f"同事{i}", quote="我喜欢徒步",
        ))
    block = provider.build("承诺 人物 " + " ".join(f"同事{i}" for i in range(12)))
    assert block is not None and len(block.content) <= MEMORY_CONTEXT_CHARS
    assert block.content.count("[c_") == len(set(
        line.split("[c_")[1].split("]")[0]
        for line in block.content.splitlines() if "[c_" in line
    ))


def test_memory_system_prompt_stays_stable_across_queries_and_due_dates(tmp_path: Path) -> None:
    builder = ContextBuilder(
        tmp_path, clock=lambda: NOW, memory_config=MemoryConfig(claims_enabled=True),
    )
    claims = builder.memory.claims
    assert claims is not None
    claims.remember(ClaimDraft(text="住在北京", kind="fact", core=True))
    claims.remember(ClaimDraft(text="完成报告", kind="commitment", due_at="2030-05-04"))
    claims.remember(ClaimDraft(text="喜欢咖啡", kind="preference"))
    prompt = builder.build_system_prompt()
    provider = MemoryRuntimeContext(builder.memory, claims)
    first = provider.build("咖啡")
    second = provider.build("报告")
    assert first is not None and second is not None and first.content != second.content
    builder.memory.clock = lambda: NOW + timedelta(days=5)
    overdue = provider.build("报告")
    assert overdue is not None and "已逾期" in overdue.content
    assert builder.build_system_prompt() == prompt
    assert "完成报告" not in prompt


def test_memory_entity_tombstone_precedes_failed_cleanup(
    store: MemoryStore, claims: ClaimStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    saved = claims.remember(ClaimDraft(
        text="小林喜欢帆船", kind="fact", person="小林", quote="我喜欢帆船",
    )).claim
    page = store.workspace / str(saved.source_file)
    unlink = Path.unlink

    def fail_cleanup(path: Path, missing_ok: bool = False) -> None:
        if path == page:
            raise OSError("disk failure")
        unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_cleanup)
        with pytest.raises(OSError, match="disk failure"):
            store.forget_memory(person="小林", all_matches=True)
    assert page.exists()
    claims.db_path.unlink()
    assert claims.search("帆船") == []
    assert claims.explain(saved.claim_id)["status"] == "已遗忘"
    with claims.index.connect() as conn:
        assert conn.execute(
            "SELECT entity_json FROM documents WHERE path=?", (saved.source_file,),
        ).fetchone()[0] is None
    store.reconcile_files()
    assert not page.exists()


def test_memory_people_budget_and_latest_project_progress(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    for i in range(6):
        claims.remember(ClaimDraft(text=f"项目进度{i}", kind="progress", project="新版"))
        for person in ("张明", "李华", "王林"):
            claims.remember(ClaimDraft(
                text=f"{person} 信息{i} " + "偏好徒步" * 50,
                kind="fact", person=person, quote="偏好徒步",
            ))
    block = MemoryRuntimeContext(store, claims).build("张明 李华 王林")
    assert block is not None
    assert all(f"项目进度{i}" in block.content for i in (3, 4, 5))
    assert "项目进度2" not in block.content
    person_lines = [line for line in block.content.splitlines() if line.startswith("- Person ")]
    assert {line.split()[2] for line in person_lines} == {"张明", "李华", "王林"}
    assert sum(len(line) + 1 for line in person_lines) <= 1500
    for person in ("张明", "李华", "王林"):
        assert sum(len(line) + 1 for line in person_lines if person in line) <= 600


def test_memory_manually_broken_entity_can_be_fixed_without_breaking_recall(
    store: MemoryStore, claims: ClaimStore,
) -> None:
    saved = claims.remember(ClaimDraft(
        text="小林喜欢咖啡", kind="fact", person="小林", quote="我喜欢咖啡",
    )).claim
    page = store.workspace / str(saved.source_file)
    original = page.read_text()
    page.write_text(original.replace('"kind":"person"', '"kind":"unknown"'))
    provider = MemoryRuntimeContext(store, claims)
    assert provider.build("小林 咖啡") is None
    page.write_text(original)
    block = provider.build("小林")
    assert block is not None and saved.text in block.content
