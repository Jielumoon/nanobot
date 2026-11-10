import json
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest
from loguru import logger

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_claims import ClaimStore
from nanobot.agent.memory_documents import (
    ClaimDraft,
    DailyNotes,
    Entity,
    claim_marker,
    decode_claim,
    encode_entity,
    strip_claim_markers,
)
from nanobot.agent.memory_index import query_terms
from nanobot.config.schema import MemoryConfig


@pytest.fixture
def memory_warnings():
    messages = []
    sink = logger.add(lambda message: messages.append(message.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


@pytest.fixture(params=[None, False], ids=["fts", "like"])
def claims(request, tmp_path):
    moment = datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc)
    daily = DailyNotes(
        tmp_path / "memory", lock=threading.RLock(), timezone="Asia/Shanghai", clock=lambda: moment,
    )
    return ClaimStore(
        tmp_path, lock=daily.lock, timezone="Asia/Shanghai",
        clock=lambda: moment, use_fts=request.param,
    )


def test_routes_metadata_and_unknown_legacy_values(claims):
    pref = claims.remember(ClaimDraft(text="只喝手冲咖啡", kind="preference", core=True, quote="我只喝手冲"))
    fact = claims.remember(ClaimDraft(text="住在上海", kind="fact"))
    progress = claims.remember(ClaimDraft(text="记忆骨架进行中", kind="progress"))
    assert pref.claim.source_file == "USER.md"
    assert fact.claim.source_file == "memory/bank/facts.md"
    assert progress.claim.source_file == "memory/daily/2030-05-04.md"
    path = claims.workspace / "USER.md"
    text = path.read_text()
    assert '"v":1' in text
    assert '<!--' not in strip_claim_markers(text)
    path.write_text(text + "- 旧偏好 <!--claim:c_0123456789ab-->\n")
    legacy = claims.get("c_0123456789ab")
    assert legacy.kind == "preference"
    assert legacy.confidence is None and legacy.created_at is None and legacy.quote is None


def test_same_text_deduplicates_after_manual_edit(claims):
    first = claims.remember(ClaimDraft(text="住在上海", kind="fact"))
    path = claims.workspace / first.claim.source_file
    path.write_text(path.read_text().replace("住在上海", "住在北京"))
    result = claims.remember(ClaimDraft(text="住在北京", kind="fact"))
    assert not result.created and result.claim.claim_id == first.claim.claim_id
    assert claims.search("上海") == []
    assert claims.get(first.claim.claim_id).text == "住在北京"


def test_projection_follows_edit_move_delete_and_rebuild(claims):
    old = claims.remember(ClaimDraft(text="周五提交旧周报", kind="commitment", core=True))
    source = claims.workspace / old.claim.source_file
    original = source.read_text()
    edited = original.replace("旧周报", "新周报").replace('"kind":"commitment"', '"kind":"decision"')
    stamp = source.stat()
    source.write_text(edited)
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert claims.get(old.claim.claim_id).text == "周五提交新周报"
    assert claims.get(old.claim.claim_id).kind == "decision"
    assert claims.active_commitments(5) == []
    destination = claims.workspace / "memory/bank/facts.md"
    destination.parent.mkdir()
    source.rename(destination)
    assert not claims.get(old.claim.claim_id).is_core
    destination.write_text("")
    assert claims.get(old.claim.claim_id) is None
    destination.write_text(edited)
    assert claims.get(old.claim.claim_id).status == "active"


def test_database_loss_preserves_all_metadata_and_many_to_many_links(claims):
    first = claims.remember(ClaimDraft(text="旧计划一", kind="fact"))
    second = claims.remember(ClaimDraft(text="旧计划二", kind="fact"))
    ids = (first.claim.claim_id, second.claim.claim_id)
    replacement = claims.remember(ClaimDraft(
        text="合并计划", kind="commitment", quote="我们一起周五前交付", confidence=0.8,
        supersedes=ids, due_at="2030-05-06", expires_at="2031-05-06",
        person="小林", project="联合计划",
    ), source_ref="history:42", session_key="cli:direct", message_id="m1")
    extra = claims.remember(ClaimDraft(text="另一个后续决定", kind="decision", supersedes=ids))
    claims.db_path.unlink()
    restored = claims.get(replacement.claim.claim_id)
    assert restored == replacement.claim
    assert all(claims.get(old).status == "superseded" for old in ids)
    with sqlite3.connect(claims.db_path) as conn:
        links = set(conn.execute("SELECT new_id, old_id FROM claim_links"))
    assert links == {
        (new, old) for new in (replacement.claim.claim_id, extra.claim.claim_id) for old in ids
    }
    assert all(hit.claim_id not in ids for hit in claims.search("计划"))


def test_future_metadata_survives_expiry_archive(claims):
    remembered = claims.remember(ClaimDraft(
        text="临时工位", kind="circumstance", expires_at="2030-05-05",
    ))
    path = claims.workspace / remembered.claim.source_file
    path.write_text(path.read_text().replace('"v":1', '"v":1,"extension":{"owner":"保留"}'))
    claims.expire_due(now=claims.clock() + timedelta(days=5))
    claim = claims.get(remembered.claim.claim_id)
    assert claim.model_extra["extension"] == {"owner": "保留"}
    assert claim.status == "expired"
    assert claims.search("临时工位") == []
    assert claims.search("临时工位", include_history=True)[0].status == "expired"
    claims.db_path.unlink()
    assert claims.get(claim.claim_id).status == "expired"


def test_forget_all_matches_in_one_file_including_copies(claims):
    a = claims.remember(ClaimDraft(text="喜欢手冲咖啡", kind="fact"))
    b = claims.remember(ClaimDraft(text="咖啡豆每月买一次", kind="fact"))
    path = claims.workspace / "memory/bank/facts.md"
    path.write_text(path.read_text() + "- 办公室咖啡机在三楼\n- 保留的普通行\n")
    before = path.read_text()
    assert len(claims.forget(query="咖啡").candidates) == 3
    assert path.read_text() == before
    result = claims.forget(query="咖啡", all_matches=True)
    assert {c.claim_id for c in result.retracted} == {a.claim.claim_id, b.claim.claim_id}
    assert "咖啡" not in path.read_text()
    assert "保留的普通行" in path.read_text()
    path.write_text(before)
    claims.db_path.unlink()
    assert claims.search("咖啡") == []
    assert path.read_text() == before  # sync never cleans Markdown
    claims.reconcile_files()
    assert "咖啡" not in path.read_text()


def test_duplicate_marker_uses_first_and_forget_removes_every_copy(claims, memory_warnings):
    remembered = claims.remember(ClaimDraft(text="秘密猫咪 Luna", kind="fact", quote="Luna 是猫"))
    path = claims.workspace / remembered.claim.source_file
    line = path.read_text().splitlines()[-1]
    duplicate = line.replace("秘密猫咪 Luna", "另一个 Luna 副本")
    path.write_text(path.read_text() + duplicate.replace("<!-- claim:", "<!--claim:") + "\n")
    original = path.read_text()
    hits = claims.search("Luna")
    assert [(hit.text, hit.claim_id) for hit in hits] == [
        ("秘密猫咪 Luna", remembered.claim.claim_id), ("另一个 Luna 副本", None),
    ]
    assert hits[1].kind is None
    assert claims.get(remembered.claim.claim_id).text == "秘密猫咪 Luna"
    projected = claims.project_text(path.read_text(), path=remembered.claim.source_file)
    assert "秘密猫咪 Luna" in projected and "另一个 Luna 副本" in projected
    claims.rebuild_index()
    claims.search("Luna")
    assert sum("重复记忆标记" in warning for warning in memory_warnings) == 1
    result = claims.forget(claim_ids=[remembered.claim.claim_id])
    assert result.changed and claims.search("Luna") == []
    assert "Luna" not in path.read_text()
    assert claims.get(remembered.claim.claim_id).text == ""
    assert claims.get(remembered.claim.claim_id).quote is None
    path.write_text(original)
    claims.db_path.unlink()
    assert claims.search("Luna") == []
    claims.reconcile_files()
    assert "Luna" not in path.read_text()


def test_duplicate_becomes_claim_when_first_location_is_deleted(claims):
    remembered = claims.remember(ClaimDraft(text="最早的会议计划", kind="fact"))
    path = claims.workspace / remembered.claim.source_file
    original = path.read_text()
    duplicate = original.splitlines()[-1].replace("最早的会议计划", "另一处会议计划")
    path.write_text(original + duplicate + "\n")
    assert claims.get(remembered.claim.claim_id).text == "最早的会议计划"
    path.write_text(duplicate + "\n")
    assert claims.get(remembered.claim.claim_id).text == "另一处会议计划"
    assert claims.search("会议")[0].claim_id == remembered.claim.claim_id


def test_forget_id_scrubs_daily_history_and_database_and_dream(tmp_path):
    store = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=True, daily_notes=True))
    saved = store.claims.remember(ClaimDraft(text="养了一只叫 Luna 的猫", kind="fact", quote="我家猫叫 Luna"))
    store.append_daily_summary("- 养了一只叫 Luna 的猫")
    store.append_history('养了一只叫 Luna 的猫。原话：我家猫叫 Luna')
    store.append_history("继续完成项目")
    result, count = store.forget_memory(claim_ids=[saved.claim.claim_id])
    assert result.changed and count == 1
    assert "Luna" not in store.history_file.read_text()
    assert "Luna" not in store.daily.path_for().read_text()
    with sqlite3.connect(store.claims.db_path) as conn:
        assert "Luna" not in "\n".join(conn.iterdump())
    prompt, _ = store.build_dream_prompt()
    assert "养了一只叫 Luna 的猫" not in prompt and "继续完成项目" in prompt


def test_history_only_forget_and_restored_history_stays_filtered(tmp_path):
    store = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=True))
    store.append_history("- 保留的项目进度\n- 只有历史里有的秘密\n- 下周开会\n")
    original = store.history_file.read_text()
    result, count = store.forget_memory(query="秘密")
    assert result.changed and count == 1
    expected = "- 保留的项目进度\n[已遗忘]\n- 下周开会"
    assert store.read_unprocessed_history(0)[0]["content"] == expected
    restored = json.loads(original)
    restored["content"] = "# 新增标题\n" + restored["content"]
    store.history_file.write_text(json.dumps(restored, ensure_ascii=False) + "\n")
    store.claims.db_path.unlink()
    assert store.read_unprocessed_history(0)[0]["content"] == "# 新增标题\n" + expected
    assert "秘密" in store.history_file.read_text()
    prompt, _ = store.build_dream_prompt()
    assert "只有历史里有的秘密" not in prompt
    assert "保留的项目进度" in prompt and "下周开会" in prompt
    assert "秘密" not in store.history_file.read_text()
    assert store.reconcile_history() == 0


def test_forget_failure_persists_tombstone_before_cleanup_and_retry(claims, monkeypatch):
    from nanobot.agent import memory_documents

    saved = claims.remember(ClaimDraft(text="不能复活的内容", kind="fact"))
    path = claims.workspace / saved.claim.source_file
    before = path.read_text()
    write = memory_documents.atomic_write_text

    def fail_content(target, content):
        if target == path:
            raise OSError("磁盘写入失败")
        write(target, content)

    with monkeypatch.context() as patch:
        patch.setattr(memory_documents, "atomic_write_text", fail_content)
        with pytest.raises(OSError, match="磁盘"):
            claims.forget(claim_ids=[saved.claim.claim_id])
    assert path.read_text() == before
    claims.db_path.unlink()
    assert claims.search("复活") == []
    assert path.read_text() == before
    claims.reconcile_files()
    assert "复活" not in path.read_text()


def test_atomic_write_failure_does_not_truncate_or_remove_old_claim(claims, monkeypatch):
    from nanobot.utils import atomic_file

    saved = claims.remember(ClaimDraft(text="原计划", kind="fact"))
    path = claims.workspace / saved.claim.source_file
    before = path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("磁盘替换失败")

    with monkeypatch.context() as patch:
        patch.setattr(atomic_file, "replace_atomic", fail_replace)
        with pytest.raises(OSError, match="替换"):
            claims.remember(ClaimDraft(text="新计划", kind="fact", supersedes=(saved.claim.claim_id,)))
    assert path.read_bytes() == before
    assert claims.get(saved.claim.claim_id).status == "active"


def test_interrupted_supersede_is_filtered_and_reconciled(claims, monkeypatch):
    from nanobot.agent import memory_documents

    saved = claims.remember(ClaimDraft(text="旧的咖啡偏好", kind="preference", core=True))
    write = memory_documents.atomic_write_text

    def fail_cleanup(path, content):
        if path == claims.workspace / "USER.md":
            raise OSError("无法清理旧页")
        write(path, content)

    with monkeypatch.context() as patch:
        patch.setattr(memory_documents, "atomic_write_text", fail_cleanup)
        with pytest.raises(OSError, match="旧页"):
            claims.remember(ClaimDraft(
                text="新的咖啡偏好", kind="preference", supersedes=(saved.claim.claim_id,),
            ))
    assert claims.search("咖啡")[0].text == "新的咖啡偏好"
    claims.reconcile_files()
    assert "旧的咖啡偏好" not in (claims.workspace / "USER.md").read_text()
    assert claims.get(saved.claim.claim_id).status == "superseded"


def test_sync_reparses_only_content_changes(claims, monkeypatch):
    from nanobot.agent import memory_index

    claims.remember(ClaimDraft(text="原事实", kind="fact"))
    claims.remember(ClaimDraft(text="偏好条目", kind="preference", core=True))
    parsed = []
    decode = memory_index.decode_claim

    def record(line, path, number, heading="", *, issues):
        parsed.append(path)
        return decode(line, path, number, heading, issues=issues)

    monkeypatch.setattr(memory_index, "decode_claim", record)
    assert claims.sync_index() == 0
    assert parsed == []
    path = claims.workspace / "memory/bank/facts.md"
    stat = path.stat()
    path.write_text(path.read_text().replace("原事实", "新事实"))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert claims.sync_index() == 1
    assert set(parsed) == {"memory/bank/facts.md"}
    assert claims.search("新事实")[0].text == "新事实"


def test_stable_entity_identity_survives_rename_and_rebuild(claims):
    entity = Entity(kind="person", display_name="老王", aliases=("王老师",))
    path = claims.workspace / "memory/people/wang.md"
    path.parent.mkdir(parents=True)
    path.write_text(encode_entity(entity) + "\n# 老王\n")
    claims.sync_index()
    path.write_text(encode_entity(entity.model_copy(update={"display_name": "王先生"})) + "\n# 王先生\n")
    path.rename(path.with_name("teacher.md"))
    claims.db_path.unlink()
    claims.sync_index()
    with sqlite3.connect(claims.db_path) as conn:
        row = conn.execute("""
            SELECT path, entity_id, entity_json FROM documents WHERE entity_id IS NOT NULL
        """).fetchone()
    assert row[:2] == ("memory/people/teacher.md", entity.entity_id)
    assert "王先生" in row[2]


def test_expired_core_is_filtered_before_maintenance_and_overdue_stays(tmp_path):
    now = datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc)
    store = MemoryStore(
        tmp_path, timezone="Asia/Shanghai", clock=lambda: now,
        config=MemoryConfig(claims_enabled=True),
    )
    expired = store.claims.remember(ClaimDraft(
        text="临时有效的事实", kind="fact", core=True, expires_at="2030-05-01",
    ))
    pending = store.claims.remember(ClaimDraft(
        text="周五提交报告", kind="commitment", due_at="2030-05-01",
    ))
    prompt = store.prompt_memory(store.read_memory())
    assert "临时有效的事实" not in prompt
    from nanobot.agent.memory_recall import MemoryRuntimeContext

    block = MemoryRuntimeContext(store, store.claims).build("")
    assert block is not None
    assert "周五提交报告" in block.content and "已逾期" in block.content
    assert "周五提交报告" not in prompt
    store.claims.expire_due()
    store.claims.db_path.unlink()
    assert store.claims.get(expired.claim.claim_id).status == "expired"
    assert store.claims.active_commitments(5)[0].claim_id == pending.claim.claim_id
    assert store.claims.search("临时有效") == []


def test_draft_uses_agent_timezone_and_rejects_invalid_or_long_inputs():
    draft = ClaimDraft.model_validate(
        {"text": "按时交付", "kind": "commitment", "due_at": "2030-05-03"},
        context={"timezone": "Asia/Shanghai"},
    )
    assert draft.due_at == "2030-05-03T15:59:59Z"
    with pytest.raises(ValueError, match="expires_at"):
        ClaimDraft(text="开会", kind="commitment", expires_at="next week")
    with pytest.raises(ValueError, match="500"):
        ClaimDraft(text="文" * 501, kind="fact")


def test_fts_only_falls_back_for_missing_capability(tmp_path, monkeypatch):
    from nanobot.agent import memory_index

    connect = sqlite3.connect

    class NoTrigram(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "CREATE VIRTUAL TABLE" in sql:
                raise sqlite3.OperationalError("no such tokenizer: trigram")
            return super().execute(sql, parameters)

    monkeypatch.setattr(
        memory_index.sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=NoTrigram),
    )
    claims = ClaimStore(tmp_path, lock=threading.RLock())
    claims.remember(ClaimDraft(text="喜欢手冲咖啡", kind="preference"))
    assert not claims.fts_enabled
    assert claims.search("咖啡")[0].text == "喜欢手冲咖啡"

    class Locked(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "CREATE VIRTUAL TABLE" in sql:
                raise sqlite3.OperationalError("database is locked")
            return super().execute(sql, parameters)

    monkeypatch.setattr(
        memory_index.sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=Locked),
    )
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        ClaimStore(tmp_path / "locked", lock=threading.RLock()).sync_index()


def test_unmarked_search_and_query_terms(claims):
    path = claims.workspace / "memory/MEMORY.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text("# Memory\n\n## Project\n- nanobot runs on a 1 CPU / 1 GB server\n")
    hit = claims.search("server CPU")[0]
    assert (hit.path, hit.line, hit.claim_id) == ("memory/MEMORY.md", 4, None)
    assert query_terms("Deploy nanobot 部署机器") == [
        "deploy", "nanobot", "部署机", "署机器", "部署", "署机", "机器",
    ]
    claim = decode_claim(
        "- 旧事实" + claim_marker("c_0123456789ab"), "memory/MEMORY.md", 1, issues=set(),
    )
    assert claim.confidence is None


def test_daily_summary_strips_tags_and_uses_clock(tmp_path):
    daily = DailyNotes(
        tmp_path / "memory", lock=threading.RLock(), timezone="Asia/Shanghai",
        clock=lambda: datetime(2030, 5, 3, 20, tzinfo=timezone.utc),
    )
    daily.append_summary("- [ephemeral] 记录进度\n(nothing)", session_key="cli:direct")
    text = daily.path_for().read_text()
    assert "2030-05-04" in text and "04:00 会话摘要 · cli:direct" in text
    assert "- 记录进度" in text


def test_restore_keeps_ledger_and_cannot_revive_claim(tmp_path):
    store = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=True))
    store.git.init()
    saved = store.claims.remember(ClaimDraft(text="需要永久遗忘的地址", kind="fact"))
    store.git.auto_commit("dream: save address")
    store.forget_memory(claim_ids=[saved.claim.claim_id])
    forgotten_commit = store.git.auto_commit("dream: forget address")
    ledger = store.claims.documents.ledger.read_text()
    assert store.git.revert(forgotten_commit) is not None
    assert store.claims.documents.ledger.read_text() == ledger
    store.claims.db_path.unlink()
    assert store.claims.search("地址") == []
    store.reconcile_files()
    assert "地址" not in (tmp_path / "memory/bank/facts.md").read_text()
    assert store.claims.get(saved.claim.claim_id).status == "retracted"


def test_new_explicit_statement_after_forgetting_can_be_remembered(claims):
    first = claims.remember(ClaimDraft(text="重新提供的事实", kind="fact"))
    claims.forget(claim_ids=[first.claim.claim_id])
    new = claims.remember(ClaimDraft(text="重新提供的事实", kind="fact"))
    assert new.claim.claim_id != first.claim.claim_id
    assert [hit.claim_id for hit in claims.search("事实")] == [new.claim.claim_id]
    assert claims.get(first.claim.claim_id).text == ""


def test_broken_markers_remain_plain_text_and_warn_once(tmp_path, memory_warnings):
    from nanobot.agent.context import ContextBuilder

    builder = ContextBuilder(tmp_path, memory_config=MemoryConfig(claims_enabled=True))
    store = builder.memory
    good = store.claims.remember(ClaimDraft(text="正常的会议计划", kind="fact", core=True))
    broken = (
        '- 损坏括号的咖啡偏好 <!-- claim:c_111111111111 {"v":1 -->\n'
        '- 损坏 JSON 的咖啡偏好 <!-- claim:c_222222222222 {oops} -->\n'
        '- 另一个损坏 JSON <!-- claim:c_333333333333 {oops} -->\n'
    )
    store.write_memory(store.read_memory() + broken)
    for _ in range(2):
        assert "损坏括号的咖啡偏好" in builder.build_system_prompt()
        hits = store.claims.search("咖啡")
        assert len(hits) == 2 and all(hit.claim_id is None for hit in hits)
        assert store.claims.get("c_222222222222") is None
        assert store.claims.get(good.claim.claim_id).text == "正常的会议计划"
    store.claims.rebuild_index()
    assert sum("记忆标记格式无效" in warning for warning in memory_warnings) == 1
    assert sum("记忆元数据无效" in warning for warning in memory_warnings) == 1
    original = store.read_memory()
    store.write_memory(store.read_memory().replace("损坏括号", "改过的损坏括号"))
    builder.build_system_prompt()
    assert sum("记忆标记格式无效" in warning for warning in memory_warnings) == 2
    assert sum("记忆元数据无效" in warning for warning in memory_warnings) == 2
    store.write_memory(original)
    builder.build_system_prompt()
    assert sum("记忆标记格式无效" in warning for warning in memory_warnings) == 2
    assert sum("记忆元数据无效" in warning for warning in memory_warnings) == 2
    store.forget_memory(claim_ids=["c_111111111111", "c_222222222222"])
    assert store.claims.search("咖啡") == []
    store.write_memory(store.read_memory() + broken)
    assert "咖啡" not in builder.build_system_prompt()
    store.reconcile_files()
    assert "咖啡" not in store.read_memory()


def test_forget_id_redacts_only_matching_history_lines_and_skips_short_terms(tmp_path):
    store = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=True, daily_notes=True))
    short = store.claims.remember(ClaimDraft(text="猫", kind="fact", quote="狗"))
    coffee = store.claims.remember(ClaimDraft(text="喜欢手冲咖啡", kind="preference", quote="茶"))
    summary = "- 猫砂放在阳台\n- 喜欢手冲咖啡，已经确认\n\n- 狗粮在柜子里\n- 茶叶每周买"
    store.append_daily_summary(summary)
    cursor = store.append_history(summary)
    original = store.history_file.read_text()
    result, count = store.forget_memory(claim_ids=[short.claim.claim_id, coffee.claim.claim_id])
    assert result.redact_terms == ["喜欢手冲咖啡"]
    assert count == 1
    expected = summary.replace("- 喜欢手冲咖啡，已经确认", "[已遗忘]")
    assert store.read_unprocessed_history(0)[0]["content"] == expected
    daily = store.daily.path_for().read_text()
    assert "喜欢手冲咖啡" not in daily
    assert "猫砂放在阳台" in daily and "狗粮在柜子里" in daily and "茶叶每周买" in daily
    history_targets = [
        event.target_id for event in store.claims.documents.events()
        if event.target_type == "history"
    ]
    assert len(history_targets) == 1 and history_targets[0].startswith(f"{cursor}:")
    store.history_file.write_text(original)
    assert store.read_unprocessed_history(0)[0]["content"] == expected
    assert store.reconcile_history() == 1
    assert store.reconcile_history() == 0


def test_history_timestamp_keeps_local_display_format_with_injected_clock(tmp_path):
    now = datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc)
    store = MemoryStore(tmp_path, timezone="Asia/Shanghai", clock=lambda: now)
    store.append_history("- 今天完成计划")
    assert store.read_unprocessed_history(0)[0]["timestamp"] == "2030-05-04 00:30"
    assert "[2030-05-04 00:30] - 今天完成计划" in store.build_dream_prompt()[0]
