from datetime import datetime, timedelta, timezone

import pytest

from nanobot.agent.memory_claims import (
    ClaimStore,
    DailyNotes,
    claim_marker,
    query_terms,
    strip_claim_markers,
)


@pytest.fixture(params=[None, False], ids=["fts", "like"])
def claims(request, tmp_path):
    (tmp_path / "memory").mkdir()
    (tmp_path / "memory" / "MEMORY.md").write_text(
        "# Memory\n\n## Project\n- nanobot runs on a 1 CPU / 1 GB server\n",
        encoding="utf-8",
    )
    return ClaimStore(tmp_path, daily=DailyNotes(tmp_path / "memory"), use_fts=request.param)


def _read(claims: ClaimStore, relative: str) -> str:
    return (claims.workspace / relative).read_text(encoding="utf-8")


def test_search_mode_matches_requested_backend(claims, request):
    assert claims.fts_enabled is (request.node.callspec.id == "fts")


def test_remember_routes_kinds_to_markdown_with_markers(claims):
    preference = claims.remember("只喝手冲咖啡", "preference", quote="我只喝手冲")
    fact = claims.remember("住在上海", "fact")
    progress = claims.remember("记忆第一版骨架进行中", "progress")

    assert f"- 只喝手冲咖啡{claim_marker(preference.claim.claim_id)}" in _read(claims, "USER.md")
    memory = _read(claims, "memory/MEMORY.md")
    assert "## 已记住的事实" in memory
    assert claim_marker(fact.claim.claim_id) in memory
    daily = claims.daily.path_for().read_text(encoding="utf-8")
    assert "## 进度" in daily and claim_marker(progress.claim.claim_id) in daily

    stored = claims.get(preference.claim.claim_id)
    assert stored is not None
    assert (stored.status, stored.kind, stored.quote) == ("active", "preference", "我只喝手冲")
    assert (stored.source_file, stored.source_line) == ("USER.md", 2)


def test_remember_same_text_returns_existing_claim(claims):
    first = claims.remember("住在上海", "fact")
    second = claims.remember("住在上海", "fact")

    assert not second.created
    assert second.claim.claim_id == first.claim.claim_id
    assert _read(claims, "memory/MEMORY.md").count("住在上海") == 1


def test_supersede_replaces_markdown_entry(claims):
    old = claims.remember("喜欢手冲咖啡", "preference")
    new = claims.remember("改喝冷萃咖啡", "preference", supersedes=[old.claim.claim_id])

    user = _read(claims, "USER.md")
    assert "喜欢手冲咖啡" not in user
    assert "改喝冷萃咖啡" in user
    assert claims.get(old.claim.claim_id).status == "superseded"
    assert new.claim.supersedes_claim_id == old.claim.claim_id
    assert [hit.claim_id for hit in claims.search("咖啡")] == [new.claim.claim_id]


def test_supersede_rejects_inactive_claims(claims):
    old = claims.remember("喜欢手冲咖啡", "preference")
    claims.remember("改喝冷萃咖啡", "preference", supersedes=[old.claim.claim_id])

    with pytest.raises(ValueError, match="superseded"):
        claims.remember("改喝茶", "preference", supersedes=[old.claim.claim_id])


def test_forget_by_id_removes_entry_and_search_hit(claims):
    kept = claims.remember("住在上海", "fact")
    gone = claims.remember("养了一只叫 Luna 的猫", "fact")

    result = claims.forget(claim_ids=[gone.claim.claim_id])

    assert [claim.claim_id for claim in result.retracted] == [gone.claim.claim_id]
    assert "Luna" not in _read(claims, "memory/MEMORY.md")
    assert claims.get(gone.claim.claim_id).status == "retracted"
    assert claims.search("Luna 猫") == []
    assert claims.get(kept.claim.claim_id).status == "active"
    assert "养了一只叫 Luna 的猫" in result.redact_terms


def test_forget_ambiguous_query_lists_candidates_without_changes(claims):
    claims.remember("喜欢手冲咖啡", "preference")
    (claims.workspace / "memory" / "MEMORY.md").write_text(
        _read(claims, "memory/MEMORY.md") + "- 办公室咖啡机在三楼\n", encoding="utf-8",
    )

    result = claims.forget(query="咖啡")

    assert not result.changed
    assert len(result.candidates) == 2
    assert "咖啡机" in _read(claims, "memory/MEMORY.md")

    applied = claims.forget(query="咖啡", all_matches=True)
    assert len(applied.retracted) == 1
    assert [line.text for line in applied.removed_lines] == ["- 办公室咖啡机在三楼"]
    assert "咖啡" not in _read(claims, "USER.md") + _read(claims, "memory/MEMORY.md")


def test_expired_claims_are_archived_in_daily_note(claims):
    claim = claims.remember("周五前提交周报", "commitment", expires_at="2026-11-13")
    assert [c.claim_id for c in claims.active_commitments(5)] == [claim.claim.claim_id]

    expired = claims.expire_due(now=datetime.now(timezone.utc) + timedelta(days=3650))

    assert [c.claim_id for c in expired] == [claim.claim.claim_id]
    assert claims.get(claim.claim.claim_id).status == "expired"
    assert "周报" not in _read(claims, "memory/MEMORY.md")
    daily = claims.daily.path_for().read_text(encoding="utf-8")
    assert "## 已过期归档" in daily and "周五前提交周报" in daily
    assert claims.active_commitments(5) == []


def test_invalid_expiry_is_rejected(claims):
    with pytest.raises(ValueError, match="expires_at"):
        claims.remember("下周开会", "commitment", expires_at="next week")


def test_search_covers_unmarked_markdown_with_line_numbers(claims):
    hits = claims.search("server CPU")

    assert hits[0].path == "memory/MEMORY.md"
    assert hits[0].line == 4
    assert hits[0].claim_id is None
    assert hits[0].text == "nanobot runs on a 1 CPU / 1 GB server"


def test_search_matches_chinese_without_segmentation(claims):
    claim = claims.remember("喜欢手冲咖啡", "preference")

    hits = claims.search("我平时喝什么咖啡")

    assert [hit.claim_id for hit in hits] == [claim.claim.claim_id]


def test_index_follows_manual_file_edits(claims):
    claim = claims.remember("住在上海", "fact")
    memory_file = claims.workspace / "memory" / "MEMORY.md"
    memory_file.write_text(
        memory_file.read_text(encoding="utf-8").replace(
            f"- 住在上海{claim_marker(claim.claim.claim_id)}\n", "- 最近在学 Rust\n",
        ),
        encoding="utf-8",
    )

    assert [hit.text for hit in claims.search("Rust")] == ["最近在学 Rust"]
    assert claims.get(claim.claim.claim_id).status == "retracted"


def test_ledger_is_rebuilt_from_markers_after_database_loss(claims):
    claim = claims.remember("周五前提交周报", "commitment", quote="周五前给老王")
    claims.db_path.unlink()

    rebuilt = ClaimStore(claims.workspace, daily=claims.daily, use_fts=claims._want_fts)
    rebuilt.rebuild_index()

    restored = rebuilt.get(claim.claim.claim_id)
    assert restored is not None
    assert (restored.kind, restored.status, restored.text) == (
        "commitment", "active", "周五前提交周报",
    )


def test_query_terms_split_latin_words_and_cjk_ngrams():
    assert query_terms("Deploy nanobot 部署机器") == [
        "deploy", "nanobot", "部署机", "署机器", "部署", "署机", "机器",
    ]


def test_strip_claim_markers_keeps_text():
    assert strip_claim_markers(f"- 住在上海{claim_marker('c_0123456789ab')}") == "- 住在上海"


def test_daily_summary_strips_tags(tmp_path):
    daily = DailyNotes(tmp_path / "memory")

    daily.append_summary("- [ephemeral] Working on memory v1\n(nothing)", session_key="cli:direct")

    text = daily.path_for().read_text(encoding="utf-8")
    assert text.startswith(f"# {daily.path_for().stem}\n")
    assert "会话摘要 · cli:direct" in text
    assert "- Working on memory v1" in text
    assert "(nothing)" not in text
