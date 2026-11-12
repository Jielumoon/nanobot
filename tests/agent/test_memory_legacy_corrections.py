"""Legacy corrections use the lifecycle store, never direct core edits by tools."""

import json
import threading

import pytest
from pydantic import ValidationError

from nanobot.agent.memory import MemoryStore
from nanobot.agent.memory_claims import ClaimStore
from nanobot.agent.memory_documents import ClaimDraft, legacy_reference
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.memory_claims import MemoryRememberTool, MemorySearchTool
from nanobot.config.schema import MemoryConfig


@pytest.fixture(params=[None, False], ids=["fts", "like"])
def store(request, tmp_path):
    return ClaimStore(tmp_path, lock=threading.RLock(), use_fts=request.param)


def seed(store, path="memory/MEMORY.md", text="旧部署位置"):
    file = store.workspace / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(f"# 记忆\n\n- {text}\n", encoding="utf-8")
    return file, store.search(text)[0].to_public()["legacy_ref"]


@pytest.mark.parametrize("path", ["USER.md", "memory/MEMORY.md"])
def test_legacy_correction_archives_unknown_evidence_and_rebuilds(store, path):
    file, ref = seed(store, path)
    saved = store.remember(ClaimDraft(
        text="新部署位置", kind="preference" if path == "USER.md" else "fact",
        core=True, supersedes_legacy=[ref],
    ), session_key="cli:test", message_id="new-message")
    old, = saved.superseded
    assert saved.claim.supersedes == (old.claim_id,)
    assert old.evidence.source_ref == ref
    assert old.created_at is None and old.kind is None
    assert old.evidence.session_key is None and old.quote is None
    assert "旧部署位置" not in file.read_text()
    assert all(h.claim_id != old.claim_id and h.text != old.text for h in store.search("旧部署位置"))
    historical, = [h for h in store.search("旧部署位置", include_history=True) if h.claim_id == old.claim_id]
    assert historical.claim_id == old.claim_id and historical.status == "superseded"
    assert historical.legacy_ref is None
    assert store.get(old.claim_id).source_file.startswith("memory/daily/")
    assert store.get(saved.claim.claim_id).session_key == "cli:test"
    store.db_path.unlink()
    assert store.get(old.claim_id).status == "superseded"
    assert store.get(saved.claim.claim_id).supersedes == (old.claim_id,)


def test_multiple_legacy_lines_and_marked_claim_can_be_corrected_together(store):
    file, _ = seed(store, text="旧线路甲")
    file.write_text(file.read_text() + "- 旧线路乙\n- 不相关保留\n")
    refs = [h.to_public()["legacy_ref"] for h in store.search("旧线路")]
    marked = store.remember(ClaimDraft(text="旧线路丙", kind="fact"))
    result = store.remember(ClaimDraft(
        text="新线路", kind="fact", core=True,
        supersedes=[marked.claim.claim_id], supersedes_legacy=refs,
    ))
    assert len(result.superseded) == 3
    assert "不相关保留" in file.read_text()
    assert "新线路" in file.read_text()
    assert "旧线路" not in file.read_text()
    historical = {h.claim_id for h in store.search("旧线路", include_history=True) if h.status == "superseded"}
    assert historical == {c.claim_id for c in result.superseded}


@pytest.mark.parametrize("change", ["edit", "move", "delete", "forget"])
def test_stale_reference_is_rejected_without_writes(store, change):
    file, ref = seed(store)
    if change == "edit":
        file.write_text(file.read_text().replace("旧部署位置", "已改位置"))
    elif change == "move":
        file.write_text("\n" + file.read_text())
    elif change == "delete":
        file.unlink()
    else:
        store.forget(query="旧部署位置", all_matches=True)
        # A stale backup must not resurrect a forgotten chunk.
        file.write_text("# 记忆\n\n- 旧部署位置\n")
    before = store.documents.snapshot()
    with pytest.raises(ValueError, match="引用已失效或已遗忘"):
        store.remember(ClaimDraft(
            text="新部署位置", kind="fact", project="不应创建", supersedes_legacy=[ref],
        ))
    assert store.documents.snapshot() == before
    assert all(h.text != "新部署位置" for h in store.search("新部署位置"))


def test_reusing_reference_is_rejected(store):
    _, ref = seed(store)
    draft = ClaimDraft(text="新位置", kind="fact", core=True, supersedes_legacy=[ref, ref])
    assert draft.supersedes_legacy == (ref,)
    assert len(store.remember(draft).superseded) == 1
    with pytest.raises(ValueError, match="引用已失效"):
        store.remember(draft)


@pytest.mark.parametrize("ref", [
    "legacy:SOUL.md:1:" + "a" * 64,
    "legacy:../USER.md:1:" + "a" * 64,
    "legacy:USER.md:0:" + "a" * 64,
    "legacy:USER.md:1:short",
])
def test_only_exact_core_line_references_are_valid(ref):
    with pytest.raises(ValidationError):
        ClaimDraft(text="新值", kind="fact", supersedes_legacy=[ref])


def test_reject_heading_and_malformed_marker(store):
    file, _ = seed(store)
    file.write_text("# 旧标题\n- 旧部署位置 <!-- claim:broken -->\n")
    assert "legacy_ref" not in store.search("旧部署位置")[0].to_public()
    for ref in [legacy_reference("memory/MEMORY.md", 1, "旧标题"),
                legacy_reference("memory/MEMORY.md", 2, "旧部署位置")]:
        with pytest.raises(ValueError):
            store.remember(ClaimDraft(text="新值", kind="fact", supersedes_legacy=[ref]))
    assert "claim:broken" in file.read_text()


def test_existing_duplicate_does_not_skip_legacy_correction(store):
    _, ref = seed(store)
    existing = store.remember(ClaimDraft(text="新部署位置", kind="fact"))
    corrected = store.remember(ClaimDraft(
        text="新部署位置", kind="fact", core=True, supersedes_legacy=[ref],
    ))
    assert corrected.created and corrected.claim.claim_id != existing.claim.claim_id
    assert len(corrected.superseded) == 1
    assert all(h.text != "旧部署位置" for h in store.search("旧部署位置"))


def test_forget_corrected_legacy_claim_does_not_revive_it(store):
    _, ref = seed(store)
    saved = store.remember(ClaimDraft(text="新值", kind="fact", core=True, supersedes_legacy=[ref]))
    old, = saved.superseded
    store.forget(claim_ids=[old.claim_id])
    store.db_path.unlink()
    assert not store.search("旧部署位置", include_history=True)
    assert store.get(saved.claim.claim_id).status == "active"


async def test_native_tools_expose_reference_and_record_only_new_turn_provenance(tmp_path):
    memory = MemoryStore(tmp_path, config=MemoryConfig(claims_enabled=True))
    seed(memory.claims)
    hits = json.loads(await MemorySearchTool(memory).execute(query="旧部署位置"))["results"]
    assert hits[0]["claim_id"] is None and hits[0]["legacy_ref"].startswith("legacy:")
    tool = MemoryRememberTool(memory)
    assert "supersedes_legacy" in tool.parameters["properties"]
    ctx = RequestContext(channel="cli", chat_id="direct", message_id="m-new", session_key="cli:direct")
    with request_context(ctx):
        result = json.loads(await tool.execute(
            text="新部署位置", kind="fact", core=True, supersedes_legacy=[hits[0]["legacy_ref"]],
        ))
    assert result["status"] == "saved"
    new = memory.claims.get(result["claim"]["claim_id"])
    assert new.message_id == "m-new"
    assert memory.claims.get(new.supersedes[0]).message_id is None


@pytest.mark.parametrize("route", ["bank", "project", "other_core", "two_cores"])
def test_cross_file_legacy_correction_is_rejected_before_publication(store, route):
    _, ref = seed(store)
    kwargs = {"core": True}
    refs = [ref]
    kind = "fact"
    if route == "bank":
        kwargs = {}
    elif route == "project":
        kwargs["project"] = "部署维护"
    elif route == "other_core":
        kind = "preference"
    else:
        _, user_ref = seed(store, "USER.md", "旧偏好")
        refs.append(user_ref)
    before = store.documents.snapshot()
    with pytest.raises(ValueError, match="不支持跨文件迁移"):
        store.remember(ClaimDraft(text="新值", kind=kind, supersedes_legacy=refs, **kwargs))
    assert store.documents.snapshot() == before


@pytest.mark.parametrize("mutation", ["change", "truncate", "remove"])
def test_file_change_between_sync_and_snapshot_is_rejected(store, monkeypatch, mutation):
    file, ref = seed(store)
    sync = store.index.sync

    def sync_then_edit(*args, **kwargs):
        result = sync(*args, **kwargs)
        if mutation == "remove":
            file.unlink()
        else:
            file.write_text("# 记忆\n" if mutation == "truncate" else "# 记忆\n\n- 完全不同的正文\n")
        return result

    monkeypatch.setattr(store.index, "sync", sync_then_edit)
    with pytest.raises(ValueError, match="引用已失效"):
        store.remember(ClaimDraft(text="新值", kind="fact", core=True, supersedes_legacy=[ref]))
    assert not file.exists() if mutation == "remove" else "claim:" not in file.read_text()


@pytest.mark.parametrize("failure", ["publish", "ledger", "archive", "remove_source"])
def test_single_core_publication_and_existing_reconciliation_survive_failures(store, monkeypatch, failure):
    file, ref = seed(store)
    before = file.read_text()
    apply = store.documents.apply
    append_events = store.documents.append_events
    calls = []

    def apply_with_failure(changes):
        calls.append(tuple(changes))
        if failure == "publish" and len(calls) == 1:
            raise OSError("injected publication failure")
        if failure == "archive" and len(calls) == 2:
            raise OSError("injected archive failure")
        if failure == "remove_source" and len(calls) == 2:
            archive_path = next(iter(changes))
            apply({archive_path: changes[archive_path]})
            raise OSError("injected source removal failure")
        return apply(changes)

    def fail_ledger(events):
        raise OSError("injected ledger failure")

    monkeypatch.setattr(store.documents, "apply", apply_with_failure)
    if failure == "ledger":
        monkeypatch.setattr(store.documents, "append_events", fail_ledger)
    with pytest.raises(OSError, match="injected"):
        store.remember(ClaimDraft(text="新部署位置", kind="fact", core=True, supersedes_legacy=[ref]))
    assert calls[0] == ("memory/MEMORY.md",)  # Both claims publish in one atomic file.
    monkeypatch.setattr(store.documents, "apply", apply)
    monkeypatch.setattr(store.documents, "append_events", append_events)
    store.db_path.unlink()
    if failure == "publish":
        assert file.read_text() == before
        assert store.search("旧部署位置")[0].legacy_ref == ref
        return
    active = [h for h in store.search("部署位置") if h.text == "新部署位置"]
    assert len(active) == 1
    new = store.get(active[0].claim_id)
    old_id, = new.supersedes
    assert store.get(old_id).status == "superseded"
    assert all(h.claim_id != old_id for h in store.search("旧部署位置"))
    store.reconcile_files()
    assert "旧部署位置" not in file.read_text()
    assert store.get(old_id).source_file.startswith("memory/daily/")
