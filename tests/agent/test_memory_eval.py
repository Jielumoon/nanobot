"""Deterministic conversation stories: real tools, documents, index and prompt assembly.

Rows are (local name, memory_remember arguments). The local name gives later
steps a stable reference to the generated claim ID; no model chooses the rows.
Recall@3 measures the fixed lexical question set, not real LLM extraction quality.
"""

from __future__ import annotations

import json
from array import array
from datetime import datetime, timedelta, timezone

import pytest

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory_recall import MemoryRuntimeContext
from nanobot.agent.tools.context import RequestContext, request_context
from nanobot.agent.tools.memory_claims import (
    MemoryExplainTool,
    MemoryForgetTool,
    MemoryProjectTool,
    MemoryRememberTool,
    MemorySearchTool,
)
from nanobot.agent.tools.registry import ToolRegistry

FACTS = (
    ("coffee", {"kind": "preference", "text": "早餐饮品只喝手冲咖啡"}),
    ("editor", {"kind": "preference", "text": "代码编辑器使用 Neovim"}),
    ("meal", {"kind": "preference", "text": "午餐不吃香菜"}),
    ("bike", {"kind": "fact", "text": "自行车车架编号 BK-731"}),
    ("database", {"kind": "decision", "text": "数据库备份采用 PostgreSQL"}),
    ("trip", {"kind": "experience", "text": "去年去过敦煌博物馆"}),
    ("work", {"kind": "circumstance", "text": "目前在苏州远程办公"}),
    ("reading", {"kind": "preference", "text": "睡前阅读科幻小说"}),
    ("music", {"kind": "preference", "text": "通勤时听爵士乐"}),
    ("exercise", {"kind": "preference", "text": "周末通常去游泳"}),
)
QUESTIONS = (
    ("早上喝什么咖啡", "coffee"), ("早餐饮品", "coffee"),
    ("Neovim", "editor"), ("午餐香菜", "meal"), ("自行车编号", "bike"),
    ("PostgreSQL", "database"), ("敦煌", "trip"), ("远程办公", "work"),
    ("科幻", "reading"), ("爵士", "music"), ("游泳", "exercise"),
    ("苏州", "work"),
)


class Story:
    def __init__(self, workspace, *, use_fts):
        self.now = datetime(2030, 5, 3, 16, 30, tzinfo=timezone.utc)
        self.builder = ContextBuilder(
            workspace, timezone="Asia/Shanghai", clock=lambda: self.now,
        )
        self.store = self.builder.memory
        assert self.store.claims is not None
        self.claims = self.store.claims
        self.claims.index.want_fts = use_fts
        self.tools = ToolRegistry()
        for tool in (
            MemoryRememberTool, MemorySearchTool, MemoryForgetTool,
            MemoryExplainTool, MemoryProjectTool,
        ):
            self.tools.register(tool(self.store))
        self.saved = {}

    async def remember(self, name, **arguments):
        with request_context(RequestContext(
            channel="cli", chat_id="eval", session_key="cli:eval",
            message_id=name, original_user_text="请记住：" + arguments["text"],
        )):
            result = json.loads(await self.tools.execute("memory_remember", arguments))
        assert result["status"] == "saved", result
        self.saved[name] = result["claim"]
        return result["claim"]["claim_id"]

    async def search(self, query, *, history=False):
        result = json.loads(await self.tools.execute("memory_search", {
            "query": query, "limit": 3, "include_history": history,
        }))
        return result["results"]

    async def prompt(self, query):
        block = await MemoryRuntimeContext(self.store, self.claims).provide(RequestContext(
            channel="cli", chat_id="eval", original_user_text=query,
        ))
        messages = self.builder.build_messages(
            history=[], current_message=query,
            runtime_context_blocks=[block] if block is not None else [],
        )
        return "\n".join(message["content"] for message in messages)


@pytest.fixture(params=[True, False], ids=["trigram", "like"])
def story(tmp_path, request):
    return Story(tmp_path, use_fts=request.param)


async def test_memory_eval_said_then_recalled(story, record_property):
    for name, arguments in FACTS:
        await story.remember(name, **arguments)
    # Distractors share some words, but not the full facts requested by the questions.
    await story.remember("distractor", kind="fact", text="咖啡机滤纸放在办公室储物柜")
    system = story.builder.build_system_prompt()
    search_hits = context_hits = 0
    for query, expected in QUESTIONS:
        claim = story.saved[expected]
        found = await story.search(query)
        search_hits += any(hit["claim_id"] == claim["claim_id"] for hit in found)
        context_hits += claim["text"] in await story.prompt(query)
    search_rate = search_hits / len(QUESTIONS)
    context_rate = context_hits / len(QUESTIONS)
    record_property("recall_at_3", search_rate)
    record_property("runtime_recall_rate", context_rate)
    assert search_rate >= 0.90, (search_hits, len(QUESTIONS))
    assert context_rate >= 0.90, (context_hits, len(QUESTIONS))
    assert story.builder.build_system_prompt() == system


async def test_memory_eval_correction_replaces_current_fact_but_explains_history(story):
    old = await story.remember("before", kind="preference", text="早餐饮品选择拿铁")
    assert "早餐饮品选择拿铁" in await story.prompt("早餐饮品")
    new = await story.remember(
        "after", kind="preference", text="早餐饮品改成手冲咖啡", supersedes=[old],
    )
    prompt = await story.prompt("早餐饮品")
    assert "早餐饮品改成手冲咖啡" in prompt
    assert "早餐饮品选择拿铁" not in prompt
    assert {hit["claim_id"] for hit in await story.search("早餐饮品")} == {new}
    assert {hit["claim_id"] for hit in await story.search("早餐饮品", history=True)} == {old, new}
    for claim_id in (old, new):
        explained = json.loads(await story.tools.execute("memory_explain", {"claim_id": claim_id}))
        assert {node["claim_id"] for node in explained["claims"]} == {old, new}
        assert explained["supersedes"] == [{"new_id": new, "old_id": old}]
        nodes = {node["claim_id"]: node for node in explained["claims"]}
        assert nodes[old]["status"] == "superseded" and nodes[new]["status"] == "active"
        assert nodes[new]["evidence"]["source_ref"] == "session:cli:eval#after"


async def test_memory_eval_expiry_archives_but_overdue_commitment_stays_visible(story):
    expiring = await story.remember(
        "temporary", kind="circumstance", text="临时在杭州出差",
        expires_at=(story.now + timedelta(days=1)).isoformat(),
    )
    await story.remember(
        "promise", kind="commitment", text="答应交付迁移报告",
        due_at=(story.now + timedelta(days=1)).isoformat(),
    )
    story.now += timedelta(days=2)
    assert not await story.search("杭州出差")
    story.store.maintain_memory(run_id="eval-expiry")
    prompt = await story.prompt("今天有什么安排")
    assert "临时在杭州出差" not in prompt
    assert "答应交付迁移报告" in prompt and "已逾期" in prompt
    archive = await story.search("杭州出差", history=True)
    assert archive[0]["claim_id"] == expiring and archive[0]["status"] == "expired"
    assert archive[0]["source"].startswith("memory/daily/")


async def test_memory_eval_forget_cleans_every_projection_and_survives_restore(story):
    store = story.store
    store.git.init()
    secret = "保险柜口令是蓝鲸松林"
    quote = "口令用蓝鲸松林"
    claim_id = await story.remember("secret", kind="fact", text=secret, quote=quote, core=True)
    await story.remember("keep", kind="fact", text="不相关的邮箱地址保留", core=True)
    store.append_history(f"- {secret}\n- 不相关的要点保留", session_key="cli:eval")
    store.append_daily_summary(f"- {secret}\n- 不相关的进度保留")
    store.dreams.append("反思", [f"- {secret}"])
    store.dreams.append("待确认", [f"- {quote}", "- 不相关的问题保留"])
    pending = story.claims.index.pending_vectors("eval-vectors", 10)
    story.claims.index.save_vectors("eval-vectors", pending, [array("f", [1, 0]) for _ in pending])
    with story.claims.index.connect() as conn:
        assert conn.execute("SELECT 1 FROM vectors WHERE claim_id=?", (claim_id,)).fetchone()
    originals = story.claims.documents.snapshot(include_dreams=True)
    old_history = store.history_file.read_text()
    store.git.auto_commit("dream: eval before correction")
    store.write_soul("# 用于检验快照恢复的后续版本")
    selected = store.git.auto_commit("dream: eval selected")
    assert selected is not None

    result = json.loads(await story.tools.execute("memory_forget", {"claim_ids": [claim_id]}))
    assert result["status"] == "forgotten"
    ledger = story.claims.documents.ledger.read_bytes()

    def assert_scrubbed():
        documents = story.claims.documents.snapshot(include_dreams=True)
        text = "\n".join(documents.values()) + store.history_file.read_text()
        assert secret not in text and quote not in text
        assert "不相关的要点保留" in text and "不相关的问题保留" in text
        assert secret.encode() not in ledger and quote.encode() not in ledger
        with story.claims.index.connect() as conn:
            assert not conn.execute(
                "SELECT 1 FROM vectors WHERE claim_id=?", (claim_id,),
            ).fetchone()
            assert not conn.execute(
                "SELECT 1 FROM chunks WHERE text LIKE ?", ("%蓝鲸松林%",),
            ).fetchone()
            row = conn.execute(
                "SELECT text, metadata FROM claims WHERE claim_id=?", (claim_id,),
            ).fetchone()
            assert row["text"] == "" and quote not in row["metadata"]
        batch = store.build_dream_prompt()
        assert batch is not None and secret not in batch.prompt and quote not in batch.prompt

    assert_scrubbed()
    assert store.restore_memory(selected) is not None
    assert_scrubbed()
    # A manual restore plus a deleted projection must still respect the ledger.
    for path, text in originals.items():
        (store.workspace / path).write_text(text)
    store.history_file.write_text(old_history)
    story.claims.db_path.unlink()
    assert not await story.search("蓝鲸松林", history=True)
    assert "蓝鲸松林" not in await story.prompt("口令")
    store.reconcile_files()
    assert_scrubbed()
    assert story.claims.documents.ledger.read_bytes() == ledger
    explained = json.loads(await story.tools.execute("memory_explain", {"claim_id": claim_id}))
    assert set(explained) == {"status", "forgotten_at"} and explained["status"] == "已遗忘"


async def test_memory_eval_people_need_evidence_and_a_mention(story):
    rejected = await story.tools.execute("memory_remember", {
        "kind": "fact", "text": "小林喜欢攀岩", "person": "小林",
    })
    assert "有证据" in rejected and not story.claims.list_claims()
    await story.remember(
        "friend", kind="fact", text="小林喜欢周六攀岩", person="小林",
        quote="我周六喜欢攀岩", aliases=["阿林", "林"],
    )
    for query, visible in (("聊聊小林", True), ("阿林", True), ("攀岩", False), ("林", False)):
        assert ("小林喜欢周六攀岩" in await story.prompt(query)) is visible
    pending = json.loads(await story.tools.execute("memory_forget", {"person": "小林"}))
    assert pending["status"] == "ambiguous"
    forgotten = json.loads(await story.tools.execute("memory_forget", {
        "person": "小林", "all_matches": True,
    }))
    assert forgotten["status"] == "forgotten"
    assert not (story.store.workspace / "memory/people/小林.md").exists()
    assert "小林喜欢周六攀岩" not in await story.prompt("小林")


async def test_memory_eval_project_completion_removes_automatic_context(story):
    progress = await story.remember(
        "project", kind="progress", text="数据库迁移演练已通过", project="蓝桥",
    )
    assert "数据库迁移演练已通过" in await story.prompt("今天安排")
    result = json.loads(await story.tools.execute("memory_project", {
        "project": "蓝桥", "status": "completed",
    }))
    assert result["project"]["status"] == "completed"
    assert "数据库迁移演练已通过" not in await story.prompt("蓝桥迁移")
    assert any(hit["claim_id"] == progress for hit in await story.search("迁移演练"))
