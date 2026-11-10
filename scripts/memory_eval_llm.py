"""Opt-in model evaluation in a new workspace; never run with real providers in CI.

Run from the repository root with python -m scripts.memory_eval_llm --help.
Configuration is read only. Artifacts and isolated session storage stay under --output.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

from nanobot.agent.dream import run_dream
from nanobot.agent.loop import AgentLoop
from nanobot.agent.memory_recall import MemoryRuntimeContext
from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.config.loader import load_config, resolve_config_env_vars
from nanobot.config.schema import Config, MemoryConfig
from nanobot.session.manager import SessionManager

# User statements, probe queries, and required literal anchors. The anchors are
# deterministic checks, not a second LLM grading its own answers.
SCENARIO = (
    ("我的早餐饮品一直是手冲咖啡，不喝速溶。", "早餐饮品", ("手冲咖啡",)),
    ("我的自行车车架编号是 BK-731。", "自行车", ("BK-731",)),
    ("蓝桥项目的数据库迁移演练已经通过，下一步是灰度发布。", "迁移演练", ("迁移", "演练")),
    ("同事叶青亲口说：我每周六都会去攀岩。", "攀岩", ("攀岩",)),
)
MIN_RECALL = 0.75


async def evaluate(config: Config, output: Path) -> dict[str, object]:
    """Run actual conversation, compaction and Dream paths, then inspect their artifacts."""
    # A fresh directory is the CLI's write boundary: no existing run or workspace is reused.
    output.mkdir(parents=True, exist_ok=False)
    owned = config.model_copy(deep=True)
    owned.agents.defaults.workspace = str(output / "workspace")
    owned.agents.defaults.memory = MemoryConfig()
    owned.agents.defaults.unified_session = False
    owned.agents.defaults.provider_retry_mode = "standard"
    owned.bind_source_path(output / "config.json")
    sessions = SessionManager(owned.workspace_path, sessions_root=output / "sessions")
    agent = AgentLoop.from_config(owned, tool_registry=ToolRegistry(), session_manager=sessions)
    try:
        # Conversation and compaction need no tools. Preserve the registry object
        # captured by the consolidator; Dream constructs its own restricted set.
        for name in agent.tools.tool_names:
            agent.tools.unregister(name)
        key = "cli:memory-eval"
        conversation_replies: list[bool] = []
        for statement, _, _ in SCENARIO:
            response = await agent.process_direct(
                "Conversation fact, in the user's own words:\n" + statement
                + "\nAcknowledge briefly. This is not a request to save memory immediately.",
                session_key=key,
            )
            conversation_replies.append(response is not None and bool(response.content))
        summary = await agent.consolidator.compact_idle_session(
            key, runtime=agent.llm_runtime(), reason="manual",
        )
        store = agent.context.memory
        history = store.read_unprocessed_history(0)
        dream = await run_dream(agent, trigger="manual")
        claims = store.claims
        assert claims is not None  # This run explicitly constructs the default enabled memory.
        current = claims.list_claims(current=True, limit=-1)
        probes: list[dict[str, object]] = []
        hits = 0
        for _, query, anchors in SCENARIO:
            found = claims.search(query, limit=3)
            matched = any(
                hit.claim_id is not None
                and all(anchor.casefold() in hit.text.casefold() for anchor in anchors)
                for hit in found
            )
            hits += matched
            probes.append({
                "query": query, "expectedAnchors": list(anchors), "matched": matched,
                "results": [hit.to_public() for hit in found],
            })
        context = MemoryRuntimeContext(store, claims)
        ordinary = await context.provide(RequestContext(
            channel="cli", chat_id="eval", original_user_text="今天安排什么",
        ))
        mentioned = await context.provide(RequestContext(
            channel="cli", chat_id="eval", original_user_text="叶青有什么爱好",
        ))
        ordinary_text = ordinary.content if ordinary is not None else ""
        mentioned_text = mentioned.content if mentioned is not None else ""
        people = [claim for claim in current
                  if claim.person_id is not None and "攀岩" in claim.text and claim.quote]
        projects = [claim for claim in current
                    if claim.project_id is not None and claim.kind == "progress"
                    and "迁移" in claim.text]
        recall = hits / len(SCENARIO)
        checks = {
            "conversationReplied": all(conversation_replies),
            "compactCompleted": bool(summary) and not summary.startswith("[RAW]"),
            "historyWritten": bool(history),
            "dailyWritten": bool(list((store.memory_dir / "daily").glob("*.md"))),
            "dreamCompleted": dream.status == "completed",
            "cursorAdvanced": bool(history) and store.get_last_dream_cursor() == history[-1]["cursor"],
            "summaryEvidence": bool(current) and all(
                claim.evidence.source_kind == "summary"
                and bool(claim.evidence.source_ref)
                and claim.evidence.session_key == key
                for claim in current
            ),
            "personEvidence": bool(people),
            "projectProgress": bool(projects),
            "personMentionOnly": any(
                f"[{claim.claim_id}]" in mentioned_text for claim in people
            ) and all(claim.text not in ordinary_text for claim in people),
            "activeProjectContext": any(
                f"[{claim.claim_id}]" in ordinary_text for claim in projects
            ),
            "recallAt3": recall >= MIN_RECALL,
        }
        report: dict[str, object] = {
            "passed": all(checks.values()), "checks": checks,
            "recallAt3": recall, "minimumRecall": MIN_RECALL, "probes": probes,
            "model": agent.model, "dreamModel": (agent.dream_runtime() or agent.llm_runtime()).model,
            "embeddingEnabled": False, "claimCount": len(current),
            "workspace": str(store.workspace), "sessions": str(sessions.sessions_dir),
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        return report
    finally:
        await agent.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description="可选真实 LLM 记忆评测，会产生模型调用费用")
    parser.add_argument("--config", type=Path, help="读取现有 nanobot 配置；不改写")
    parser.add_argument("--output", type=Path, required=True, help="新的评测目录，必须尚不存在")
    args = parser.parse_args()
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    config = resolve_config_env_vars(load_config(args.config), config_path=args.config)
    report = asyncio.run(evaluate(config, args.output.expanduser().resolve()))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("评测通过" if report["passed"] else "评测未通过，请查看 report.json 中的检查项")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
