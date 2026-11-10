"""Exercise the optional evaluator with a scripted provider, never a real API."""

from __future__ import annotations

import json

import pytest

from nanobot.config.schema import Config
from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest
from scripts.memory_eval_llm import SCENARIO, evaluate


class ScriptedProvider(LLMProvider):
    def __init__(self, *, omit_person: bool):
        super().__init__(provider_name="eval")
        self.requests = []
        summary = "\n".join(f"- {statement}" for statement, _, _ in SCENARIO)
        calls = [ToolCallRequest(
            id="reflect", name="memory_reflect",
            arguments={"section": "反思", "lines": ["四条陈述都有明确来源"]},
        )]
        memories = [
            {"text": "早餐饮品是手冲咖啡", "kind": "preference", "source": "h:1:1"},
            {"text": "自行车车架编号 BK-731", "kind": "fact", "source": "h:1:2"},
            {"text": "数据库迁移演练已经通过", "kind": "progress", "project": "蓝桥",
             "source": "h:1:3"},
            {"text": "叶青每周六去攀岩", "kind": "fact", "person": "叶青",
             "quote": "我每周六都会去攀岩", "source": "h:1:4"},
        ]
        for index, arguments in enumerate(memories[:3] if omit_person else memories):
            calls.append(ToolCallRequest(
                id=f"remember-{index}", name="memory_remember", arguments=arguments,
            ))
        self.responses = iter([
            *(LLMResponse(content="收到") for _ in SCENARIO),
            LLMResponse(content=summary),
            LLMResponse(content="整理有证据的记忆", tool_calls=calls),
            LLMResponse(content="已完成整理"),
        ])

    async def chat(self, messages, **kwargs):
        self.requests.append(messages)
        return next(self.responses)

    def get_default_model(self):
        return "eval-model"


@pytest.mark.parametrize("omit_person", [False, True])
async def test_memory_llm_eval_runs_pipeline_and_reports_missing_evidence(
    tmp_path, monkeypatch, omit_person,
):
    original = tmp_path / "existing-workspace"
    original.mkdir()
    private = original / "USER.md"
    private.write_text("# 不属于评测的私人原文")
    config = Config.model_validate({
        "agents": {"defaults": {
            "workspace": str(original), "model": "eval-model", "provider": "custom",
            "maxTokens": 2048, "contextWindowTokens": 32000,
            "memory": {"claimsEnabled": False},
        }},
        "providers": {"custom": {
            "apiKey": "credential-must-not-appear", "apiBase": "https://unused.example/v1",
        }},
    })
    before = config.model_dump_json()
    provider = ScriptedProvider(omit_person=omit_person)
    monkeypatch.setattr("nanobot.providers.factory.make_provider", lambda *_args, **_kwargs: provider)
    output = tmp_path / "evaluation"

    report = await evaluate(config, output)

    assert report["passed"] is (not omit_person)
    assert report["recallAt3"] == (0.75 if omit_person else 1.0)
    checks = report["checks"]
    assert checks["conversationReplied"] and checks["compactCompleted"]
    assert checks["dreamCompleted"] and checks["cursorAdvanced"] and checks["summaryEvidence"]
    assert checks["projectProgress"] and checks["activeProjectContext"]
    assert checks["personEvidence"] is (not omit_person)
    assert checks["personMentionOnly"] is (not omit_person)
    assert len(provider.requests) == 7
    saved = (output / "report.json").read_text()
    assert json.loads(saved) == report and "credential-must-not-appear" not in saved
    assert "私人原文" not in json.dumps(provider.requests, ensure_ascii=False)
    assert config.model_dump_json() == before
    assert private.read_text() == "# 不属于评测的私人原文"
    assert not (original / "memory").exists()
    assert (output / "workspace/memory/history.jsonl").exists()
    assert list((output / "workspace/dreams").glob("*.md"))
    assert list((output / "sessions").glob("*/sessions.sqlite3"))
    with pytest.raises(FileExistsError):
        await evaluate(config, output)
    assert (output / "report.json").read_text() == saved
