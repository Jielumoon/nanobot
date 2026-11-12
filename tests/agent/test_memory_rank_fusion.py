"""Short-list fusion keeps either retriever's head above weak tail agreement."""
from __future__ import annotations

from dataclasses import replace

import pytest

from nanobot.agent.memory_index import MemoryIndex, SearchHit


def hit(number: int, *, claim: bool = True) -> SearchHit:
    return SearchHit(
        path="memory/bank/facts.md", line=number + 1, text=f"Fact {number}",
        claim_id=f"c_{number:012x}" if claim else None, kind="fact", score=0.5,
    )


@pytest.mark.parametrize("semantic_head", [False, True])
def test_single_head_outranks_agreement_in_both_tails(semantic_head):
    target = hit(100)
    tails = [hit(n) for n in range(20)]
    ranking = [hit(101), target, *[hit(200 + n) for n in range(10)], *tails[:8]]
    other = tails
    lexical, semantic = (other, ranking) if semantic_head else (ranking, other)
    result = MemoryIndex._fuse(lexical, semantic)
    # A rank-2 head must beat weak dual matches deep in the other ranking.
    ranks = {item.claim_id: n for n, item in enumerate(result)}
    assert ranks[target.claim_id] < ranks[tails[5].claim_id]
    assert target.claim_id in {item.claim_id for item in result[:5]}


@pytest.mark.parametrize("semantic_only", [False, True])
def test_one_retriever_preserves_order_and_metadata(semantic_only):
    ranking = [replace(hit(n, claim=False), legacy_ref=f"legacy-{n}") for n in range(7)]
    lexical, semantic = ([], ranking) if semantic_only else (ranking, [])
    result = MemoryIndex._fuse(lexical, semantic)
    assert [replace(item, score=0.5) for item in result] == ranking
    assert [item.score for item in result] == [1 / (n + 2) for n in range(7)]


def test_fusion_retains_exact_lexical_head_despite_weak_semantics():
    exact = replace(hit(100), text="ZXQ98473 /opt/example/sessions.sqlite3")
    semantic = [*[hit(n) for n in range(19)], exact]
    result = MemoryIndex._fuse([exact], semantic)
    assert result[0].claim_id == exact.claim_id
    assert result[0].score == 1 / 2 + 1 / 21
    assert len(result) == 20


def test_fusion_combines_claim_identity_and_keeps_unmarked_locations_distinct():
    claim = hit(1)
    old = replace(hit(2, claim=False), legacy_ref="legacy-ref")
    another = replace(old, line=old.line + 1)
    result = MemoryIndex._fuse([claim, old, another], [claim, old])
    assert len(result) == 3
    assert result[0].claim_id == claim.claim_id and result[0].score == 1.0
    assert result[1].legacy_ref == "legacy-ref" and result[1].score == 2 / 3
    assert result[2].line == another.line


def test_empty_fusion_does_not_invent_candidates():
    assert MemoryIndex._fuse([], []) == []


@pytest.mark.parametrize("semantic_head", [False, True])
def test_single_rank_two_beats_dual_rank_six(semantic_head):
    target, shared = hit(100), hit(101)
    head = [hit(0), target, hit(1), hit(2), hit(3), shared]
    other = [*[hit(200 + n) for n in range(5)], shared]
    lexical, semantic = (other, head) if semantic_head else (head, other)
    result = MemoryIndex._fuse(lexical, semantic)
    ranks = {item.claim_id: n for n, item in enumerate(result)}
    assert ranks[target.claim_id] < ranks[shared.claim_id]


def test_fusion_score_tie_keeps_lexical_first():
    lexical, semantic = hit(1), hit(2)
    result = MemoryIndex._fuse([lexical], [semantic])
    assert [item.claim_id for item in result] == [lexical.claim_id, semantic.claim_id]
    assert result[0].score == result[1].score
