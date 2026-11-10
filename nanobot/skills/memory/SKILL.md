---
name: memory
description: Search past conversations in the agent's history log, and remember, look up, or forget long-term memory.
---

# Memory

## Search Past Events

Search the exact `History log` path from the system prompt using an available text-search tool. This path belongs to the agent workspace, which can differ from the current project.

The append-only JSONL log stores `cursor`, `timestamp`, and `content` per entry. Retrieve entries on demand by topic or date, and inspect neighboring entries when context matters.

## Remember, Look Up, and Forget

When the memory tools are available:

- Explicitly remember: use `memory_remember` with an atomic statement, kind, and actual user words as `quote`. Non-core preference/experience/circumstance/fact entries go to their bank pages. Decisions and commitments default to `bank/facts.md`; progress without a project goes to today's daily note. Use `core=true` only for stable information needed frequently.
- Link a project or person: pass `project` and/or `person`, with explicit person `aliases` when known. Existing display names and aliases reuse stable IDs. Project pages take precedence over person pages. People memory must be enabled; a quote or verifiable turn source is required. Do not infer personal facts or invent evidence.
- Update a project: use `memory_project(project, status, rename?)`. Completed/paused projects stop automatic injection; their evidence remains searchable.
- Correct a fact: use `memory_search`, then `memory_remember` with every replaced ID in `supersedes`. Use `due_at` for deadlines and `expires_at` only for information that ceases to be valid. Overdue commitments remain active.
- Look up or explain: `memory_search` searches core, bank, projects, people, and daily. `include_history=true` also returns expired/superseded entries. Use `memory_explain(claim_id=... or query=...)` for evidence, file location, entity links, ledger history, and the bounded bidirectional correction chain.
- Explicitly forget: call `memory_forget` with claim IDs or a query. Use `person` or `project` to list all entity-linked candidates, including claims in other pages and archives. Show ambiguous candidates and confirm the scope before retrying with IDs or `all_matches=true`. Entity confirmation also deletes the page and permanently tombstones its ID.

Report only what the tool confirms. Permanent tombstones prevent restored files from resurrecting forgotten entries. Forgetting cleans identifiable copies and matching history lines, but does not delete original chat sessions or Git history. Forgotten evidence is never returned by `memory_explain`.

Stable core stays in the system prompt. Current commitments, the most recent active projects, unambiguously mentioned people, and query matches share one bounded, deduplicated runtime context. A missing automatic recall is not proof that no memory exists; use `memory_search` when needed.
