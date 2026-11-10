---
name: memory
description: Search past conversations in the agent's history log, and remember, look up, or forget long-term memory.
---

# Memory

## Search Past Events

Search the exact `History log` path from the system prompt using an available text-search
tool. This path belongs to the agent workspace, which can differ from the current project.

The append-only JSONL log stores `cursor`, `timestamp`, and `content` per entry. Retrieve
entries on demand by topic or date, and inspect neighboring entries when context matters.

## Remember, Look Up, and Forget

When the `memory_remember`, `memory_search`, and `memory_forget` tools are available:

- The user explicitly says to remember something: call `memory_remember` with one atomic
  statement, the best `kind` (`preference`, `fact`, `commitment`, `progress`), and the user's
  own words as `quote`. Add `expires_at` for deadlines or temporary details.
- The user corrects a remembered fact: `memory_search` for the old entry, then
  `memory_remember` the new statement with `supersedes` set to the old `claim_id`.
- The answer may depend on something remembered that is not in the prompt: `memory_search`
  first. Results cite `file:line` and `claim_id`.
- The user asks to forget something: `memory_forget` with `claim_ids` from `memory_search`,
  or with a `query`. When the result is `ambiguous`, show the candidates and ask which ones
  to forget before retrying with `claim_ids` or `all_matches`.

Report only what the tool result confirms. Forgetting removes the memory entries and redacts
the history log; it does not delete saved chat sessions.
