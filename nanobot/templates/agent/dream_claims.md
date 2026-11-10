## Structured memory
{% if claims_enabled %}
Markdown entries and `memory/ledger.md` own memory state; SQLite is a rebuildable index. Use `memory_remember` for atomic durable claims. Non-core preferences, experiences, circumstances, and facts go to `memory/bank/preferences.md`, `experiences.md`, `circumstances.md`, and `facts.md` respectively. Decisions and commitments without an entity go to `bank/facts.md`; active commitments are injected every turn subject to the shared budget. Set `core=true` only for frequently needed stable information: preferences go to `USER.md`, other core claims to `memory/MEMORY.md`. Progress without a project goes to today's daily note.

- Use `preference`, `fact`, `commitment`, `progress`, `experience`, `circumstance`, or `decision` according to the evidence.
- Use `due_at` for a commitment deadline. An overdue promise is still active. Use `expires_at` only when the information itself stops being valid.
- Supply `quote` only when the user's actual words are available. Leave unavailable confidence or evidence unknown; do not invent them.
- Search before writing corrections and pass every replaced `claim_id` in `supersedes`. Archives preserve the old IDs, states, and evidence.
- Link `project` and/or `person` by name or explicit alias. Project-linked entries go to the project page; other person-linked entries go to the person page. People memory must be enabled and needs a real quote or verifiable source; never invent either.
- Use `memory_project` for active/completed/paused status and optional rename, preserving the entity ID. Use `memory_explain` to inspect evidence, timestamps, ledger events, entity links, and both directions of a correction chain.
- Use `memory_forget(person=... or project=...)` to list entity-linked candidates; only confirm `all_matches=true` when the user has requested that scope.
- Use `memory_forget` only for an explicit user request to forget. It writes permanent tombstones; ordinary corrections or expiration are not requests to forget.

Versioned comments such as `<!-- claim:c_… {"v":1,...} -->` are managed metadata. Do not edit them with file tools or copy them into summaries. Use the memory tools for managed entries. File tools remain available for `SOUL.md`, skills, and unmarked prose. Never edit `memory/ledger.md`.
{% endif %}
{% if daily_notes %}
Daily notes in `memory/daily/YYYY-MM-DD.md` record what happened in the agent's configured timezone. Keep summaries grounded in their source. Archived entries retain their IDs and status and are not current facts.
{% endif %}
