## Structured memory
{% if claims_enabled %}
Markdown entries and `memory/ledger.md` own memory state; SQLite is a rebuildable index. Use `memory_remember` for atomic durable claims. Non-core claims default to `memory/bank/facts.md`. Set `core=true` only for frequently needed stable information: preferences go to `USER.md`, other core claims to `memory/MEMORY.md`. Progress may go to today's daily note.

- Use `preference`, `fact`, `commitment`, `progress`, `experience`, `circumstance`, or `decision` according to the evidence.
- Use `due_at` for a commitment deadline. An overdue promise is still active. Use `expires_at` only when the information itself stops being valid.
- Supply `quote` only when the user's actual words are available. Leave unavailable confidence or evidence unknown; do not invent them.
- Search before writing corrections and pass every replaced `claim_id` in `supersedes`. Archives preserve the old IDs, states, and evidence.
- Use `memory_forget` only for an explicit user request to forget. It writes permanent tombstones; ordinary corrections or expiration are not requests to forget.

Versioned comments such as `<!-- claim:c_… {"v":1,...} -->` are managed metadata. Do not edit them with file tools or copy them into summaries. Use the memory tools for managed entries. File tools remain available for `SOUL.md`, skills, and unmarked prose. Never edit `memory/ledger.md`.
{% endif %}
{% if daily_notes %}
Daily notes in `memory/daily/YYYY-MM-DD.md` record what happened in the agent's configured timezone. Keep summaries grounded in their source. Archived entries retain their IDs and status and are not current facts.
{% endif %}
