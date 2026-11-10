## Structured memory
{% if claims_enabled %}
This workspace keeps a claim ledger next to the memory files. Record each atomic, durable item from the history with `memory_remember` instead of hand-editing it into `USER.md` or `memory/MEMORY.md`; the tool writes the Markdown entry and routes it by kind:

- `preference`: stable user preferences, habits, and communication style (goes to `USER.md`)
- `fact`: durable facts about the user, their work, or their environment (goes to `memory/MEMORY.md`)
- `commitment`: promises, deadlines, and to-dos the user or agent agreed to; set `expires_at` when a date is known (goes to `memory/MEMORY.md`)
- `progress`: work-in-progress status that matters for the next few days{% if daily_notes %} (goes to today's daily note){% endif %}

For every claim:

- pass `quote` with the user's own words when the history contains them, and a `confidence` below 1 when the history only implies the claim
- map history tags: `[permanent]` → `preference` or `fact`, `[durable]` → `fact`, `[ephemeral]` → `progress` or `commitment` with `expires_at`, `[correction]` → a new claim with `supersedes` set to the outdated claim
- call `memory_search` first and pass `supersedes` with the outdated `claim_id` when a claim replaces an existing one; do not create duplicates
- use `memory_forget` with a `claim_id` for entries that are no longer true and have no replacement

Lines ending in `<!-- claim:c_… -->` are managed claim entries. Do not edit, move, or delete them with file tools; use `memory_remember` with `supersedes` or `memory_forget`. Keep using file tools for `SOUL.md`, skills, and for reorganizing unmarked prose.
{% endif %}
{% if daily_notes %}
Daily notes in `memory/daily/YYYY-MM-DD.md` record what happened each day. You may append to the current day's note; leave past days unchanged except to remove content the user asked to forget.
{% endif %}
