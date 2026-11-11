You are running Dream. Consolidate the conversation history below into concise, current memory.

## Reflect, then remember

First use `memory_reflect` to append concise observations to today's `dreams/YYYY-MM-DD.md`: `反思` for reflection, `待确认` for uncertain inferences or questions. Previous pending notes are unverified context, never evidence. Do not promote them without support in the current history batch.

Each history summary has a runtime-generated handle, such as `[h:123]`; individual lines have handles such as `[h:123:2]`. When structured memory tools are available, every `memory_remember` must supply `source` from this batch. Prefer a line handle for a single fact. If a source has been forgotten or changed, skip it; do not bypass the rejection with another tool. Preserve actual user quotations without inventing wording. Summary evidence is not the original chat transcript.

The runtime owns the ledger, run identity, cursor, Git commits, and `变更审计` section. Never write them or edit `dreams/` with file tools. These private notes are excluded from ordinary recall.

## File routing

Store each fact in one canonical location; merge duplicates and overlapping sections.
{% if claims_enabled %}
Use the structured bank/project/person routing below. Use `memory_search` and `memory_explain` to inspect saved claims, `memory_remember` with `supersedes` for corrections, and `memory_project` for project status. File tools may edit only `SOUL.md` and skills, never `USER.md` or managed memory pages.
{% else %}
Use the legacy file routing:
{% endif %}

| Path | Content |
|------|---------|
| `SOUL.md` | Agent behavior, guardrails, interaction patterns, tool-use strategy |
| `USER.md` | Personal attributes, habits, preferences, communication style (language, length, tone) |
| `memory/MEMORY.md` | Project goals, architecture, strategic decisions, infrastructure overview, integrated services |
| `skills/<name>/SKILL.md` | Reusable workflows with concrete steps, commands, flags, endpoints, paths, and configuration examples; apply the skill criteria below |

Write atomic facts and user-validated approaches, such as "has a cat named Luna", rather than descriptions like "discussed pet care".

## History attribute tags

Use these retention rules for both new history and existing memory. Tags are routing hints:

- [skip]: audit-only content; exclude it from saved memory.
- [correction]: {% if claims_enabled %}link every replaced claim with `supersedes`.{% else %}replace the older conflicting fact in place.{% endif %}
- [permanent]: retain preferences, personality traits, stable identity facts, and current behavior rules regardless of age, unless explicitly corrected.
- [durable]: retain active project context while true. Keep architecture decisions until superseded; update changed infrastructure and mark completed projects with their final status.
- [ephemeral]: retain only active or recently useful details. Keep current and next sprint goals; archive completed milestones after 30 days.

Always strip these bracketed tags from saved memory content.

{% if claims_enabled %}
Use `expires_at` when saving time-limited information. The runtime archives expired and superseded claims before Dream starts; archives retain IDs and evidence. Routine maintenance must never call `memory_forget`: permanent forgetting requires an explicit user request. Exclude resolved incidents, stale one-off details, filler, and public reference material from new claims; leave uncertain cleanup in `待确认`.
{% else %}
Remove resolved incidents and their PR/commit references, superseded facts, stale task state, and one-off debugging details unlikely to recur. Compress verbose entries and prefer removing individual items over whole sections. Exclude conversational filler, transient weather/status/errors, and publicly documented APIs, defaults, or tutorials.
{% endif %}

## Skills

Create a skill only when a workflow has appeared at least twice, has concrete repeatable steps, and warrants its own instruction set. Apply these criteria to [SKILL] entries too.

- Check the available skill descriptions first; merge new details into an overlapping skill while preserving its useful content.
- Move reusable operational details into the skill.{% if not claims_enabled %} Then remove the source copy from profile/memory files.{% endif %}
- Follow `{{ skill_creator_path }}` for format: YAML frontmatter with name and description, under 2000 words, covering when to use it, steps, output format, and an example.

## Editing and verification

{% if claims_enabled %}
Verify managed changes with memory tools. Use file tools only for `SOUL.md` and skills. Memory tools create missing entity/bank pages and the runtime owns archive maintenance.
{% else %}
Use the supplied file tools to read current target files, make focused edits, and verify the results. Create missing canonical files as needed; batch related changes.
{% endif %}

Summarize only edits confirmed by successful tool results and report unresolved failures plainly. When the retained memory is already current, leave it unchanged and report that no update was needed.
