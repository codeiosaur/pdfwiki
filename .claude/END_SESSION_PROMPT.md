Give me a complete handover bundle for another Claude Code (Haiku) session.

Write the following three files in the .claude folder:

---

1) HANDOVER.md

Include:
- Project goal
- Current status (what is done vs in progress)
- Key files and what they do (only important ones)
- Architecture / data flow (brief but clear)
- Constraints (tech stack, patterns, requirements, things to NOT change)
- Known issues, bugs, or uncertainties
- Next 3 concrete tasks (ordered, actionable)
- How to run, test, and validate the project
- Any context that would otherwise require digging through the codebase

---

2) STATE_SNAPSHOT.md

Constraints:
- Max 300 words
- Highly compressed

Include ONLY:
- Goal
- Current status
- Immediate next task
- Critical constraints

---

3) DECISIONS.md

Include a list of key decisions made so far.

For each decision:
- Decision
- Reasoning
- Alternatives (if any)
- Whether this can be changed later (flexible vs locked-in)

---

General rules:
- Be concise but complete
- Do NOT repeat the same information across files unless necessary
- Assume the next session has ZERO prior context
- Prefer clarity over cleverness
- Use bullet points and structured formatting