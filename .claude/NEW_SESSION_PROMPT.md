**Status:** Optimizing for 900k token combined test run with multi-backend distribution - colloquially referred to as the "War and Peace" run due to its length despite the PDF being unrelated to the novel. Completed 500k token test with Gemma 3 primary and Cerebras/Groq fallbacks.

### Read These First (in priority order)
1. **CLAUDE.md** - Rules of engagement for this session and the project
2. **STATE_SNAPSHOT.md** (5 min) — Current goal, status, immediate next task, critical constraints
3. **DECISIONS.md** (10 min) — Key design decisions, reasoning, what can/cannot change
4. **HANDOVER.md** (15 min) — Complete context: architecture, files, full task list, how to build/test

### Key Assumptions
- These handover docs are authoritative for this phase of work
- You have ZERO prior context beyond these docs; read them before asking clarifying questions
- If you question a design decision, check DECISIONS.md reasoning first
- If you need to understand *why* something exists, HANDOVER.md has architectural context
- Never run git commands (commit, merge, etc.) without asking first. Adding "written by Claude" or similar to commit messages is unnecessary and may be considered a breach of protocol.
- CLAUDE.md contains the "rules of engagment" for this session and project. It is the constitution; the other documents are the laws and history. Never deviate from CLAUDE.md without explicit permission to do so. Never modify CLAUDE.md without explicit permission to do so.

### Immediate Next Steps
1. **Read the four docs** (30 min total)
2. **Understand the constraint map** (What's locked-in? What's flexible? See DECISIONS.md)
3. **Verify build/test works:** Run the command in HANDOVER.md → "How to run, test, and validate"
4. **Pick up where it left off:** Start with the first actionable item in the "Next 3 concrete tasks" section of HANDOVER.md
5. **Ask clarifying questions** if the docs don't cover something, but assume they're complete first

### If Something Isn't in the Docs
- **Architecture question?** → Check HANDOVER.md (Architecture & Data Flow)
- **Why was X chosen?** → Check DECISIONS.md (Reasoning column)
- **How do I test Y?** → Check HANDOVER.md (How to run, test, and validate)
- **What files do I touch?** → Check HANDOVER.md (Key Files table)
- **Is X change safe?** → Check DECISIONS.md (Locked-in vs. Flexible column)
- **What am I allowed to do?** -> Check CLAUDE.md (Rules of Engagement)

If still unclear after checking docs, ask; the docs may be incomplete. Ask as needed.

### Working Agreement
- Follow the protocol/constraints described in the docs (they apply unless you get explicit permission to deviate)
- If you find a doc is wrong or incomplete, correct it immediately so the next session benefits (except CLAUDE.md)
- Before starting a new branch or phase, create/update these docs (except CLAUDE.md) so the handoff is smooth
- Communicate openly about blockers, questions, and progress