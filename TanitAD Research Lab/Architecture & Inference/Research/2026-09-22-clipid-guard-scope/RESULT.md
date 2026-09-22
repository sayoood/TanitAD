
<!-- CLIPID-GUARD-SCANNED-THE-WRONG-TREE-2026-09-22 -->

### ⛔⛔ 2026-09-22 — the clip-id growth guard has been scanning a tree that is NEITHER what we bank NOR a superset of it: wrong in BOTH directions, now fixed with `--from-ref`

MEASURED by me, read-only, CPU only
(`TanitAD Research Lab/Architecture & Inference/Research/2026-09-22-clipid-guard-scope/`).
Found while checking my OWN package against the guard — which is the only reason it surfaced.

**`tools/clipid_scan.py` defaults to `--root .`, the LOCAL WORKTREE. Landings go to the PUSH
CLONE and never touch that worktree.** MEASURED:

| | worktree | push clone (what is banked) |
|---|---|---|
| research packages dated 2026-09-2x | **3** | **18** |
| `.md` files under the two scanned roots | 1,835 | 1,826 |

⇒ the two trees have **diverged in both directions**: the worktree is missing **15 of 18** recent
packages, *and* carries files the branch does not. It is not a subset, not a superset — a different
population.

**So the guard was wrong both ways at once:**
* ⛔ **FALSE NEGATIVE** — 15 banked packages were **invisible** to it. A growth guard that cannot
  see the growth is the `pgrep`/`df` family again: a probe reporting on the wrong scope, read as an
  answer.
* ⛔ **FALSE POSITIVE** — it reported `+1 uuid` in
  `2026-09-20-refe-plan/REVIEW_4_FINAL.md`, a file that exists **only in the worktree** and was
  **never landed**. I nearly recorded that as a leak in the record.

⭐ **AND THE RECORD ITSELF IS CLEAN.** Scanned properly, the banked tree reads **NO-GROWTH:
27 files / 120 uuids / 0 prefixes** — *exactly* the recorded floor. Nothing has leaked into what we
actually bank; the alarm was an artifact of where the guard was looking. ⚠️ That is a better
outcome than the alarm suggested and a worse one than "the guard was fine", and both halves are
stated.

**THE FIX: `--from-ref <ref> [--git-dir …]`** exports the banked tree with `git archive` and scans
that. `--root` answers *"what is on this disk"*; the question this guard exists to answer is *"what
is in the RECORD"*, and those are different populations.

⛔ **AND THE EMPTY-EXPORT REFUSAL IS THE LOAD-BEARING PART.** A failed or empty `git archive` would
scan an empty tree and report a confident **NO-GROWTH** — the *"empty read scored as an answer"*
family, and it would have made this guard permanently, silently green. So a non-zero exit or empty
output **REFUSES** by name. PROVEN: `--from-ref no/such/ref` exits **1** with
*"An EMPTY export would scan an empty tree and report NO-GROWTH, which is why this refuses instead
of continuing."*

⚠️ **Not claimed:** nothing here changes PI item 17's policy (banked ids stay; no redaction, no
history rewrite). This changes only **which tree the guard reads**. The `REVIEW_4_FINAL.md` UUID is
local, unlanded drift — it is not in the record and needs no redaction; if that file is ever landed,
the guard will now see it, which is the point.
