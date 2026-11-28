
## 2. The patch (Master Mind applies)

Two applyable variants carrying the same content.
**Option A** replaces the defective line(s) in place and keeps every old figure inside a "(corrected 2026-10-11: was ...)" note. It passes a numeric-token-superset guard (C1-C3, C8, C9) and **fails a strict line-superset guard by construction** (M4 proves that guard can go red).
**Option B** is insert-only: the old lines stay byte-identical (guard C6), at the price of leaving the mislabelled sentence standing with a correction beside it. Use B if the lander refuses any removed line.
Anchor on the verbatim OLD text (unique in both trees), not on a line number.

### 2.1 GOALS_AND_CLAIMS.md - hunk G1 (Option A) - line @@G1@@ at `@@REV@@` (live worktree line @@LG@@)

OLD (verbatim):
```
@@OLDG@@
```
NEW:
```
@@NEWG@@
```
The trailing `(update 2026-10-11: ...)` sentence in the status cell (random-pick control) is separable from the mislabel; drop that one sentence if only the attribution fix is wanted - the first cell stands alone.

### 2.2 GOALS_AND_CLAIMS.md - hunk G1-B (Option B): insert this one line directly after line @@G1@@; the old row is unchanged
```
@@BG@@
```

### 2.3 MODEL_REGISTRY.md - hunk R1 (Option A) - replace lines @@R2@@-@@R3@@ at `@@REV@@` (live worktree @@LR2@@-@@LR3@@)

OLD (verbatim, two physical lines):
```
@@OLDR1@@
@@OLDR2@@
```
NEW (three physical lines):
```
@@NR0@@
@@NR1@@
@@NR2@@
```

### 2.4 MODEL_REGISTRY.md - hunk R0 (OPTIONAL tier label) - line @@R0@@ at `@@REV@@`
OLD: `@@OLDH@@`
NEW: `@@NEWH@@`
Raw stamp (all three panel JSONs): "@@TIER@@".

### 2.5 MODEL_REGISTRY.md - hunk R1-B (Option B): insert block 1 directly above old line @@R2@@, blocks 2 and 3 directly after old line @@R3@@
```
@@BR0@@
```
```
@@BR1@@
```
```
@@BR2@@
```
