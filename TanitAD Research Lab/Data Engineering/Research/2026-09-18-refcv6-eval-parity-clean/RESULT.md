# refcv6 eval139 -> a parity-clean **128**, proved by round trip and control

| | |
|---|---|
| Decision | `D-REFCV6-EVAL139-PARITY` (raised at `5e4a93d`, *"R18 CLOSED: … the gate immediately found 11 of the 139 eval clips inside the parity TRAIN corpus"*) |
| Agent / date | Data FlyWheel, 2026-09-18 |
| Worktree base | `ee1635a` (detached, `C:/Users/Admin/tanitad-wt-bevtac`); branch tip `fb008fb` on `agent/arch-inf-20260803` |
| Oracle module | `stack/tanitad/data/parity.py` — **byte-identical at `ee1635a`, `5e4a93d` and the tip `fb008fb`** (`git diff --name-only ee1635a agent/arch-inf-20260803 -- stack/tanitad/data/` is EMPTY; the apparent md5 mismatch worktree `d2dfffa4…` vs blob `e1923a6b…` is CRLF, 2 495 CRLFs, and the LF-normalised worktree file hashes to the blob) |
| Reproducer | `code/p1_eval139_parity_clean.py` |
| Primary artifact | `raw/parity_clean_report.json` |
| Evidence class | **MEASURED** (this package) unless a line says otherwise |

---

## 0. THE ONE SENTENCE

**The refcv6 eval set that may be QUOTED for a held-out read is 128 clips, never 139** — and
per half-cache it is **64 (splitA) and 64 (splitB)**, not 70 and 69.

This is the same shape as the banked `4713 vs 4719` lesson: *the corpus is 4 713 clips; quote
THAT number, never 4 719*. The bigger number is the number of files on disk; the smaller one is
the number of clips whose score means what the reader thinks it means. The gate itself prints the
sentence — `"The corpus is 128 clips; quote THAT number, never 139."`

---

## 1. Input — re-counted, not inherited

| fact | value |
|---|---|
| lines (non-empty) in `C:/Users/Admin/qland/ids139.txt` | 139 |
| unique clip ids | **139** |
| duplicates | 0 |
| sha256 of the id file | `5e34e6f01e61a6a6b7066965848b46ec43cee9f7fdb451fc7979750b3a15fa91` |

## 2. Oracles (a guard whose oracle is empty is not a guard)

`parity.require_ingest_gate()` returned `{'parity_train_clips': 2400, 'deployed_val_clips': 40}`;
the v7.2 eval oracle resolved with 147. Membership is **per-clip `sha256(clip_id)`**, self-checked
by `load_clip_digests` (count + digest-of-digests), never by provenance or by file name.

- `parity_train_clip_digests.json` — 2 400 digests, corpus `physicalai-train-e438721ae894`
- `deployed_val40_clip_digests.json` — 40 digests
- `v72_eval_clip_digests.json` — 147 digests

## 3. Both directions, re-derived (the brief's 11 was NOT taken on trust)

| direction | function | n of 139 |
|---|---|---|
| inside the **deployed val40** (what `role=""` checks) | `clips_in_deployed_val` | **0** |
| inside **parity TRAIN** `physicalai-train-e438721ae894` (what `role="eval"` checks) | `clips_in_parity_train` | **11** |
| inside the **v7.2 EVAL split** (147) | `clips_in_v72_eval` | 139 — *all of them* |
| ⇒ parity-clean for a held-out read | 139 − 11 | **128** |

Independently re-derived count **agrees with the brief: 11.**

## 4. The gate on the FULL 139 — both roles, because they are different questions

| call | outcome | wanted |
|---|---|---|
| `guard_corpus_build(139, role="eval", mode="refuse")` | **ParityViolation — REFUSED** | REFUSED |
| `guard_corpus_build(139, role="",     mode="refuse")` | **PASS** (`disjoint: True`, `decision_grade: True`) | PASS |

The refusal text: `corpus requested : 139 clip(s) / role : eval / disqualifying : 11 clips inside
physicalai-train-e438721ae894   <-- LEAK`.

⚠️ The `role=""` PASS is not an absolution. It answers *"does this corpus swallow the 40 episodes
every published open-loop number is quoted over?"* — no. It says nothing about held-out-ness, and
reading it as a clean bill of health is precisely the §10c role-table trap.

## 5. The clean list

`guard_corpus_build(139, label="refcv6-eval139", role="eval", mode="exclude")` →
**kept 128, dropped 11**, record `rule: "excluded because present: clips inside
physicalai-train-e438721ae894"`, `decision_grade: True`.

## 6. ⭐ THE ROUND TRIP — the positive proof (a count alone is not one)

Re-running the *same gate* at the *same role* on the *produced* list:

```
guard_corpus_build(kept_128, role="eval", mode="refuse")
  -> n_in 128 | in_parity_train 0 | in_deployed_val 0 | n_disqualifying 0
     disjoint True | decision_grade True | returned ids == the 128 submitted
```

**PASSES.** A filter that produced the right *number* by the wrong *rule* would still fail here;
this closes that gap.

## 7. ⛔ THE DISCRIMINATING CONTROL — the gate can still be made to fail

| control | trials | result |
|---|---|---|
| clean 128 **+ one excluded clip re-added** (every excluded clip tried, one at a time, n=129 each) | 11 | **11 / 11 REFUSED** — `1 in physicalai-train-e438721ae894` |
| clean 128 **+ 3 synthetic ids in neither oracle** (n=131) | 1 | **PASS** — an addition per se is not a leak |

Both outcomes shown, so the filter is demonstrated to *discriminate*, not merely to shrink a list.

## 8. ⚠️ The two 416×1024 half-caches — per-half clean n

`D:/Projects/TanitAD-artifacts/v2ep-eval139-416x1024cyl-{splitA,splitB}` (clip ids read from the
files themselves via `parity.v2_clip_ids`, i.e. what is ON DISK, not what a manifest claims):

| cache | clips present | excluded (in parity train) | **CLEAN n** | gate at `role="eval"` |
|---|---|---|---|---|
| `…-splitA` | 70 | **6** | **64** | REFUSED |
| `…-splitB` | 69 | **5** | **64** | REFUSED |
| `…-416x1024cyl` (full) | 139 | 11 | 128 | REFUSED |

Algebra, measured: `A ∩ B = 0`, `A ∪ B = exactly the 139 ids` (0 extra, 0 missing), full dir =
exactly the 139 ids, and `64 + 64 = 128` ✔. The halves are unequal in size (70/69) but **equal in
clean n (64/64)** — a panel that reports per half must use 64 and 64; a panel that pools must use
128. Neither half is usable as-is: **both caches contain excluded clips and both trip the gate.**
Nothing was rebuilt or deleted; the exclusion is a *selection* to apply at load time.

## 9. Population-level context (digest-set intersections; no ids enumerated)

| intersection | n |
|---|---|
| v7.2 EVAL (147) ∩ parity TRAIN (2 400) | **11** |
| v7.2 EVAL (147) ∩ deployed val40 (40) | 6 |
| parity TRAIN (2 400) ∩ deployed val40 (40) | **0** (re-verifies the MEASURED 2026-08-18 claim in `parity.py` §10c) |

⭐ The 11 in the v7.2 eval split as a whole are **the same 11** found in our 139 — the eval139
selection captured the entire overlap, so the 8 v7.2-eval clips outside our 139 are clean and
there is no second, unseen tranche to hunt.

Arithmetic worth stating so nobody re-derives it wrongly: `147 − 6 deployed-val40 = 141`, which is
the "141 of the v7.2 EVAL split's 147" the §10d comment names as living inside the B1 train cache
(INHERITED, repo source comment). eval139 is **139 of those 141**; why 2 are absent from the build
is **OPEN** — not established here and not needed for this result.

## 10. Agreement with the registered decision, and what this package adds

`Project Steering/GOALS_AND_CLAIMS.md` at the tip `fb008fb` (anchor
`<!-- GC-PARITY-EVAL139-11-IN-TRAIN-2026-09-18 -->`) registers `D-REFCV6-EVAL139-PARITY`. Every
number it states — 0 of 40 deployed val, 11 of 139 = 7.9 % in parity train, `role=""` PASSES,
`role="eval"` REFUSES, clean 128, and the `4713 vs 4719` shape — is **independently re-derived
here and agrees**. (The entry itself landed *after* `5e4a93d`; `5e4a93d` is the commit at which
the gate was wired and the 11 first surfaced.)

What this package adds on top of the registered entry: the produced **clean list** (sha12), the
**round trip** (§6), the **discriminating control** (§7), the **per-half clean n** (§8), and the
oracle-level intersections (§9).

## 11. What this does and does not change

- **Does not** move any landed claim: §10.6 was a COVERAGE pass (INHERITED, brief + `5e4a93d`).
- **Does** bind every future refcv6 arm scored on this set *as held-out*: exclude the 11 first,
  quote 128 (or 64 / 64 per half), and carry the `guard_corpus_build` record into the run's
  manifest — a filtered build whose manifest omits what was filtered reports a clip count that no
  longer matches the selection it names.
- The clean list is **not** a new cache. No cache was rebuilt, nothing was deleted.

## 12. ⛔ Defect found in passing — `parity.py:2405`, the refusal message misreports the other direction

The *record* is correct; the *human-readable refusal* is not. On our own call
(`role="eval"`, `in_parity_train=11`, `in_deployed_val=0`) the message printed:

```
  disqualifying    : 11 clips inside physicalai-train-e438721ae894   <-- LEAK
  (other direction : 11 in the val deployment, recorded, not disqualifying for this role)
```

while the record returned by the *same call* says `in_deployed_val: 0`. Source:

```python
f"  (other direction : {len(in_train) if heldout else len(in_val)} "
f"{'in the val deployment' if heldout else 'in the parity train split'}"
```

The number ternary is inverted relative to the label ternary, so **in both roles the disqualifying
count is printed a second time under the other direction's name**. Consequence, in the exact case
that produced this work package: an operator reading the refusal concludes 11 deployed-val
episodes are inside the eval set — a far more alarming and entirely false fact (the true value is
0), and it is the "true but wrong for the reader" class banked earlier. No test asserts this line
(`stack/tests/test_build_parity_guard.py` asserts only the `"INGEST GATE REFUSED"` substring).

**NOT fixed here** — `parity.py` is outside this package's mandate and is shared. ESCALATED to the
caller; a one-line fix plus a test that reads both numbers off one refusal is the remedy.

## 13. Deliverable manifest

In-repo (worktree `C:/Users/Admin/tanitad-wt-bevtac`, staged by explicit path, **not** committed):

| path | what |
|---|---|
| `TanitAD Research Lab/Data Engineering/Research/2026-09-18-refcv6-eval-parity-clean/RESULT.md` | this file |
| `…/code/p1_eval139_parity_clean.py` | the reproducer (input set → both directions → gate both roles → exclude → round trip → control → half-caches) |
| `…/raw/eval128_clean_sha12.txt` | **the clean eval set**, 128 lines, `sha12` only |
| `…/raw/eval139_excluded_sha12.txt` | the 11 excluded, `sha12` only |
| `…/raw/parity_clean_report.json` | every number above, machine-readable, incl. per-half `clean_sha12` lists |
| `…/raw/oracle_intersections.json` | §9 digest-set intersections |

**Out of repo** (gated-confidential, never commit): `C:/Users/Admin/qland/refcv6_eval128_sha12_map.json`
— the `sha12 → clip_id` mapping for `clean_128`, `excluded`, `splitA_clean`, `splitB_clean`.

`sha12` rule: `sha256(clip_id).hexdigest()[:12]` (`tanitad.data.semantic_map_gt.sha12`) — a HASH,
not a truncation of the id. The in-repo artifacts were leak-scanned: 0 UUID-shaped strings.

Run it again with:

```
set PYTHONPATH=C:\Users\Admin\tanitad-wt-bevtac\stack
set PYTHONIOENCODING=utf-8
python "TanitAD Research Lab/Data Engineering/Research/2026-09-18-refcv6-eval-parity-clean/code/p1_eval139_parity_clean.py"
```

Exit 0 iff **all** of: 139 unique in, `role="eval"` REFUSES the 139, `role=""` passes them,
exclude keeps exactly `139 − in_parity_train`, the round trip PASSES with 0 disqualifying, every
single-readd control REFUSES, and the synthetic-id control PASSES.
