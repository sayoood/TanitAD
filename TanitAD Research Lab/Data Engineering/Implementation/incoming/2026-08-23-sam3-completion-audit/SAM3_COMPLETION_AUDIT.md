# SAM3 backfill — completion audit, by content

**Package** `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-sam3-completion-audit/`
**Date** 2026-08-23 · **Evidence class of every number below** MEASURED, from
`raw/sam3_completion_audit.json`, produced by `code/sam3_audit.py` reading all **230** records
(115 per prefix) off HF `Sayood/tanitad-ph0-aug120`. Nothing here is quoted from a prior report.

---

## 0. Headline — state both numbers

| corpus | COMPLETE | INCOMPLETE | verdict |
|---|---|---|---|
| `sam3_backfill/` (v1) | **83 / 115** | **32 / 115** | ⚠️ incomplete, and **SUPERSEDED** |
| `sam3_backfill_v2/` (v2) | **115 / 115** | **0 / 115** | ✅ **complete** |

⭐ **THE BACKFILL IS COMPLETE — at `sam3_backfill_v2/`, not at the address the brief named.**
The 32 residual clips in v1 are not outstanding work: v2 re-ran all 115 from scratch at a lower
detection floor with the scene channel, and every one of the 32 is complete there. **There is no
remaining GPU work on this backfill.** The remaining work is a consumer move and a supersession
marker — §6.

⛔ **Both INHERITED claims I was handed are REFUTED by content:**

| inherited claim | measured | where the claim came from |
|---|---|---|
| *"77/115 content-complete"* | **83/115** on v1, **115/115** on v2 | 77 = complete **AND** `n_det_total > 0`. It silently penalises **6** clips that are complete with a live control and a legitimately empty scene. §3.1 |
| *"clip `24b6948f` stores `live: false` against `{road: 2, sky: 0}`"* | **no record on either prefix stores any derived liveness field at all** — `records_storing_a_derived_field: 0`, both prefixes | true on **2026-08-16**, fixed the same day. The claim outlived its defect. §4 |

---

## 1. The criterion — stated before measuring, and not a file count

C77 is the reason this section exists: a 115-clip run banked 115 well-formed, correctly-named,
correctly-counted records whose payload was **an error census with zero detections**, and it passed
verification because verification enumerated *containers and identity* and never evaluated *the
quantity the artifact exists to produce*.

A record `R` for fixture clip `C` is **COMPLETE** iff all four hold:

| | predicate | source of the rule |
|---|---|---|
| **P1** | `R` exists under the prefix, is non-zero-byte, parses, and `R["clip_id"] == C` | a record filed under the wrong name is not evidence about `C` |
| **P2** | `R["liveness"]` is a mapping carrying a non-empty `R["liveness"]["n_det"]` | `ph0_sam3.py:1131-1173` — `liveness_probe` returns `rec = {"concepts": cs, "n_det": out}`. Absent ⇒ an empty scene and a dead engine are **indistinguishable** for that clip |
| **P3** | `is_live(R["liveness"])` — i.e. **any** of `road`/`sky` > 0 | `ph0_sam3.py:955-978`. Recomputed from the counts here, never read from a stored flag |
| **P4** | **zero errors**: no `"error"` key in any `frames[*].det[*]` / `frames[*].scene[*]`, and no `liveness.errors` | `ph0_sam3.py:1293-1310` builds `n_err_total`/`err_kinds` for exactly this check |

⛔ **`n_det_total > 0` is deliberately NOT in the predicate.** Zero *agent* detections is a
legitimate **abstention** — an empty road has no car and no pedestrian (`ph0_sam3.py:146-152`, and
the `_note` banked by `main()`). Separating that from a dead engine is the entire purpose of the
road/sky control. Folding it in is what produces 77.

⚠️ **P4 is evaluated on the PAYLOAD, not on the summary keys.** This is not pedantry — see §3.2:
all 32 stale v1 records carry **no `n_err_total` and no `err_kinds` key at all**, so a consumer
that trusts the summary sees `{}` and reads 1 295 hard errors as "clean". A second reason: the
summary is lossy by construction — `ph0_sam3.py:1309` keys it on `str(d["error"]).split(":")[0]`,
so `err_kinds` records `RuntimeError` and discards which runtime error it was. The audit censuses
the **full** string.

**Reported beside the verdict, never folded into it** — the v2 run manifest's own extra gate:
**P5** `schema_version >= 2` and `engine.confidence_threshold == 0.25`. A v1-floor record is
present, non-empty, error-free and live *while being the wrong record*: a detection floor is
invisible in a payload, it shows up only as rows that are not there.

---

## 2. `sam3_backfill/` (v1) — 83 complete, 32 incomplete

```
records            115 / 115 vs fixture · missing 0 · extra 0 · zero-byte 0
run manifest       sam3_backfill/_runs/20260816-145005-sam3-backfill.json
n_frames_run       658
n_det_total        2 496          scene channel absent (v1 has none)
per concept        car 1260 · traffic sign 538 · traffic light 385 · pedestrian 180
                   · truck 97 · bus 26 · cyclist 10
liveness           control present 83 · absent 32 · live 83 · DEAD 0 · partial 2
stored `live` flag 0 records  (residue: none)
P5                 wrong_schema 115 · wrong_conf 115   (v1 stamps neither field)

COMPLETE           83 / 115
INCOMPLETE         32 / 115      reasons: P2_no_control 32, P4_errors 32
```

### 2.1 The error census — one distinct error, 1 295 occurrences

| count | error string |
|---:|---|
| **1 295** | `RuntimeError: mat1 and mat2 must have the same dtype, but got BFloat16 and Float` |

All 1 295 sit inside the 32 incomplete records. **Zero errors appear in any of the 83 records the
fixed engine produced.**

### 2.2 The 32 are exactly the known residual — set equality, not a count

MEASURED: the 32 clips my predicate fails are **set-equal** to
`…/2026-08-16-sam3-dtype-fix/raw/residual_32_clips.json` (`sorted(inc) == sorted(res)` → `True`).
Every one of the 32, uniformly:

| property | value |
|---|---|
| liveness control present | **0 / 32** |
| `n_det_total` | **0** on all 32 |
| `n_err_total` key present | **0 / 32** (absent) |
| `err_kinds` non-empty | **0 / 32** |
| payload errors | **1 295** across **185** frames |

They are the original C77 payloads: the free-Colab T4 was reclaimed three times and the last 32
clips were never re-run by the dtype-fix pass.

---

## 3. Where the inherited "77" came from — measured, not guessed

### 3.1 The 6 clips a wrong criterion discards

My auditor emits both numbers side by side, precisely so this cannot recur:

```
n_complete                    83
n_complete_AND_nonzero_det    77     <-- the inherited claim
n_complete_but_zero_det        6
```

Those 6 clips have a **live** road/sky control and **zero agent detections**: a genuinely empty
scene, correctly answered. Scoring them incomplete inverts the meaning of the control — it marks a
clip incomplete *for the engine having correctly reported nothing*, which is the failure the
control was added to make impossible. `zero_det_with_dead_control` is **0** on both prefixes: no
clip on either corpus has a zero that is an engine failure.

### 3.2 Why a summary-key check would have missed the real defect

The 32 stale records carry **no error-summary key at all** (§2.2). A completeness check reading
`err_kinds` finds `{}` on every one of them and concludes "no errors" — while the payload holds
1 295. This is C77's own shape at the level of the *verification*, and it is why P4 is specified
against the payload with the summary key only cross-checked.
(Cross-check result: `n_records_summary_disagrees_with_payload` = **0** on both prefixes — where a
summary key exists, it is truthful. The hazard is its **absence**, not its inaccuracy.)

---

## 4. The `24b6948f` question — settled by content: the defect is CLOSED

**What `live` was supposed to mean.** It is not supposed to exist. `ph0_sam3.py:1164-1167`, inside
`liveness_probe`, states it verbatim:

> ⛔ NO `live` BOOLEAN IS STORED. See `is_live` — the counts are the primitive, the verdict is
> derived at read time, and a derived field that is written down is a cache of a rule that has
> already changed once mid-corpus.

and `ph0_sam3.py:955-978` (`is_live`) is the only place the rule lives — `any(int(v) > 0 for v in
nd.values())`, reading `n_det`, with the docstring naming this very clip as the case that set the
rule (`all(...)` → `any(...)`).

**What is actually on the far side, MEASURED today, both prefixes:**

| | `sam3_backfill/` | `sam3_backfill_v2/` |
|---|---|---|
| `liveness` keys | `['concepts', 'frame_idx', 'n_det']` | `['concepts', 'det', 'frame_idx', 'n_det']` |
| stored `live` / `all_fired` / `alive` / `is_live` | **ABSENT** | **ABSENT** |
| `liveness.n_det` | `{road: 2, sky: 0}` | `{road: 2, sky: 0}` |
| `is_live()` recomputed | **True** | **True** |
| `n_det_total` | 22 | 44 |
| `liveness.errors` | none | none |

⇒ **Neither a stale-semantics bug nor a dead record.** The boolean is gone from the schema and gone
from the data; the record is live under the only rule that exists and carries real detections.

**How many OTHER clips have the same inconsistency: ZERO — and the class is empty, not merely
unpopulated.** `records_storing_a_derived_field` is **0 / 115** on *both* prefixes, so no record
stores a verdict that *could* disagree with its counts. The inconsistency was real on 2026-08-16
(`…/2026-08-16-sam3-dtype-fix/raw/strip_stale_live_flag.json`: 115 records, **81** carried the
derived field, **1** disagreed — this clip — `stripped: 81`, `dry_run: false`) and the strip is
confirmed applied by my independent read of all 230 records.

⚠️ **What remains true about `24b6948f`, and is NOT a defect:** it is a **partial control** —
`road 2 · sky 0`, an underpass with the sky occluded. Under `any` it is live, and it is reported
rather than hidden because it is the case that set the rule. Partial controls: **2** on v1
(`24b6948f`, `a6b2719b {road: 1, sky: 0}`), **1** on v2 — the v2 floor of 0.25 recovered
`a6b2719b`'s sky (`{road: 2, sky: 2}`), leaving only the genuine underpass.

⇒ **No patch to `ph0_sam3.py` is warranted.** The emitting code is already correct on this point,
and I did not modify it. §6 carries the one patch that *is* warranted, which is elsewhere.

---

## 5. `sam3_backfill_v2/` — 115 complete, 0 incomplete

```
records            115 / 115 vs fixture · missing 0 · extra 0 · zero-byte 0
run manifest       sam3_backfill_v2/_runs/20260816-223213-sam3-v2.json
n_frames_run       658                (identical frame set to v1 — same clips, same stride)
n_det_total        9 505   agent      n_scene_det_total  23 116
per concept        car 4269 · traffic sign 2496 · traffic light 1444 · pedestrian 738
                   · truck 430 · bus 90 · cyclist 38
per scene          road marking 10084 · lane marking 8776 · road curb 3140 · guardrail 1116
ERROR CENSUS       EMPTY  (0 errors, 0 distinct strings)
liveness           control present 115 · absent 0 · live 115 · DEAD 0 · partial 1
stored `live` flag 0 records
P5                 wrong_schema 0 · wrong_conf 0

COMPLETE           115 / 115
INCOMPLETE         0 / 115
```

The single zero-agent-detection clip is `566a3afd` — control **live** (`road 1 · sky 2`) and
**72 scene detections** on the same frames, so the frame is not empty, it simply holds no agents.
A correct abstention, not a gap.

This independently reproduces `raw/v2_census.json` and `raw/p3_run_manifest.json` to the unit,
from code that did not produce the data — a third derivation, after `hf_v2_census.py` and
`f3_homogeneity.py`.

---

## 6. The exact remaining work — none on the data; one consumer move

⛔ **No GPU work remains on this backfill.** Any re-run of the 32 residual clips would redo, at the
*wrong* detection floor, into a *superseded* prefix, work that v2 has already completed.
`SAM3_EXTRACTION_V2.md:530` states the intent — *"v2 supersedes it wholesale; the v1 prefix should
be marked superseded rather than repaired"* — and this audit supplies the measurement that the
supersession is now safe.

**The trigger condition written into the code is now satisfied, and the move has not happened.**
`colab/s2_lab_lib.py:59-70` says, in its own comment:

> consumers move when v2 is complete, which is one decision at one moment instead of a race.

v2 is complete: **115/115, MEASURED above**. Yet:

| site | current | risk |
|---|---|---|
| `colab/s2_lab_lib.py:58` | `BACKFILL_PREFIX = "sam3_backfill/"` | the v1 prefix is still the unmarked default name |
| `colab/nb_build.py:120-121` | production `BANK_PREFIX = L.BACKFILL_PREFIX` | **`SAM3_BACKFILL_115.ipynb` banks a new run into the superseded v1 prefix at the v1 floor** |
| `colab/RUNNER.md:140` | documents `sam3_backfill/<clip>.json` as the bank address | sends the next operator to v1 |

⚠️ **I do not own these files and did not edit them** (`colab/` is outside this package and another
stream may hold it). The precise patch, for the owner:

```diff
--- a/colab/s2_lab_lib.py
+++ b/colab/s2_lab_lib.py
@@
-BACKFILL_PREFIX = "sam3_backfill/"                 # in DS_LABELS (real runs)
+#: ⛔ SUPERSEDED 2026-08-23 — v2 is 115/115 complete by content
+#: (…/incoming/2026-08-23-sam3-completion-audit/raw/sam3_completion_audit.json).
+#: This prefix holds 83 complete records at floor 0.5 plus **32 C77 payloads**
+#: that carry NO error-summary key, so a consumer reading `err_kinds` sees {}
+#: and reads 1 295 dtype errors as clean. Kept for provenance (it is the
+#: primary source of the concept-reliability study); NOT a read target.
+BACKFILL_PREFIX_V1_SUPERSEDED = "sam3_backfill/"   # provenance only
```

```diff
--- a/colab/nb_build.py
+++ b/colab/nb_build.py
@@
 BANK_PREFIX = (L.SMOKE_PREFIX + 'sam3_backfill/') if SMOKE else \
-    L.BACKFILL_PREFIX
+    L.BACKFILL_V2_PREFIX
```

**The command that closes it** — no GPU, run from the dev box, and it is a *verification*, not a
re-run:

```
"C:/Users/Admin/venvs/tanitad/Scripts/python.exe" \
  "TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-sam3-completion-audit/code/sam3_audit.py" \
  --prefix sam3_backfill_v2/ --out /tmp/v2_recheck.json
```

Expected, and MEASURED today: `COMPLETE 115/115 · INCOMPLETE 0 · error census EMPTY ·
control live 115 · wrong_schema 0 · wrong_conf 0`.

⚠️ **Out of scope here but adjacent, and larger:** the fused perception layer is a **mixed-floor
corpus** — 115 clips at 0.25 (v2) and **86** at the vendor default 0.5 — union 201, sets disjoint
(`AUG120_REFUSE.md:40-46`). **No per-concept detection rate may be pooled across the 201.** That is
an 86-clip re-run (~43 GPU-min) and a PI spend decision. This audit neither measured nor changed it.

---

## 7. Reproduction

```
"C:/Users/Admin/venvs/tanitad/Scripts/python.exe" code/sam3_audit.py --out raw/sam3_completion_audit.json
```
Reads the HF token in place from the git-ignored `Keys.txt`, never to argv or stdout. Read-only HF
API — no metered compute was used and nothing was re-run. The dev-box TLS proxy needs
`truststore.inject_into_ssl()`, which the script does.

⚠️ **Environment note for the next agent.** The `G:` Google-Drive mount degraded mid-session:
`stat`/`listdir` kept serving cached metadata while **content reads failed** with `OSError errno 22`
/ *"Invalid request code"* — including `git` failing to read its own `commondir`. The same reads
succeeded seconds later. `code/sam3_audit.py` therefore retries every Drive read (`_read_bytes`,
40 attempts). **A single failed read on this mount is not evidence of a missing file.**

---

## 8. Deliverable manifest

Every row below was resolved with `git ls-files --cached <path>` **before this table was written**
(C78 — a manifest row is verified by resolving it, never typed from intent).

| artifact | path | staged |
|---|---|---|
| this report | `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-sam3-completion-audit/SAM3_COMPLETION_AUDIT.md` | ✅ |
| the auditor | `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-sam3-completion-audit/code/sam3_audit.py` | ✅ |
| raw audit output, 230 records, per-clip rows | `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-sam3-completion-audit/raw/sam3_completion_audit.json` | ✅ |
| run log | `TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-08-23-sam3-completion-audit/raw/audit_run.log` | ✅ |

**Far side, read-only, unmodified:** `Sayood/tanitad-ph0-aug120` → `sam3_backfill/` (115) and
`sam3_backfill_v2/` (115). This audit wrote nothing to HF.

**Staged, never committed, never pushed.**
