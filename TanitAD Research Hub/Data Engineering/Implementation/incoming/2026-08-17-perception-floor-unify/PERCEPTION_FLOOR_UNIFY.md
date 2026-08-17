# The aug120 mixed-floor defect: the diagnosis is PROVEN from the records, the corpus is instrumented so it cannot go mixed silently again — and the re-run is blocked on free-tier T4 capacity, not on code

**Package owner:** arch-inf agent, 2026-08-17 · branch `agent/arch-inf-20260803`
**Scope:** the **201-clip aug120 cohort's perception layer only.** ⛔ Nothing here is scaled to
the 4,472 build.

Every number is **MEASURED** on this run unless stamped otherwise, and carries its n and its
corpus. Raw: `raw/{f0_floor_probe,floor_homogeneity_manifest}.json`.

---

## 0. TL;DR

| | before | after this package |
|---|---|---|
| evidence for "the 86 are at floor 0.5" | **INHERITED** — *"floor known from the RUN, not from the record"* | ⭐ **MEASURED from the records**: min score **0.500100** over **2 939** detections, **0 below 0.50** |
| defects known to be in the 86 | **1** (the floor) | ⚠️ **6** — floor · no floor/schema stamp · no liveness control · no scene channel · no contours/extents · C85-flattened `rle_rows` |
| "is this clip done?" | two rules would be needed (HF far side / local bank) | **ONE** predicate, `census_records`, two thin transports |
| a mixed corpus is detectable | ⛔ **no** — invisible in the payload | ✅ **yes** — `raw/floor_homogeneity_manifest.json` + `stack/tests/test_perception_floor_homogeneity.py` |
| the 86 re-detected at 0.25 | — | ⛔ **NOT DONE — blocked on T4 capacity.** §5 |

⛔ **THE ESCALATION, IN THE HEADLINE — and it is a RESOURCE fact, not a code fact.**

> **The 86-clip re-run did not execute. Google Colab returned HTTP 503 Service Unavailable on
> all 54 T4 assignment attempts across ~100 minutes, and this account is entitled to NO other
> accelerator.** MEASURED, and the three responses are diagnostically different:
>
> | request | response |
> |---|---|
> | `colab new --gpu T4` | **503 Service Unavailable** on `…/assign?…&variant=GPU&accelerator=T4` — entitled, no capacity |
> | `colab new --gpu L4` / `--gpu A100` | *"Backend rejected accelerator … You may not have quota or entitlement"* — **not entitled at all** |
> | `colab new` (CPU) | **READY in seconds** — so the account, auth and CLI are all healthy |
>
> ⇒ There is **no alternative accelerator to escalate to** and no code change that unblocks it.
> Thor was not touched (it runs the live 30k S-W). **Everything else is complete and staged, the
> runner is symbol-preflighted and its transport is validated end-to-end on a live Colab session,
> and the residual is 86 named clips that resume with one command.** §5.3.

⭐ **AND THE DIAGNOSIS I WAS SENT TO ACT ON GOT STRONGER, NOT WEAKER, WHEN I CHECKED IT.** The
brief asked me to stop if the 86 turned out to be at 0.25 or to have a third provenance. They do
not — but the *evidence class* was upgradeable, and that is worth more than the re-run's
convenience: the previous package could only say the floor was known from the launching code.
It is knowable from the artifact. §1.

---

## 1. ⭐ The floor IS recoverable from the record — the previous package's caveat was too pessimistic

`…/2026-08-17-aug120-refuse/raw/inputs_manifest.json` states, verbatim:

> *"batch-pipeline SAM3 leg: vendor default confidence_threshold=0.5, pre-schema (stamps neither
> `schema_version` nor `engine.confidence_threshold`). **Floor known from the RUN, not from the
> record.**"*

That is an honest statement of a weak evidence class — and this programme does not spend a GPU-day
on an INHERITED claim. So it was re-derived from the only source that cannot go stale.

**The estimator, and it is one-sided.** `Sam3Processor` applies `keep = out_probs >
confidence_threshold` **inside** the vendor forward pass (`stack/scripts/ph0_sam3.py:404`). The
floor is therefore not merely "rows that are not there" — it is a **hard lower bound on every
surviving score**, and over thousands of detections the minimum converges onto it from above.

**MEASURED** (`code/f0_floor_probe.py`, `raw/f0_floor_probe.json`), n = 2 939 detections over the
86 clips:

| | v1 batch leg (86 clips) | v2 leg (115 clips) |
|---|---|---|
| detections | **2 939** | **9 505** |
| **min score** | **0.500100** | **0.250000** |
| detections below 0.50 | **0** | 6 081 |
| detections below 0.25 | 0 | **0** |
| median score | 0.7259 | — |
| **share in [0.25, 0.50)** | — | ⭐ **63.97 %** |

⇒ **`MIXED_FLOOR_CONFIRMED`.** The 86 are at 0.5, the 115 at 0.25, the legs are disjoint and their
union is exactly the 201-clip cohort (all re-asserted, not assumed).

⚠️ **What this test CANNOT do, stated rather than glossed:** a minimum of 0.5001 proves the floor
was **≤ 0.5001**. It does not prove it was exactly 0.5000. That distinction does not change any
decision here, and claiming more precision than the estimator has is how
`overlapping_holdout_se` got called a jackknife.

⭐ **The 63.97 % is the number that sizes the damage.** Nearly two thirds of what the v2 extraction
finds lives in the band the old floor discarded — so the 86 are not "slightly sparser", they are
missing the majority of what the same engine sees on the same corpus. ⚠️ **It is not a threshold
effect measured on one population**: the two legs differ in floor AND extraction AND clip
membership, and this figure is quoted for the **v2 leg only**, as a property of where v2's
detections sit — never as a v1-vs-v2 delta.

---

## 2. ⚠️ The floor was the defect that was NAMED. The 86 carry FIVE MORE

Reading the records rather than the ticket (`raw/f0_floor_probe.json`, n = 86):

| defect | measured | why it matters |
|---|---|---|
| detection floor 0.5 | 86/86 | the named one — blocks pooling |
| ⛔ **no liveness control** | **86/86** carry no `liveness` block | **their zeros are unreadable.** A zero-detection clip could be an empty road or a dead engine and nothing distinguishes them — C77's exact shape |
| no `schema_version` / `engine` stamp | 86/86 | the record cannot certify its own membership of any corpus |
| no scene channel | 86/86, `n_scene_det_total` 0 | no lane markings, curbs, guardrails, road markings, no `ego_lane` |
| no contours / oriented extents | 0 of 2 939 detections | detection keys are exactly `{box_xyxy, concept, mask_area_px, rle_rows, score}` |
| ⛔ `rle_rows` flattened (**C85**) | **all 2 939 — re-measured, see below** | cannot redraw its own mask; retracted in `SAM3_EXTRACTION_V2.md` §5 |

⇒ **The re-run closes six defects, not one**, which materially improves its value-per-GPU-minute.

⭐ **C85 IS RE-MEASURED ON THIS LEG RATHER THAN INHERITED, because "the retraction says so" is
exactly the evidence class this programme refuses.** Over all 86 clips (frame **448×179**,
uniform), 2 939 detections, **53 042 runs**:

| | v1 leg (86) | **v2 leg (115) — the control** |
|---|---|---|
| distinct row indices across every run | ⛔ **{0}** — all 53 042 | ✅ **179 distinct** = exactly the frame HEIGHT |
| runs whose END COLUMN exceeds the frame width (448) | ⛔ **52 996 / 53 042 = 99.9 %** | ✅ **0** |
| run lengths still sum to `mask_area_px` | ⚠️ **2 939 / 2 939 — 100 %** | — |
| `FLATTENED` verdict | **true** | **false** |

⭐ **The v2 leg is the positive control and it lands perfectly:** its runs span **179** distinct
rows on a 448×**179** frame — every row of the image — so the probe is demonstrably able to
distinguish a healthy encoding from a flattened one, and the defect is confirmed present on one leg
and absent on the other **in the same measurement**.

⇒ ⚠️ **And the third row is why nothing ever caught it:** the invariant anyone would have checked —
run lengths summing to the mask area — still holds at 100 % under the flattening. A serialiser that
accepted a shape it never asserted, sitting next to a reducer that was shape-agnostic and therefore
agreed with it.

---

## 3. One completion predicate, two transports — the change that made a no-push run possible

The 115 run used HF as **both** the durable bank and the resume state. This task may not push to
HuggingFace, and a Colab VM is reclaimed without warning, so the durable bank had to become the dev
box — which is a second place to ask *"is this clip done?"*.

⛔ **Two implementations of that question is how a corpus goes mixed while both checks report
green** — this census's own defect, one level up. So `colab/s2_lab_lib.py` was **split, not
duplicated**:

```
census_records(items, want, require_schema, require_conf)   # the PREDICATE, no transport
├── content_census(api, repo, prefix, …)                    # far side  (unchanged behaviour)
└── content_census_local(dirpath, …)                        # dev box   (new)
```

⭐ **The refactor is verified by REPRODUCTION, not by inspection.** Run over the banked 115, the
local adapter returns the numbers `SAM3_EXTRACTION_V2.md` §6.1 published — to the unit:

> records **115** · agent detections **9 505** · scene detections **23 116** · error census
> **EMPTY** · liveness live **115/115** · zero-det clips **1** · `wrong_schema` **0** ·
> `wrong_conf` **0** · per concept car 4 269 · traffic sign 2 496 · traffic light 1 444 ·
> pedestrian 738 · truck 430 · bus 90 · cyclist 38 · per scene 10 084 / 8 776 / 3 140 / 1 116

A second, independent re-derivation of that table by code that did not produce it.

---

## 4. The homogeneity pin — the invariant, and the instrument that enforces it

**The corpus cannot be the pin**: ~24 MB of records, not in git (neither v1 nor v2 ever was), and a
unit test has no network. So the repo holds the **manifest** — per clip its floor, schema, md5,
detection counts and liveness verdict, and at the top the **distinct sets**.

`raw/floor_homogeneity_manifest.json` (`code/f3_homogeneity.py`) · `stack/tests/test_perception_floor_homogeneity.py`
(**12 tests**) pins **both halves**:

**The data** — `len(distinct_confidence_thresholds) == 1`. That is the invariant the corpus was
built violating, and the one nothing could see. Plus: every record stamps floor **and** schema;
liveness live on all; error census empty; and ⚠️ **the manifest cannot lie about its own
coverage** (`covers_cohort` is re-derived in the test, and a corpus short of the cohort is
*required* to name its residual).

**The detector** — pinning only the data would leave the corpus defended by an instrument that can
itself rot. So `census_records` is fed deliberately broken corpora and must refuse them: a record
at the wrong floor (`wrong_conf == 1`, `pass_` False); a pre-schema record stamping nothing — **the
86's exact shape**; a dead liveness control; an error entry; a filename/content mismatch. With a
**positive control for the controls** (a clean corpus must still pass, or a predicate that refused
everything would look identical), and a check that both public censuses still *delegate* to the
shared one rather than having been forked back apart.

### 4.1 ⭐ The pin was tested against a corpus deliberately shaped like the real defect

Asserting that an instrument *would* catch something is the weakest claim in this report's family,
so it was executed. A synthetic 201-clip corpus was assembled — the 115 real v2 records plus 86
stubs standing in for un-re-detected clips — and put through all three gates. **MEASURED, all
three refuse, and each names the defect rather than dying obscurely:**

| gate | verdict on the mixed corpus |
|---|---|
| `code/f3_homogeneity.py` | `HOMOGENEOUS **False**` · floors `['0.25', 'None']` · schemas `['2', 'None']` · `wrong_conf` **86** · `wrong_schema` **86** · `records_without_control` **86** · zero-split **1 empty / 86 dead** |
| `code/f4_build_inputs_unified.py` | ⛔ refuses: *"the corpus is MIXED-FLOOR: ['0.25', 'None']. No per-concept rate may be pooled across it and nothing fused from it is attributable."* |
| `…/2026-08-17-aug120-refuse/code/build_inputs.py` (untouched) | ⛔ refuses: *"86 clips appear in BOTH SAM3 legs — the floors would be silently mixed PER CLIP and the pick would be arbitrary"* — **§6's claim, now MEASURED rather than read off the source** |

⇒ The corpus cannot reach a fuser through any of the three paths while it is mixed.

⚠️ **Coverage is reported as XFAIL-with-its-residual-named while the corpus is incomplete**, not as
a red suite: "one floor" is an invariant the code controls, "all 201 clips" is a programme goal
gated on a GPU. It becomes an ordinary hard assertion the moment the 86 land — nothing has to be
remembered or un-marked. ⛔ **That is not a licence to leave it xfailed**, which is why it is in
this section and in §7.

---

## 5. The run — what is built, what is proven, and exactly what did not happen

### 5.1 ⛔ It did not execute. The reason is capacity, and it is measured

Across the session, **every** `colab new --gpu T4` returned
`ColabRequestError: … assign?…&variant=GPU&accelerator=T4: **Service Unavailable**` — **54
attempts** (14 + 16 + 10 + 11 across four independent retry loops, plus 3 isolated diagnostic
calls), spanning **~100 minutes** of wall-clock, with **zero** successes and **not one** distinct
error message. The three-way probe in §0 separates capacity from entitlement from account health,
so this is not a guess about which of the three it is.

⛔ **Thor was not touched** — it runs the live 30k S-W. ⛔ **No paid accelerator was started**:
provisioning and spend are the PI's, and in any case L4/A100 are not entitled on this account.

### 5.2 What IS proven about the runner, so the GPU minute is not spent discovering a typo

| check | result |
|---|---|
| **transport end-to-end** on a live Colab session | ✅ upload → exec → tar → download: VM reported **296 B**, dev box received **296 B**, 3/3 files byte-exact |
| **symbol preflight** of every `L.*` / `ph0_sam3.*` / `ph0_pilot.*` attribute `f1_run86.py` touches | ✅ **18/18 present** |
| all six scripts compile | ✅ |
| `f5_refuse_delta.py` against the banked corpus | ✅ reproduces **10 464 tracks**, **115** scene clips, and token visibility **g_str 201 · a_str 201 · a_tac_lat 188 · a_tac_lon 147** — the last two matching `AUG120_REFUSE.md` §1.1's null counts (13, 54) exactly |

⭐ **The preflight exists because of a named trap:** an analysis-time import that fails *after* the
rollout destroys a run whose compute is already paid for. Here that would be ~43 GPU-min of a
scarce free-tier T4.

### 5.3 The residual, named and resumable

**86 clips**, listed in `raw/floor_homogeneity_manifest.json` → `residual` (and echoed by the
xfail message). One command, content-resumable, safe to re-run any number of times:

```
bash <scratchpad>/acquire_and_run.sh            # retries T4, ships, then drives
# or, with a session already up:
python code/f2_drive86.py --aug120 <aug120> --v2-dir <v2> --bank <bank> \
                          --session tanitad-floor86 --chunk 22 --ship
```

**Why it survives a reclaim.** The dev box is the resume authority: before every chunk the driver
recomputes the done-set **by content** (`content_census_local`, the §3 predicate — liveness present
AND zero errors AND schema ≥ 2 AND floor == 0.25), ships only the outstanding clips, and pulls the
whole bank back after each chunk. A reclaim costs **one chunk**, never the run. A chunk that adds
no complete clips is counted a **stall** and aborts after two, rather than spinning to the timeout
and presenting a partial run as an interruption.

⚠️ **One latent bug was fixed on the way in.** `bridge_batch` copies the side files
(`_geometry.json`, `_v2manifest.pt`) of `loc[batch[0]]`'s segment **only**, so a batch spanning two
segments bridges the second against the first's geometry. The 115 run's todo happened to be
segment-ordered; this residual list is not. `f1_run86.py` therefore batches **by segment**. Free,
and it removes a confound that would have been invisible in the output.

### 5.4 Budget, so the next attempt is sized rather than guessed

**INHERITED (MEASURED in `SAM3_EXTRACTION_V2.md` §4, 5-clip pilot):** 29.82 s/clip wall on a T4 at
schema v2 / floor 0.25 ⇒ **86 × 29.82 s ≈ 42.7 GPU-min**, matching the brief's ~43. Peak GPU
4.241 GB — a T4's 16 GB is not the constraint. Output ≈ 86 × 120 KB ≈ **10.3 MB**, so the
per-chunk pull is ~2.6 MB and the transport is not a constraint either.

---

## 6. The re-fuse — assembled and asserted, gated on §5

⛔ **`…/2026-08-17-aug120-refuse/code/build_inputs.py` WILL REFUSE THIS INPUT, AND THAT IS THE
SCRIPT WORKING.** It exists to *merge* two SAM3 legs and asserts they are disjoint. Once the 86 are
re-detected, all 201 are in the v2 corpus and the batch leg becomes a strict **subset** — overlap
86, assertion fires. With both legs holding the same clip at two different floors, picking one
silently is precisely the defect it was written to prevent. **It is left untouched and its refusal
is reported as a result**, not defeated — and **MEASURED, not predicted** (§4.1).

`code/f4_build_inputs_unified.py` replaces it with a **stricter** gate, because one leg admits
assertions a merge never could: union == cohort exactly, no clip in two source dirs, and — before
anything is fused — **every** record at floor 0.25, schema ≥ 2, liveness live, zero errors. Its
output directory is named `sam3_refuse/` so that **`refuse_run.py`, `analyze_refuse.py` and
`pi_check.py` run completely unchanged**.

`code/f5_refuse_delta.py` then verifies by content, with **two predictions committed in advance**:

* **P1 — the 115 clips' fused records are byte-identical before and after.** Their SAM3 input, the
  fuser and the Engine A sidecar are all unchanged. This is the **control**: if any of them moves,
  the 86's delta is not attributable and the whole comparison is void.
* **P2 — not one of the four label families moves on any of the 201.** `AUG120_REFUSE.md` §1
  MEASURED that swapping 115 clips from absent perception to full v2 perception changed **zero**
  `g_str`/`a_str`/`a_tac_lat`/`a_tac_lon`, because `emit_vocab` reads the VLM, Alpamayo and
  Engine A while SAM3 reaches only `corroborate()` and the census. If that structural claim holds,
  re-detecting the 86 must also move zero. ⚠️ **A non-zero result would REFUTE the orthogonality
  claim** — it would not be a perception win — and that is the more interesting outcome.

⛔ **P2 carries a positive control for itself.** "Zero tokens changed" is *also* what a comparison
reading the wrong field reports. The four families live under **`vocab.<family>.token`**, not at
top level — MEASURED on a banked record — and `f5` **refuses to report P2** unless it can see real
tokens. Against the banked corpus it sees 201/201/188/147, so the reader is live.

⚠️ **What will NOT be presented:** a v1-vs-v2 per-concept delta as a threshold effect. The 86
change floor **and** extraction at once, on a population 2.63× larger in detections. What the
re-fuse buys is stated as a **capability** — that a per-concept rate becomes poolable across the
201 at all — with per-leg numbers kept labelled by the leg that produced them.

---

## 7. Escalations — decisions, not notes

1. ⛔ **The 86-clip re-run needs a T4 and free-tier capacity is 503.** No code fix applies; no other
   accelerator is entitled on this account. **Options, all the PI's:** (a) retry later — capacity is
   transient and the job is one command; (b) authorise Colab Pro / compute units, which also unlocks
   L4/A100; (c) authorise a different GPU host. **Until it closes, no per-concept perception rate may
   be pooled across the 201** — the constraint `AUG120_REFUSE.md` escalation 2 raised is still open.
2. ⛔ **Nothing was pushed to HuggingFace, and the fused v2 corpus is still dev-box-only.** That is
   `AUG120_REFUSE.md` escalation 1, unchanged and still needing the PI's authorisation. This package
   adds per-clip md5s (`floor_homogeneity_manifest.json` → `clips[].md5`) so any far side can be
   verified against the repo whenever a push is authorised. ⚠️ **Datum for that decision:**
   `s2_lab_lib.ensure_repo` creates `Sayood/tanitad-ph0-aug120` with `private=True`, so the target is
   a *private* dataset — which may or may not change the PI's answer, but should be in front of them.
3. ⚠️ **The coverage test is XFAIL, not red, and must not be allowed to calcify.** It flips to a hard
   assertion automatically when the 86 land. If the re-run is abandoned rather than completed, the
   honest action is to delete the goal, not to leave an xfail standing in for it.
4. ⚠️ **`COLAB_CLI_MCP.md` §8's *"spell T4 EXACTLY (else it falls back to A100)"* does not hold on
   this account** — A100 is rejected for entitlement. Owner: the colab-tooling package.
5. ⚠️ **The 86's `rle_rows` are C85-flattened** and will stay so until the re-run lands; the decode
   rule (`row = start // W`) in `as_2d_mask`'s docstring is the interim path.

---

## 8. Deliverable manifest

⚠️ Everything is **in the repo and staged** on `agent/arch-inf-20260803` except the rows marked.

| artifact | where |
|---|---|
| the shared C77 predicate + local transport | `colab/s2_lab_lib.py` (`census_records`, `content_census_local`) |
| ⭐ the homogeneity pin, 12 tests (data **and** detector) | `stack/tests/test_perception_floor_homogeneity.py` |
| this report | `…/incoming/2026-08-17-perception-floor-unify/PERCEPTION_FLOOR_UNIFY.md` |
| floor diagnosis from the records | `code/f0_floor_probe.py` → `raw/f0_floor_probe.json` |
| the 86 runner (VM) + chunk driver (dev box) | `code/f1_run86.py`, `code/f2_drive86.py` |
| homogeneity manifest builder | `code/f3_homogeneity.py` |
| ⭐ per-clip floor · schema · **md5** · counts · liveness | `raw/floor_homogeneity_manifest.json` (115 rows + the 86 named as residual) |
| single-leg input assembly for the re-fuse | `code/f4_build_inputs_unified.py` |
| re-fuse content verification + the two predictions | `code/f5_refuse_delta.py` |
| ⛔ **the 86 re-detected records** | **DOES NOT EXIST** — §5.1. Resume: `code/f2_drive86.py` |
| ⚠️ acquire-and-run helper | **SCRATCHPAD ONLY** — `<scratchpad>/acquire_and_run.sh`; it hard-codes dev-box paths, and §9 is the portable form |
| ⚠️ the banked 115 v2 corpus | **DEV BOX + HF** `Sayood/tanitad-ph0-aug120 → sam3_backfill_v2/` (pre-existing, **unmodified** by this package) |
| ⚠️ the fused corpus | **DEV BOX ONLY**, unchanged from `…/2026-08-17-aug120-refuse/` |

**Suites**, both, named interpreter (`PYTHONUTF8=1 OMP_NUM_THREADS=6 …/python.exe -m pytest -q`) —
see §10 for the paired numbers.

---

## 9. Reproduce

⚠️ **THE TWO INPUTS ARE A DEV-BOX CACHE, SO HERE IS HOW TO REBUILD THEM.** Every step below takes
`<aug120>` and `<sam3_v2>` as given; on a fresh machine they do not exist, and a resume that cannot
reconstruct its own inputs is stranded work by a slower route. **Both rebuilders are already banked
in the repo** (checked, not assumed) — they are simply in other packages, so they are named here:

| input | holds | rebuild with |
|---|---|---|
| `<aug120>` | `merged/ph0_v2.json` (the 201-clip cohort), `merged/sam3.json` (the v1 leg), `ego/`, `aux/records.parquet` | `…/2026-08-15-aug120-fusion/code/hf_pull_labels.py` then `…/hf_pull_ego.py` |
| `<sam3_v2>` | the banked 115 v2 records | `…/2026-08-17-aug120-refuse/code/pull_v2.py --out <sam3_v2>` |

Both are **pulls** from `Sayood/tanitad-ph0-aug120` (a *private* dataset) — reads, not writes.

```
# --- 0 GPU: the diagnosis, from the records --------------------------------- #
python code/f0_floor_probe.py --aug120 <aug120> --v2-dir <sam3_v2> \
                              --out raw/f0_floor_probe.json

# --- the re-run (needs a T4) ------------------------------------------------- #
colab new -s tanitad-floor86 --gpu T4
python code/f2_drive86.py --aug120 <aug120> --v2-dir <sam3_v2> \
                          --bank <bank> --session tanitad-floor86 \
                          --chunk 22 --ship          # resumable; re-run freely
colab stop -s tanitad-floor86                        # an unstopped session burns units for 24 h

# --- 0 GPU: pin, re-fuse, verify --------------------------------------------- #
python code/f3_homogeneity.py --corpus <sam3_v2> <bank> --aug120 <aug120> \
                              --out raw/floor_homogeneity_manifest.json
python code/f4_build_inputs_unified.py --corpus <sam3_v2> <bank> \
                              --aug120 <aug120> --work <work>
python "…/2026-08-17-aug120-refuse/code/refuse_run.py" --work <work> \
       --aug120 <aug120> --stack <repo>/stack --arms A2,A3 \
       --engine-a "…/2026-08-16-s2-v1-labels/labels/engine_a_aug120.jsonl"
python code/f5_refuse_delta.py --before <old A3> --after <work>/fused_aug120_v2 \
       --leg86 <bank> --out raw/f5_refuse_delta.json \
       --index-out raw/fused_aug120_v3_index.jsonl
```

⚠️ `PYTHONUTF8=1` is required for every `colab` invocation — colab-cli 0.6.0 opens the script with
the locale codec (cp1252 here) and any file carrying a ⛔/⚠️ dies before a line reaches the VM.

---

## 10. Suites — paired across my edit, and nothing else's

`cd <suite> && PYTHONUTF8=1 OMP_NUM_THREADS=6 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -m pytest -q`

| suite | brief's baseline | measured here |
|---|---|---|
| `stack` | **3770** passed / 0 failed | **3781 passed / 0 failed / 7 skipped / 3 xfailed** in 369 s |
| `taniteval` | **1092** / 0 failed | **1092 passed / 0 failed** — untouched by this package, re-run to prove it |

⭐ **+11 passed and +1 xfailed = exactly my 12 new tests, and the arithmetic closes:**
3770 + 12 = 3782 = 3781 passed + 1 xfailed. Nothing else in the tree moved across my edit, which is
the check worth having when four agents are live in one working tree — an unpaired total would
attribute their work to this package.

⚠️ The **named interpreter** matters: a bare `pytest` resolves to a different venv on this shell and
reports *"191 errors during collection"* while exiting 0. And without `PYTHONUTF8=1` this shell
reports four failures that do not exist (C84).
