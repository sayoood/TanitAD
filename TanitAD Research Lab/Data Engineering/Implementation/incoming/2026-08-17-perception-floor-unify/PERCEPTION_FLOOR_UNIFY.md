# The aug120 mixed-floor defect: diagnosed from the records, instrumented so it cannot recur silently — and now CLOSED, because the re-run that free-Colab could not schedule fits on the dev box's own GPU

**Package owner:** arch-inf agent, 2026-08-17 · branch `agent/arch-inf-20260803`
**Closed out:** local-GPU run, same day, `§5.5`
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
| the 86 re-detected at 0.25 | — | ✅ **DONE — 86/86, on the dev box's own GPU.** §5.5 |
| the aug120 perception layer | ⛔ **MIXED** — two floors, two schemas | ⭐ **UNIFIED**: 201/201, floors `['0.25']`, schemas `['2']`, live 201, dead 0, errors 0 |
| the coverage test | ⚠️ **XFAIL** with its residual named | ✅ **HARD PASS** — 12 passed, 0 xfailed |

⭐ **THE HEADLINE IS NOW THE CLOSE-OUT, AND THE ESCALATION IT REPLACES IS KEPT BELOW BECAUSE THE
HISTORY IS THE LESSON.**

> **The 86 are re-detected at floor 0.25 / schema v2, and the corpus is UNIFIED.** MEASURED on the
> dev box's **NVIDIA RTX 4060 (Ada, sm_89)**, torch 2.11.0+cu128: **86/86 complete**, **7 478**
> agent + **17 618** scene detections, **liveness live 86/86**, **0 errors**, `wrong_conf` **0**,
> `wrong_schema` **0**, in **2 612 s** (**30.16 s/clip**) at peak **4.24 GB**.
> ⇒ Union over the cohort: **201/201**, `distinct_confidence_thresholds` **`['0.25']`**,
> `distinct_schema_versions` **`['2']`**, `UNIFIED` **true**, residual **0**;
> and the fuser reports `perception_engine_mixed` **false** with a single engine row
> (schema 2 · floor 0.25 · 201 clips). `raw/f7_run86_local.json`, `raw/floor_homogeneity_manifest.json`.

⚠️ **THE COLAB BLOCKER WAS REAL AND IS NOT RETRACTED — it was ROUTED AROUND, not refuted.** Free
Colab returned **503 Service Unavailable on all 54 T4 assignment attempts across ~100 minutes**,
and the account is entitled to no other accelerator. The three responses are diagnostically
different and that separation is what made "route to another host" the right move rather than a guess:

| request | response |
|---|---|
| `colab new --gpu T4` | **503 Service Unavailable** on `…/assign?…&variant=GPU&accelerator=T4` — entitled, no capacity |
| `colab new --gpu L4` / `--gpu A100` | *"Backend rejected accelerator … You may not have quota or entitlement"* — **not entitled at all** |
| `colab new` (CPU) | **READY in seconds** — so the account, auth and CLI are all healthy |

⭐ **The lesson is the one the package had already half-learned: ACQUISITION AND EXTRACTION ARE
SEPARABLE.** `f1_run86.py` was made host-agnostic (every `/content/...` literal became an `F86_*`
env var defaulting to the Colab value), so **the same extraction file ran unmodified on the 4060**.
⛔ **No second extraction path was written** — that would have rebuilt the exact heterogeneity this
package exists to remove, one level up. The dev box needed only a **driver** (`f7_drive86_local.py`),
and it reuses `f2`'s population computation and `census_records` rather than re-implementing either.
⛔ Thor was not touched (live 30k S-W). ⛔ Nothing was pushed to HuggingFace.

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

✅ **THE XFAIL HAS FLIPPED, EXACTLY AS DESIGNED AND WITH NOTHING TO REMEMBER.** Coverage was
reported as XFAIL-with-its-residual-named while the corpus was incomplete — "one floor" is an
invariant the code controls, "all 201 clips" was a programme goal gated on a GPU. The 86 landed and
`test_the_perception_layer_covers_the_whole_cohort` became an ordinary **hard pass** on the next
run, with **no marker edited and no test rewritten**:

```
tests/test_perception_floor_homogeneity.py .............  12 passed in 0.58s
```

⭐ **12 passed / 0 xfailed** (was 11 passed / 1 xfailed). That an xfail can retire itself when the
world changes — rather than needing a human to notice — is the property worth copying.

---

## 5. The run — what was built, what was proven, and where it finally executed

### 5.1 ⛔ It did not execute ON COLAB. The reason is capacity, and it is measured

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

⭐ **AND THE ESTIMATE HELD, ON DIFFERENT SILICON.** MEASURED on the 4060: **30.16 s/clip** over the
85-clip batch against the T4's **29.82** (+1.1 %), and peak **4.24 GB** against **4.241** — so the
budget was transferable and the T4 figure was not a T4 artifact.

### 5.5 ⭐ Where it actually ran — the dev box's own GPU, same file, same knobs

⛔ **THE EXTRACTION FILE WAS NOT FORKED.** `code/f1_run86.py` is host-agnostic (`F86_ROOT`,
`F86_REPO`, `F86_OUT`, `F86_WORK`, `F86_TODO`/`F86_TODO_INLINE`, `F86_TAR`), so the dev box ran
**the same bytes Colab would have run**, with `F86_TAR=""` because the bank is already durable
here. The only new file is a **driver**, `code/f7_drive86_local.py`, which lifts `f2`'s population
computation and calls `content_census_local` for resume — never a second predicate.

**MEASURED, `raw/f7_run86_local.json`** · host NVIDIA RTX 4060, sm_89, 8.59 GB, torch 2.11.0+cu128:

| | the 86 | the union (201) |
|---|---|---|
| records complete | **86 / 86** | **201 / 201** |
| agent detections | **7 478** | **16 983** |
| scene detections | **17 618** | **40 734** |
| liveness live / dead / no-control | **86 / 0 / 0** | **201 / 0 / 0** |
| error census | **EMPTY** | **EMPTY** |
| `wrong_conf` · `wrong_schema` | **0 · 0** | **0 · 0** |
| distinct floors · schemas | `['0.25']` · `['2']` | **`['0.25']` · `['2']`** |
| zero-detection split | **2 empty scene / 0 dead** | **3 empty scene / 0 dead** |
| wall · s/clip · peak GPU | **2 612 s** · **30.16** · **4.24 GB** | — |

per concept (the 86): car **3 207** · traffic sign **2 373** · traffic light **1 033** ·
pedestrian **462** · truck **298** · bus **102** · cyclist **3**
per scene (the 86): road marking **7 790** · lane marking **6 741** · road curb **2 188** ·
guardrail **899**

⭐ **THE PROOF-GATE FIRED FIRST, AND IT IS A DETECTION — NOT AN ABSENT TRACEBACK.** The C77 dtype
fix was developed on a **T4 (sm_75)**; this GPU is **Ada (sm_89)**, so "it did not crash" would have
been worth nothing — 115 structurally perfect EMPTY records is exactly what C77 banked. The driver
therefore ran **one** clip and refused the batch unless that record came back live. It came back
`road 4 · sky 1`, **319** agent + **320** scene detections, **0** errors, peak **4.23 GB**, and
`dtype_fix.applied` **true** via `sam3.model.vitdet.addmm_act`. Only then did the other 85 start.

⚠️ **THE ZEROS ARE READABLE, WHICH WAS HALF THE POINT.** Three clips in the union hold zero agent
detections and **all three carry a LIVE road/sky control** (`566a3afd` road 1/sky 2, `6149267e`
road 1/sky 1, `922bb1c8` road 4/sky 1) ⇒ **empty scenes, not a dead engine.** Under the old records
that distinction did not exist.

### 5.6 ⚠️ Three environment defects were hit on the way, and only one of them is interesting

None is a model, GPU or dtype fault, and all three are recorded because each cost a run:

1. ⛔ **The gated-weights 401 — the vendor fetches its own checkpoint and does NOT pass a token.**
   `sam3.model_builder.download_ckpt_from_hf` calls `hf_hub_download(repo_id, filename)` bare, so it
   authenticates only from the ambient environment. Every one of *our* HF reads goes through
   `L.hf_download`, which passes the token explicitly — so the run got all the way through the v2
   labels and the shard index and died **only** at the weights with
   `GatedRepoError: 401 … Access to model facebook/sam3 is restricted`. Colab hid this because the
   notebook environment already carried `HF_TOKEN`. ⇒ the driver puts `HF_TOKEN` in the child's env
   (read in place from `Keys.txt`, never printed, never in argv).
2. ⭐ **THE ONE WORTH KEEPING — `import pyarrow.dataset` AFTER torch/sam3 SEGFAULTS THIS BOX.**
   MEASURED: the run died at **rc 3221225477 = 0xC0000005 ACCESS_VIOLATION** with faulthandler
   putting the fault exactly on the **lazy** `import pyarrow.dataset` that `pandas.read_parquet`
   performs inside `v2_to_pilot.pick_clips`. A/B, **both outcomes fixed in advance**:

   | order | result |
   |---|---|
   | pyarrow imported **after** torch + sam3 + triton + hf | ⛔ **rc 139, segfault** |
   | pyarrow imported **first** (a `sitecustomize` shim on `PYTHONPATH`) | ✅ **rc 0**, `read_parquet` → **23 644 rows** |

   It is a **DLL load-ORDER conflict**, not a data or GPU fault. ⛔ Fixed in the DRIVER's child
   environment, **not** in `f1_run86.py`, so the Colab path stays byte-identical — Linux does not
   have the conflict.
3. ⚠️ **A CRASH WITH ZERO OUTPUT IS A BUFFERING BUG, NOT A MYSTERY.** The first two failures printed
   **nothing at all** — the child's stdout is a pipe, so Python block-buffers it, and only the
   `[bank]` lines carry `flush=True`; `[cfg]`, `[v2]`, `[assets]`, `[sam3] up` all died inside an
   unflushed 8 KB buffer and the log looked like the run had never started. ⇒ `PYTHONUNBUFFERED=1`
   on the child. **Same family as the self-matching monitor in `CLAUDE.md`: an instrument that
   cannot see the failure it exists to report.**
   ⚠️ It also caused **my own misdiagnosis, logged rather than quietly dropped**: with no output I
   attributed the first crash to `hf_xet` buffering the 3.45 GB checkpoint in RAM. That was
   **WRONG** — a later re-fetch returned the file from cache in **1 s** and `build_processor` loaded
   it in **8 s** at peak **3.575 GB**, so the download had in fact succeeded. The refusal message was
   also fixed to distinguish *"no record produced"* (a run failure) from *"record with a dead
   control"* (the C77 shape), because printing the second for the first is the
   symptom-read-as-root-cause trap.

### 5.7 ⛔ The dev box's torch was the thing most worth NOT breaking, and it is MEASURED intact

`CLAUDE.md`: *"`uv pip install <anything>` CAN SILENTLY REPLACE TORCH WITH A WHEEL THE DRIVER
CANNOT RUN"* — MEASURED twice on pod4, where `accelerate` and then `compressed-tensors` each
dragged torch off the default PyPI index and broke CUDA for every job on the box. **Neither command
named torch.** This GPU is load-bearing (the frozen-trunk probes, and the only GPU not running the
live 30k on Thor), so the protection had to be **structural, not procedural**:

⭐ **sam3's whole closure went into a SEPARATE venv** (`C:/Users/Admin/venvs/sam3run`) that reaches
torch through **one `.pth` line** pointing at the tanitad venv's `site-packages`. torch,
torchvision, numpy and av therefore resolve to **the same files the probes use** — there is no
second torch to go wrong, nothing was re-installed, and every `pip` write landed in `sam3run`.
⇒ It does not depend on anyone remembering `--no-deps` on a future command, and **`rm -rf` of one
directory restores the box exactly**. (`--no-deps` was used on every install regardless.)

**MEASURED after the run** (`raw/f7_env_local_gpu.json`, `code/f8_env_probe.py`), both interpreters,
verified with a **real `conv2d` on CUDA** — not `import torch`, not `is_available()`, because cuBLAS
can succeed while cuDNN/conv is broken:

| | baseline (before any install) | after |
|---|---|---|
| `tanitad` torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 | ✅ **identical** |
| `tanitad` cuDNN | 91900 | ✅ **91900** |
| `tanitad` real CUDA conv2d · matmul | — | ✅ **both pass** |
| **`TANITAD_UNDISTURBED`** | — | ⭐ **true** (drift set empty) |
| `sam3run` torch | — | resolves **from tanitad**, conv2d passes |

⚠️ **ONE STORED ASSUMPTION IS REFUTED, and it is worth propagating: "no Triton on Windows".**
MEASURED today — `triton-windows` 3.7.1 installs on py3.13 and **JIT-compiles and runs a real kernel
on the 4060 with no MSVC present** (`max err 0.0`). ⛔ **It was not optional here:** SAM3's IMAGE
path reaches `nms_triton` (`perflib/nms.py::generic_nms` takes the CUDA branch when
`torch_generic_nms` is absent), so a stub would have broken or silently altered NMS. ⚠️ It was
installed **only into `sam3run`**, so `torch.compile` in the tanitad venv is unaffected and the
existing "inductor is unusable here" note still stands for that venv.

---

## 6. The re-fuse — EXECUTED, and both pre-registered predictions HELD

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

### 6.1 ⭐ The result — MEASURED, `raw/f5_refuse_delta.json`

`f4` gate passed (201 records, single leg, floor 0.25, schemas `['2']`), then `refuse_run.py` ran
arms **A2** and **A3** unchanged. The fuser's own summary is the union check this package pins:

```
n_fused 201 · n_v2 201 · n_sam3 201 · sam3_missing 0 · with_scene_channel 201
perception_engines [{schema_version: 2, confidence_threshold: 0.25, n_clips: 201}]
perception_engine_mixed  FALSE
corroborated 88 · conflicts 10 · with_alpamayo 201
```

⭐ **`perception_engines` is a ONE-ROW list and `perception_engine_mixed` is `false`.** That is the
invariant the corpus was built violating, now asserted by the fuser itself rather than by a report.

**Both predictions, committed in advance in §6, HELD:**

| prediction | outcome |
|---|---|
| **P1** — the 115's fused records byte-identical before/after | ✅ **HOLDS — 0 of 115 moved.** The control is clean, so the 86's delta is attributable |
| **P2** — no label family moves on any of the 201 | ✅ **HOLDS — 0 changed** in `g_str`, `a_str`, `a_tac_lat`, `a_tac_lon` (and 0 within the 86 specifically) |

⚠️ **P2 holding is the BORING outcome and that is the honest reading.** It CONFIRMS the structural
orthogonality claim — `emit_vocab` reads the VLM, Alpamayo and Engine A; SAM3 reaches only
`corroborate()` and the census — and it is worth exactly that and no more. It is **not** evidence
that the re-run was pointless: the label families were never the channel the perception layer feeds.

⭐ **What the re-run actually bought, stated as the capability it is:** pooled over the unified 201,
`f5` reports **14 335 tracks · 0 absent · 201 scene clips · floors `{'0.25': 201}`** — against the
**10 464 tracks / 115 scene clips** the 115-leg alone could offer. **A per-concept rate is now
poolable across the whole cohort at all**, which it provably was not before.

## 7. Escalations — decisions, not notes

1. ✅ **CLOSED — the 86-clip re-run is done.** It needed a GPU, not a T4. Free-tier capacity was
   503 on all 54 attempts and no other accelerator is entitled, so the job was routed to the **dev
   box's own RTX 4060**, where `f1_run86.py` ran unmodified in **43.5 min** at peak **4.24 GB**.
   ⭐ **The constraint `AUG120_REFUSE.md` escalation 2 raised — *"no per-concept perception rate may
   be pooled across the 201"* — is now LIFTED**: the corpus is single-floor, single-schema, 201/201.
   ⚠️ **The general lesson for the PI, and it is worth more than this package:** the programme has a
   second usable GPU that was not in the scheduling picture. **Any job under ~7 GB VRAM that is not
   Thor-bound can run here**, and free-Colab capacity is not a hard dependency.
2. ⛔ **STILL OPEN AND STILL THE PI'S — nothing was pushed to HuggingFace.** The unified 201-clip
   corpus and the re-fused output are **DEV BOX ONLY**. This package banks per-clip md5s
   (`floor_homogeneity_manifest.json` → `clips[].md5`, and `<bank>/_md5.json` for the 86) so any far
   side can be verified against the repo whenever a push is authorised. ⚠️ **Datum for that
   decision:** `s2_lab_lib.ensure_repo` creates `Sayood/tanitad-ph0-aug120` with `private=True`, so
   the target is a *private* dataset. ⚠️ **This is now the ONLY thing standing between the unified
   corpus and durability** — it lives on one disk, which is the stranding failure `CLAUDE.md` rule 3
   exists to prevent.
3. ✅ **CLOSED — the coverage test is a hard pass**, 12 passed / 0 xfailed, with no marker edited.
4. ⚠️ **`COLAB_CLI_MCP.md` §8's *"spell T4 EXACTLY (else it falls back to A100)"* does not hold on
   this account** — A100 is rejected for entitlement. Owner: the colab-tooling package. **Unchanged.**
5. ✅ **CLOSED — the 86's `rle_rows` are no longer C85-flattened**; they were re-detected by the v2
   extraction, which encodes rows correctly (the v2 leg was the positive control in §2 and spans all
   179 frame rows). The `row = start // W` interim decode rule is no longer needed for this cohort.
6. ⚠️ **NEW, and it is an ENVIRONMENT escalation, not a code one — this box segfaults on
   `import pyarrow.dataset` when torch/sam3 are already loaded.** §5.6. It is worked around here by
   a load-order shim in the driver's child environment, which is correct for this package but is a
   **latent trap for any future dev-box job that mixes torch and parquet** — and it fails with
   0xC0000005 and no traceback, so the next person will not recognise it. Owner: whoever owns the
   dev-box environment; the durable fix is probably a pinned pyarrow/torch pair.

---

## 8. Deliverable manifest

⚠️ Everything is **in the repo and staged** on `agent/arch-inf-20260803` except the rows marked.

| artifact | where |
|---|---|
| the shared C77 predicate + local transport | `colab/s2_lab_lib.py` (`census_records`, `content_census_local`) |
| ⭐ the homogeneity pin, 12 tests (data **and** detector) | `stack/tests/test_perception_floor_homogeneity.py` |
| this report | `…/incoming/2026-08-17-perception-floor-unify/PERCEPTION_FLOOR_UNIFY.md` |
| floor diagnosis from the records | `code/f0_floor_probe.py` → `raw/f0_floor_probe.json` |
| ⭐ the ONE extraction path, host-agnostic (`F86_*`) | `code/f1_run86.py` — ran unmodified on Colab's design and on the 4060 |
| the Colab chunk driver (remote transport) | `code/f2_drive86.py` |
| ⭐ **the LOCAL-GPU driver** (population + resume + proof-gate, no extraction) | `code/f7_drive86_local.py` |
| ⭐ **the run record** — census, proof-gate, peak GPU, per-call log | `raw/f7_run86_local.json` |
| ⭐ **torch-protection evidence** (both interpreters, real CUDA conv2d) | `code/f8_env_probe.py` → `raw/f7_env_local_gpu.json` |
| homogeneity manifest builder | `code/f3_homogeneity.py` |
| ⭐ per-clip floor · schema · **md5** · counts · liveness | `raw/floor_homogeneity_manifest.json` (**now 201 rows, residual 0**) |
| single-leg input assembly for the re-fuse | `code/f4_build_inputs_unified.py` |
| re-fuse content verification + the two predictions | `code/f5_refuse_delta.py` → `raw/f5_refuse_delta.json`, `raw/fused_aug120_v3_index.jsonl` |
| ⚠️ **the 86 re-detected records** | ✅ **EXIST — 86 JSON + `_md5.json` + `_run_manifest.json`.** **DEV BOX ONLY**: `<scratchpad>/floor86/sam3_86_v2/` (~14.4 MB; the corpus has never been in git). Escalation 2 |
| ⚠️ acquire-and-run helper (Colab) | `code/f6_acquire_and_run.sh` — unused on this path |
| ⚠️ the banked 115 v2 corpus | **DEV BOX + HF** `Sayood/tanitad-ph0-aug120 → sam3_backfill_v2/` (pre-existing, **unmodified** by this package) |
| ⚠️ **the unified re-fused corpus (201)** | **DEV BOX ONLY** — `<scratchpad>/floor86/work_v3/{sam3_refuse,fused_A2,fused_aug120_v2}` |
| ⚠️ the sam3 run venv (isolated, reversible) | **DEV BOX ONLY** — `C:/Users/Admin/venvs/sam3run`; `rm -rf` restores the box |

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

# --- the re-run, ROUTE A: a LOCAL CUDA GPU (what actually ran) ---------------- #
# ⭐ same extraction file as route B; the driver only supplies host + resume.
#    Content-resumable, safe to re-run; refuses the batch unless clip 1 is LIVE.
PYTHONUTF8=1 OMP_NUM_THREADS=6 <venv>/python.exe -u code/f7_drive86_local.py \
       --aug120 <aug120> --v2-dir <sam3_v2> --bank <bank> --root <work86>
# ⚠️ the venv needs sam3's closure. Installed ISOLATED so the load-bearing
#    tanitad venv is never written to (§5.5 / raw/f7_env_local_gpu.json):
#      python -m venv <venv>; echo <tanitad>/Lib/site-packages > \
#        <venv>/Lib/site-packages/_zz_tanitad_base.pth      # torch shared BY PATH
#      <venv>/python -m pip install --no-deps <sam3 clone> timm open_clip_torch \
#        ftfy==6.1.1 wcwidth regex iopath portalocker safetensors einops \
#        pycocotools triton-windows      # ⛔ --no-deps on ALL of them
python code/f8_env_probe.py --out raw/f7_env_local_gpu.json   # proves torch intact

# --- the re-run, ROUTE B: a Colab T4 (unchanged, still valid) ----------------- #
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

| suite | brief's baseline | after the instrument landed | ⭐ **after the 86 landed** |
|---|---|---|---|
| `stack` | **3770** passed / 0 failed | **3781** passed / 0 failed / 7 skipped / **3 xfailed** (369 s) | **3782 passed / 0 failed / 7 skipped / 2 xfailed** (359 s) |
| `taniteval` | **1092** / 0 failed | **1092** passed / 0 failed | **1092 passed / 0 failed** (103 s) — untouched, re-run to prove it |

⭐ **+11 passed and +1 xfailed = exactly the 12 new tests, and the arithmetic closes:**
3770 + 12 = 3782 = 3781 passed + 1 xfailed.

⭐ **AND THE SECOND PAIRING IS THE ONE THAT MATTERS HERE — IT PROVES THE RE-RUN MOVED EXACTLY ONE
TEST AND NOTHING ELSE.** 3781 passed + 3 xfailed = **3784**; 3782 passed + 2 xfailed = **3784**.
The total is unchanged and the shift is **+1 passed / −1 xfailed**, which is the coverage test and
only the coverage test. With several agents live in one working tree that is the check worth
having: an unpaired total would silently attribute their work to this package.

⚠️ The **named interpreter** matters and is not optional here: a bare `pytest` resolves to a
different venv on this shell and reports *"191 errors during collection"* **while exiting 0**.

⚠️ The **named interpreter** matters: a bare `pytest` resolves to a different venv on this shell and
reports *"191 errors during collection"* while exiting 0. And without `PYTHONUTF8=1` this shell
reports four failures that do not exist (C84).
