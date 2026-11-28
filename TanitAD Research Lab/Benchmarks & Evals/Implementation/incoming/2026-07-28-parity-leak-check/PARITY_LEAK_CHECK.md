# The parity val split is CLEAN — settled BY CONTENT, not by `episode_id`

**Agent:** `parity-leak-check` · **Date:** 2026-07-28 · **Repo HEAD:** `a186204`
**Evidence class:** `MEASURED (ours)` — sha256 of raw bytes, both hosts, read-only on every cache.
🔒 Gated-confidential: counts and hashes only. No clip UUIDs appear in any artifact.

---

## 0. Headline

> **`physicalai-val-0c5f7dac3b11` × `physicalai-train-e438721ae894`: overlap = 0.**
> Not one of the **40 deployed evaluation episodes** — the substrate every open-loop and
> closed-loop number this program has published is scored on — shares a single byte-identical
> `poses` tensor, `actions` tensor or `frames_u8` tensor with any of the parity corpus's
> **2,376 training episodes**. And **not one of its 7,964 frames appears anywhere among the
> corpus's 472,627 training frames**, so there is no sub-window or shifted leak either.
> Split-wide (all **600** registered val episodes, poses-level): **0 / 600**.

**The registry's claim is upgraded.** `MODEL_REGISTRY.md` §0.3 says *"40 episodes → 881 windows,
**episode-disjoint** from train ✅"*. That was an `episode_id` claim. It is now **content-verified**
at both the pose and the sensor level. **v1's line, the fan work, Bar A, T3, E-GOAL-1→4, the arm
panel and the small validation now running are NOT affected by a val leak.**

**And the instrument is proven able to find a leak**, on the same corpus, the same code path and the
same train cache: run against the known-leaky sibling `physicalai-val-f1b378f295ae` it returns
**62 of 80 by `poses_sha256` AND 62 of 80 by `frames_sha256`** — the first *content* confirmation of
a number the registry has carried since 07-23 as an `episode_id` intersection.

**The `ci.py` question is also settled: DISPLAY-ONLY.** Every estimator function in
`/root/taniteval/taniteval/ci.py` (md5 **`ef925f06…`**, asserted off the loaded module object) is
**byte-identical in source** to HEAD's (`c92618a0…`). Across 28 driven cases: **0 with any
statistical field differing. Every published v3 interval stands as a number.**

---

## 1. Pre-registration — fixed before any intersection was computed

Written into `code/intersect_pai.py` and echoed into every result JSON's `pre_registration` block
before the first run.

| outcome | consequence, committed in advance |
|---|---|
| **overlap = 0 ⇒ CLEAN** | The parity split is sound by content. The registry's `episode_id` claim is upgraded to content-verified. Say so plainly. |
| **overlap > 0 ⇒ LEAKED** | Report the **count**, the **identities**, and **the fraction of each affected metric's mass** the leaked episodes carry. ⛔ Not softened into "minor". |

**Stated in advance, what a leak would and would not invalidate:**

* a val leak invalidates **generalisation** claims — held-out ADE, the CV/CTRV floors, every A/B
  contrast read as "on unseen data";
* it does **not** invalidate **training facts** — parameter counts, step counts, loss curves,
  the parity key itself;
* it hits **arms differently**, because an arm is only contaminated by episodes *it* trained on.
  All flagship/REF-B/REF-C arms share `e438721ae894`, so a leak would have been common-mode for
  them; the frozen-encoder REF-A arms and any 320-ep subset arm would have needed a separate check.

⚠️ And pre-registered: **a content-disjoint sub-split is a RE-SELECTION.** Had this come back
leaked, the clean subset could have been quoted as a *diagnostic* but **could never be registered as
parity** — parity is the 40-episode split as built, and re-selecting breaks cross-arm comparability
with every number already published. That escape hatch was closed before the probe ran. It was not
needed.

---

## 2. Method — the sibling's, re-pointed at PhysicalAI

Reused from `…/incoming/2026-07-27-anchor-settlement/` (C43), not reinvented.

**Seven hash families per episode**, six of them identifying:

| family | over | why |
|---|---|---|
| `poses_sha256` | `poses` [T,4] float32 raw bytes | the brief's named **primary** |
| `poses_xy_sha256` | the (x, y) columns | robust to any heading-label protocol change |
| `poses_yaw_sha256` | the yaw column | isolates the one column a repair could rewrite |
| `poses_v_sha256` | the speed column | ditto |
| `actions_sha256` | `actions` [T,2] float32 | independent derivation |
| **`frames_sha256`** | **`frames_u8` [T,9,256,256] uint8** | **the raw sensor bytes — independent of every label protocol** |
| `maneuvers_sha256` | `maneuvers` [T] int64 | ⚠️ a **categorical label**, NOT identifying — see §5.3 |

Plus **`frame_digests`** — sha1[:16] of every individual frame — which catches the leak mode a
whole-tensor hash structurally cannot see: a val episode that is a **sub-window or shifted copy** of
a train clip. 7,964 val frames were checked against an index of all **472,627** train frames.

⛔ **Names are never the evidence.** `episode_id`, the `ep_XXXXX.pt` filename and the tag index are
recorded only as cross-checks *that could have disagreed*. They did — see §5.1, §5.2.

**Substrates.** Read-only throughout. pod1 (training) and pod2 (small validation) were never touched.

| set | host | path | n | frames | size |
|---|---|---|---|---|---|
| **VAL — deployed** (what every number is scored on) | `tanitad-eval` | `/root/valdata/physicalai-val-0c5f7dac3b11` | 40 | 7,964 | 4.7 GB |
| **TRAIN — the parity corpus** | `tanitad-pod3` | `/workspace/pai_epcache/physicalai-train-e438721ae894` | 2,376 | 472,627 | 278.8 GB |
| VAL — registered split, poses-only view | `tanitad-pod3` | `/workspace/s3parity/views/physicalai-val-0c5f7dac3b11` | 600 | — | 4.6 MB |
| **known-positive control** | `tanitad-pod3` | `/workspace/pai_epcache/physicalai-val-f1b378f295ae` | 80 | 15,906 | 9.4 GB |

`/root/valdata/physicalai-val-0c5f7dac3b11` is not a guess about the substrate: it is **hard-coded in
every `taniteval` entrypoint** — `runner.py:43`, `bench.py:669`, `closedloop.py:1066`,
`hierarchy.py:1138`, `pathspeed.py:447`, `planner_p2.py:607`, `planning.py:267`, `efficiency.py:55`,
`refc_rerank.py:85`, `strategic_probes.py:545`, `registry.py:280`, `flagship_overlay.py:42`.

All 278.8 GB of the train corpus was hashed frame-by-frame in **180 s** (12 workers, pod3).

---

## 3. Controls — because a leak check that cannot find a leak is worse than none

### 3.1 The known-positive: a real 62-episode leak, recovered by content ⭐

The strongest control available was already in the program: `physicalai-val-f1b378f295ae`, which
`MODEL_REGISTRY.md` (line ~2011) records as **62 of 79 populated episodes (78.5 %) IN the parity
train corpus**, measured by **`episode_id` intersection**. Run through the *identical* code path,
against the *identical* train fingerprints:

| family | overlap found |
|---|---|
| `poses_sha256` | **62 / 80** |
| `poses_xy_sha256` | **62** |
| `poses_yaw_sha256` | **62** |
| `poses_v_sha256` | **62** |
| `actions_sha256` | **62** |
| **`frames_sha256`** | **62** |
| `maneuvers_sha256` *(label, not identifying)* | 66 |
| `episode_id` *(name-derived cross-check)* | 62 |

**All six identifying families agree exactly at 62**, and `poses_bitwise_equal` is `true` for every
pair. ⇒ **The instrument detects a real leak of exactly this kind, at exactly this scale, on exactly
this corpus and train cache.** A 0 from it is therefore informative.

⭐ **Two by-products.** (a) The 62/79 figure is now **content-verified** for the first time — it was
an `episode_id` claim, of exactly the class C43 says is not evidence, and it happens to have been
right. (b) A small correction: the cache holds **80** `ep_*.pt` files carrying **80 distinct**
poses/frames hashes but only **79 distinct `episode_id`s** — two files share an id. The honest
figure is **62 / 80 = 77.5 %**, not 62/79 = 78.5 %.

### 3.2 SELF — can the matcher match?

Val intersected with itself through the same function: **40/40 on all seven families** (and 600/600,
80/80 on the other substrates). Passes.

### 3.3 SPIKE — end-to-end detection at real scale

Three *real* val records injected into an in-memory **copy** of the 2,376-episode train set (no cache
touched). Required: exactly 3, with the right identities.
**Found 3 by `poses_sha256` and 3 by `frames_sha256`; identities `ep_00000, ep_00001, ep_00002`.** Passes.

⚠️ **Reported honestly:** on the *leaky* control the same assertion prints `passes: false`, returning
**63**. That is not a failure, it is the assertion being written for a clean baseline. The leaky
split already contains 62 real leaks; of the three spiked tags, `ep_00001` and `ep_00002` are
**already** among the 62 and `ep_00000` is not — so 62 + 1 = **63**, exactly. The number the naive
check flags is the number that confirms the instrument.

### 3.4 MUTANT — is the instrument matching on something trivial?

Val intersected against a copy of itself with **one bit of one float** of the pose tensor flipped.
Required 0. **Got 0** on `poses_sha256` and `poses_xy_sha256`. Passes.

### 3.5 View fidelity — is the derived artefact usable as evidence?

The 600-episode registered split no longer exists as a full cache on any reachable host; only its
**poses-only view** survives. A view is a *derived* artefact, so it was checked rather than trusted:
**2,376 / 2,376 tags in the train view carry a `poses_sha256` identical to the real cache**. And the
40 deployed val episodes are a **content-subset of the 600** — 40/40 found, and the tag names line up
too — so the split-wide poses result covers the deployed substrate as well, and agrees with it.

---

## 4. The verdict

`raw/ANSWER_0c5f40_x_train_FULL.json` · `raw/SPLITWIDE_0c5f600_x_train_POSES.json`

### 4.1 The deployed 40 — pose level AND sensor level

| family | overlap with the 2,376-episode parity train corpus |
|---|---|
| `poses_sha256` (primary) | **0** |
| `poses_xy_sha256` | **0** |
| `poses_yaw_sha256` | **0** |
| `poses_v_sha256` | **0** |
| `actions_sha256` | **0** |
| **`frames_sha256`** (raw sensor) | **0** |
| `maneuvers_sha256` *(categorical label — see §5.3)* | 10 |

**All six identifying families agree at 0.** The seventh is a low-entropy label and is the one
disagreement the brief asked to be treated as a finding; §5.3 settles it — the 10 are label
collisions carrying **zero** identifying-family match.

### 4.2 Frame-level containment — the sub-window leak mode

| | |
|---|---|
| val episodes sharing **any** frame with **any** train episode | **0 of 40** |
| max fraction of a val episode's frames found anywhere in train | **0.000** |
| val frames checked | 7,964 |
| train frames indexed | 472,627 |

No partial overlap, no shifted copy, no re-cut window. This is a strictly stronger statement than
whole-tensor disjointness and it is the one the comma case (`C43`) would have needed.

### 4.3 Split-wide — all 600 registered val episodes

Poses-level only (no frames survive for the 570 non-deployed episodes): **0 / 600** on all four pose
families, against 2,376 train episodes.

### 4.4 Within-cache duplicates

`poses_sha256`: **40/40**, **600/600**, **2,376/2,376** distinct. `frames_sha256`: **2,376/2,376**
distinct. No episode is cached twice under two names in any of these caches.

---

## 5. What else the instrument found

### 5.1 ⭐ 40/40 and 600/600 filename overlap, with 0 real overlap

Every deployed val filename (`ep_00000 … ep_00039`) also exists in the train cache; at split scale
**all 600 do**. Content overlap is **0** in both. The program's "600/600 filename overlap with 0/600
real overlap" is reproduced *in situ*, on the very caches under audit. In the known-positive control
the same disconnect runs the other way: val `ep_00001` is train `ep_00000`, val `ep_00003` is train
`ep_00009`. **The filename carries no information about identity in either direction.**

### 5.2 ⚠️ `episode_id` produces 20 FALSE POSITIVES at split scale — and is not even a key

**20 of the 600 registered val episodes share an `episode_id` with a train episode. None of them
shares any content.** Had anyone "verified" the registered split by `episode_id`, they would have
reported a 20-episode leak that does not exist.

And inside the train corpus alone: **2,342 distinct `episode_id`s for 2,376 episodes — 33 ids reused
across 67 episodes, maximum multiplicity 3.** An id that is not a key inside one cache cannot
establish disjointness between two.

⇒ In C43 `episode_id` corroborated a real leak. Here it manufactures 20 phantom ones and fails
uniqueness. **Both directions of error are now measured. `episode_id` is not evidence — full stop.**
*(For the deployed 40 it happens to agree with content at 0.)*

### 5.3 The one hash family that disagreed — and why it is not a leak

`maneuvers_sha256` reported 10 "matches". `maneuvers` is a `[T] int64` **categorical label
sequence**, not content, and it has nowhere near the entropy to identify an episode:

| family | distinct values over the 2,376 train episodes | largest identical group |
|---|---|---|
| `poses_sha256` | **2,376** | 1 |
| `frames_sha256` | **2,376** | 1 |
| `maneuvers_sha256` | **1,883** | **450** |

All 10 colliding val episodes collide with the **same 450-episode modal group** (an all-one-maneuver
sequence), and for **0 of the 10** does any identifying family also match. ⇒ Not a leak; a
demonstration that **a label hash is not a content hash**, which is precisely why the sibling's
protocol requires agreement across *independent* families rather than a single one.

### 5.4 ⚠️ Two near-duplicate pairs, inside single caches — flagged, not a val leak

* **Train:** `ep_01036` and `ep_02371` are **bit-identical in `poses_xy` AND `poses_v`** but differ
  in yaw and in pixels.
* **Registered val 600:** `ep_00247` and `ep_00552`, the same shape.

Identical (x, y, speed) with different heading and different frames is not what two independent
clips look like. Most likely the same source clip cached twice under different heading/rig handling.
It does **not** touch the verdict (both are within-cache, and neither is among the deployed 40), but
it is worth an owner: a duplicated trajectory inside the train corpus mildly re-weights it, and a
duplicated one inside a val split would inflate an episode-cluster bootstrap's effective *n*.

---

## 6. `ci.py` — DISPLAY-ONLY. The intervals stand.

`raw/ci_equivalence.json` · `raw/ci_diff.txt`

**md5 of the file ACTUALLY LOADED**, read back off the imported module object (the standing rule C44
leaves behind), not the one intended:

| | module `__file__` | md5 |
|---|---|---|
| the one on the import path | `/root/taniteval/taniteval/ci.py` | **`ef925f06febd20a99f5901491fcf75cb`** |
| HEAD | `/workspace/TanitAD-head/taniteval/taniteval/ci.py` | **`c92618a02b36f8191a581fb74a491a8d`** |

`/root/idm2/idm2_lib.py:19` and `/workspace/idm3/idm3_a0.py` both `sys.path.insert(0,
"/root/taniteval")` **unconditionally**, before `from taniteval import ci` — so every published v3
interval did come through the `ef925f06` file. **What came through it is the same estimator.**

**Source identity of the estimator core** (`inspect.getsource`, both modules imported side-by-side):
`overlapping_holdout_se`, `episode_index`, `_draws`, `_red_mean`, `_red_rms`, `_red_median`,
`_quantile_reducer`, `resolve_reducer`, `_reducer_name`, `episode_cluster_bootstrap`,
`bootstrap_metrics` — **all eleven byte-identical**. Exactly one function differs:
`paired_episode_cluster_bootstrap`.

**Driven comparison**, 28 cases (24 randomised over reducer / n_episodes / effect size / seed, plus
4 constructed degenerate cases) with identical inputs and identical seeds:

| | |
|---|---|
| cases where **any** statistical field differs (`separated`, `p_delta_gt0`, `reducer`, `n_windows`, `n_episodes`, `n_boot`, `estimator`) | **0 / 28** |
| cases **bitwise identical in every field** | 25 / 28 |
| cases where HEAD escalated the rendering | 3 — `DEGENERATE_1e-9`, `DEGENERATE_1e-14`, `NEAR_5e-5` |
| **non-escalated cases with any numeric difference** | **0** |
| unpaired `episode_cluster_bootstrap`, 8 cases | **all identical** |

The single changed function replaces `round(x, 4)` with `round(x, dp)` and adds `display_dp` /
`display_note` / `degenerate` keys. `dp` **is 4** unless 4 dp would print an interval that
*contradicts* `separated` — and `separated` is computed on the **unrounded** bounds in **both**
files. ⇒ **No published v3 interval changes. The intervals stand.**

⚠️ *One methodological note, recorded rather than hidden:* a naive "do they agree at 4 dp"
comparator flags one case, because it **double-rounds** — `round(round(x, 5), 4)` need not equal
`round(x, 4)` at a tie. The mismatch is in the comparator, not the estimator. The verdict is keyed on
statistical-field identity plus HEAD's own `display_dp` stamp instead.

### 6.1 But the display defect DID reach 18 published records

Scanning **every committed JSON at HEAD** — 2,161 files, 12,621 interval records — for the signature
(`separated: true` beside a printed interval that does not exclude zero):

**18 records, in 9 files.** These are records where a reader, or an automated gate keying on
`separated`, sees a verdict beside `0.0 [0.0, 0.0]`.

| n | file | node |
|---:|---|---|
| 5 | `…/2026-07-26-idm-v2/compare.json` | `paired_vs_B0` |
| 3 | `…/2026-07-26-e1c-heldout-gated-clsft/e1c_evaluator_smoke_result.json` | `points` |
| 2 | `…/2026-07-26-idm-v2/compare.json` | `paired_vs_A0` |
| 2 | `…/2026-07-27-vtband-decision/raw/legA_v5config_structural.json` | `options` |
| 2 | `…/2026-07-26-wheelbase-impact/tier2b_along_cross.json` | `paired_vs_shipped` |
| 1 | `…/2026-07-25-closedloop-horizon-and-shift/e1a_horizon_heldout44.json` | `paired_common_start` |
| 1 | `…/2026-07-28-tactical-action-input/artifacts/blockA/blockA_full_panel_20arm.json` | `paired` |
| 2 | `…/2026-07-27-registry-repair/raw/rerender_*.json` | `committed_rendering` *(deliberate exhibits of the old rendering)* |

⇒ **Not a statistics problem — a re-render problem.** The values are correct; the *printed* records
are misleading and none carries HEAD's `degenerate` marker. **Escalation, §8.**

---

## 7. What this does NOT establish — stated plainly

1. **Only PhysicalAI, only these two caches.** The comma and cosmos val caches on `tanitad-eval`
   (`comma2k19-val-76b6e94a97a1`, `cosmos-val-*`) were **not** checked against their training
   corpora here. C43 settled the comma anchor's 22-episode substrate; the rest is open.
2. **The 570 non-deployed val episodes are checked at pose level only.** Their source cache no
   longer exists on any reachable host. For the 40 that every number actually uses, both levels
   were checked and both read 0.
3. **REF-A's frozen-feature caches are a different substrate.** `/root/featcache/{dinov2,ijepa}/…`
   are *derived from* the episodes checked here, so they inherit this result — but the separately
   flagged **REF-A I-JEPA ~80 % leak vs its own 320-episode subset** is a different overlap and
   remains **UNVERIFIED**. This result does not touch it.
4. **This is a leak check, not a distribution check.** Content-disjoint does not mean the val split
   is representative, hard, or drawn from unseen scenes/routes. Two episodes from the same recording
   session are content-disjoint and still correlated. Nothing here speaks to that.
5. **`ci.py`**: 28 driven cases, not a proof. It is a strong empirical equivalence backed by
   byte-identical source for 11 of 12 functions and a line-level diff of the twelfth.

---

## 8. Escalations — these need a decision, not a README line

1. **⭐ `MODEL_REGISTRY.md` §0.3 should be upgraded** from *"episode-disjoint from train ✅"* to
   *"content-verified disjoint (sha256 of raw `poses` and `frames_u8`; 0/40 and 0/600;
   2026-07-28)"*, citing `raw/ANSWER_0c5f40_x_train_FULL.json`. **I did not edit the registry** —
   agents do not, and a claim upgrade this load-bearing should be made by its owner.
2. **The 18 mis-rendered interval records need a re-render**, not a re-measurement. HEAD's `ci.py`
   already produces the fix; the sweep is mechanical.
3. **The `ef925f06` `ci.py` is still first on the import path** for `idm2_lib.py` / `idm3_a0.py`.
   It is statistically equivalent, so nothing is urgent — but it will *stay* equivalent only until
   the next HEAD change, and the stale-import guard reports `ok: true` because it probes capability,
   not identity (C44).
4. **⚠️ `stack/` is NOT green in the current working tree — and it is not this agent's doing.**
   `pytest -q` in `stack/`: **1 failed, 1,575 passed, 12 skipped** (briefed expectation: 1,576 / 12).
   The failure is `tests/test_heldout_gate.py::test_the_admitted_component_set_is_PINNED_at_the_first_probe`,
   raised from `taniteval/taniteval/pseudosim.py:874`. **Cause proven, not guessed:** an isolated
   tree with HEAD's `pseudosim.py` and everything else identical runs the same file **18/18 green**;
   with the working tree's version, 1 fails. A concurrent sibling (`2026-07-28-closedloop-control-suite`)
   has `taniteval/taniteval/pseudosim.py` modified — adding a `floor_frac_max` admissibility gate and
   setting `comfort`'s weight to `0.0` — which makes the pinned component set unreachable. **A
   `taniteval` edit is breaking a `stack` test; that owner needs to update the pin or the gate.**
   `taniteval/`: **697 passed, 0 failed, 0 skipped** (briefed 663 — the suite has grown; no new skips).
5. **Two intra-cache near-duplicate pairs** (§5.4) want an owner.

---

## 9. Reproduce

```bash
# pod3 (free, 96 cores) — read-only; 278.8 GB hashed in 180 s at 12 workers
bash code/run_fingerprints_pod3.sh

# tanitad-eval — the 40 deployed val episodes
python3 code/fingerprint_pai_cache.py \
  --cache /root/valdata/physicalai-val-0c5f7dac3b11 \
  --out fp_val_0c5f_deployed40_full.jsonl --mode full --workers 8
# relay the 0.37 MB fingerprint file to pod3 and md5-verify (8ff53d7affea625b3edce9592b06867f)

python3 code/intersect_pai.py --val fp_val_0c5f_deployed40_full.jsonl \
    --train fp_train_e4387_full.jsonl --out ANSWER_0c5f40_x_train_FULL.json
python3 code/intersect_pai.py --val fp_val_f1b378f295ae_full.jsonl \
    --train fp_train_e4387_full.jsonl --out KNOWNPOSITIVE_f1b378_x_train.json   # must read 62
python3 code/aux_checks.py .
python3 code/ci_equivalence.py                       # on tanitad-eval
python3 code/scan_degenerate_records.py <repo> degenerate_records_scan.json
```

The staged `raw/hashes_*.json` let anyone re-derive **every** count in this document without
re-reading a byte of the corpus. They are hashes only — `poses_b64` and `frame_digests` are stripped
because they are raw bytes of a gated corpus (§ manifest).
